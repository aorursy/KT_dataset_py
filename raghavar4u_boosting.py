%matplotlib inline  

# To make data visualisations display in Jupyter Notebooks 

import numpy as np   # linear algebra

import pandas as pd  # Data processing, Input & Output load

import matplotlib.pyplot as plt # Visuvalization & plotting

import datetime  



import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

from xgboost.sklearn import XGBClassifier # Extrame GB

from xgboost import plot_importance ## Plotting Importance Variables 



import joblib  #Joblib is a set of tools to provide lightweight pipelining in Python (Avoid computing twice the same thing)



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

                                    # GridSearchCV - Implements a “fit” and a “score” method

                                    # train_test_split - Split arrays or matrices into random train and test subsets

                                    # cross_val_score - Evaluate a score by cross-validation

from sklearn.metrics import mean_squared_error

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, make_scorer, accuracy_score, roc_curve, confusion_matrix, classification_report                                    # Differnt metrics to evaluate the model 

import pandas_profiling as pp   # simple and fast exploratory data analysis of a Pandas Datafram



import warnings   # To avoid warning messages in the code run

warnings.filterwarnings("ignore")
def plot_roc_curve(y_train_actual, train_pred_prob, y_test_actual, test_pred_prob, *args):

    '''

    Generate the train & test roc curve

    '''



    AUC_Train = roc_auc_score(y_train_actual, train_pred_prob)

    AUC_Test = roc_auc_score(y_test_actual, test_pred_prob)



    if len(args) == 0:

        print("Train AUC = ", AUC_Train)

        print("Test AUC = ", AUC_Test)

        fpr, tpr, thresholds = roc_curve(y_train_actual, train_pred_prob)

        fpr_tst, tpr_tst, thresholds = roc_curve(y_test_actual, test_pred_prob)

        roc_plot(fpr, tpr, fpr_tst, tpr_tst)



    else:

        AUC_Valid = roc_auc_score(args[0], args[1])

        print("Train AUC = ", AUC_Train)

        print("Test AUC = ", AUC_Test)

        print("Validation AUC = ", AUC_Valid)

        fpr, tpr, thresholds = roc_curve(y_train_actual, train_pred_prob)

        fpr_tst, tpr_tst, thresholds = roc_curve(y_test_actual, test_pred_prob)

        fpr_val, tpr_val, thresholds = roc_curve(args[0], args[1])

        roc_plot(fpr, tpr, fpr_tst, tpr_tst, fpr_val, tpr_val)

def roc_plot(fpr, tpr, fpr_tst, tpr_tst, *args):

    '''

    Generates roc plot

    '''



    fig = plt.plot(fpr, tpr, label='Train')

    fig = plt.plot(fpr_tst, tpr_tst, label='Test')



    if len(args) == 0:

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.0])

        plt.title("ROC curve using ")

        plt.xlabel('False Positive Rate (1 - Specificity)')

        plt.ylabel('True Positive Rate (Sensitivity)')

        plt.legend(loc='lower right')

        plt.grid(True)

        plt.show()



    else:

        fig = plt.plot(args[0], args[1], label='Validation')

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.0])

        plt.title("ROC curve using ")

        plt.xlabel('False Positive Rate (1 - Specificity)')

        plt.ylabel('True Positive Rate (Sensitivity)')

        plt.legend(loc='lower right')

        plt.grid(True)

        plt.show()

# Read-in the dataset

Insurance_Data = pd.read_csv('../input/carInsurance_train.csv')

print('Train Data Shape - ', Insurance_Data.shape)

Insurance_Data.head()
# What type of values are stored in the columns?

Insurance_Data.info()
pp.ProfileReport(Insurance_Data)
# Let's look at some statistical information about our dataframe.

Insurance_Data.describe()
# This is how we can get summary for the categorical data

Insurance_Data.describe(include=np.object) 
Target = 'CarInsurance'

pd.crosstab(Insurance_Data[Target], columns='N', normalize=True)
Insurance_Data.head()
num_cols = Insurance_Data.select_dtypes(include=[np.number]).columns.tolist()

non_num_cols = Insurance_Data.select_dtypes(exclude=[np.number]).columns.tolist()

# Lets drop columns which we will not use

num_cols = Insurance_Data.drop(['Id', 'CarInsurance'],axis=1).select_dtypes(include=[np.number]).columns.tolist()

non_num_cols = Insurance_Data.drop(['CallStart', 'CallEnd'],axis=1).select_dtypes(exclude=[np.number]).columns.tolist()

print('Numeric Columns \n', num_cols)

print('Non-Numeric Columns \n', non_num_cols)
# Lets drop CarLoan, HHInsurance, Default from the numeric columns as these are dummies

num_cols_viz = ['DaysPassed', 'Age', 'NoOfContacts', 'PrevAttempts', 'LastContactDay', 'Balance']



fig, axes = plt.subplots(3,2,sharex=False,sharey=False, figsize=(15,15))

Insurance_Data.loc[:,[Target]+num_cols_viz].boxplot(by=Target, ax=axes,return_type='axes')
non_num_cols_viz = non_num_cols+['CarLoan', 'HHInsurance', 'Default']

fig, axes = plt.subplots(len(non_num_cols_viz),sharex=False,sharey=False, figsize=(15,25))

for i in range(len(non_num_cols_viz)):

    pd.crosstab(Insurance_Data[non_num_cols_viz[i]], Insurance_Data[Target]).plot(kind='bar', stacked=True, grid=False, ax=axes[i])

        
Insurance_Data.isnull().sum()
Insurance_Data_Org = Insurance_Data.copy()
Insurance_Data['Job'].value_counts(dropna=False)
Insurance_Data['Job'] = Insurance_Data['Job'].fillna('None')

Insurance_Data['Job'].isnull().sum()
# Fill missing education with the most common education level by job type

Insurance_Data['Education'].value_counts()



# Create job-education level mode mapping

edu_mode=[]



# What are different Job Types

job_types = Insurance_Data.Job.value_counts().index



# Checking which job is most 

Insurance_Data['Job'].value_counts()
# Now according to the job type we will crate a mapping where the job and mode of education is there.

# It means when there are many people in the managment job then most of them are in which education.

# We can find that in below mapping



for job in job_types:

    mode = Insurance_Data[Insurance_Data.Job==job]['Education'].value_counts().nlargest(1).index

    edu_mode = np.append(edu_mode,mode)

edu_map=pd.Series(edu_mode,index=Insurance_Data.Job.value_counts().index)



edu_map
# Apply the mapping to missing eductaion obs. We will replace education now by jobs value

for j in job_types:

    Insurance_Data.loc[(Insurance_Data['Education'].isnull()) & (Insurance_Data['Job']==j),'Education'] = edu_map.loc[edu_map.index==j][0]



# For those who are not getting mapped we will create a new category as None

Insurance_Data['Education'].fillna('None',inplace=True)

# Fill missing communication with none 

Insurance_Data['Communication'].value_counts(dropna=False)
Insurance_Data['Communication'] = Insurance_Data['Communication'].fillna('None')
# Check for missing value in Outcome

Insurance_Data['Outcome'].value_counts(dropna=False)
# Fill missing outcome as not in previous campaign, we are adding one category to Outcome

# We will add category if the value of DaysPassed is -1

# Can you do it other ways.. yes this is one way of doing you can do it other ways also.



Insurance_Data.loc[Insurance_Data['DaysPassed']==-1,'Outcome']='NoPrev'

Insurance_Data['Outcome'].value_counts(dropna=False)
# Check if we have any missing values left

Insurance_Data.isnull().sum()
Insurance_Data_num = Insurance_Data[num_cols+['Id', 'CarInsurance']]
# Categorical columns data

Insurance_Data_cat = Insurance_Data[non_num_cols]

non_num_cols
# Create dummies

Insurance_Data_cat_dummies = pd.get_dummies(Insurance_Data_cat)

print(Insurance_Data_cat_dummies.shape)

Insurance_Data_cat_dummies.head()
Insurance_Data_final = pd.concat([Insurance_Data_num, Insurance_Data_cat_dummies], axis=1)

print(Insurance_Data_final.shape)

Insurance_Data_final.head()
# Checking if there are missing values before we run model

Insurance_Data_final.isnull().sum(axis = 0)
train_df = Insurance_Data_final.drop(['Id', 'CarInsurance'], axis=1)

train_label = Insurance_Data_final['CarInsurance']
#random_state is the seed used by the random number generator. It can be any integer.

# Train test split

X_train, X_test, y_train, y_test = train_test_split(train_df, train_label, train_size=0.7 , random_state=100)
print('Train shape - ', X_train.shape)

print('Test shape  - ', X_test.shape)
# Define Model parameters to tune

model_parameters = { 

        'n_estimators':[10, 50, 100, 200, 500],

        'max_depth': [3, 5, 10],

        'min_samples_leaf': [np.random.randint(1,10)]

                  }
# Gridsearch the parameters to find the best parameters. Using L2 penalty

model = XGBClassifier()

gscv = GridSearchCV(estimator=model, 

                    param_grid=model_parameters, 

                    cv=5, 

                    verbose=1, 

                    n_jobs=-1,

                    scoring='f1')



gscv.fit(X_train, y_train)  ## Model building 
print('The best parameter are -', gscv.best_params_)
# Re-fit the model with the best parameters

final_mod = XGBClassifier(max_depth = 3, min_samples_leaf = 4, n_estimators = 500)

final_mod.fit(X_train, y_train)
# Prediction

train_pred = final_mod.predict(X_train)

test_pred = final_mod.predict(X_test)
print('Classification report for train data is : \n',

      classification_report(y_train, train_pred))

print('Classification report for test data is : \n',

      classification_report(y_test, test_pred))

# Save the variables used in the model as it will be required in future for new datasets prediction

final_mod.variables = X_train.columns
joblib.dump(final_mod, 'best_model.joblib')
# Generate ROC

plt.subplots(figsize=(10, 5))

train_prob = final_mod.predict_proba(X_train)[:, 1]

test_prob = final_mod.predict_proba(X_test)[:, 1]



plot_roc_curve(y_train, train_prob,

               y_test, test_prob)
# make predictions for test data

y_pred = final_mod.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Load the saved model



best_model = joblib.load('best_model.joblib')
# Load the test data

Insurance_test = pd.read_csv('../input/carInsurance_test.csv')

print('Test Data Shape  - ', Insurance_test.shape)

Insurance_test.head()
# Handle missing values on the test data

# The function takes the dataframe and does the same preprocessing that was done for train data



def handle_missing_values(df):

    #Job 

    df['Job'] = df['Job'].fillna('None')

    

    #Education

    # Apply the mapping to missing eductaion obs. We will replace education now by jobs value

    for j in job_types:

        df.loc[(df['Education'].isnull()) & (df['Job']==j),'Education'] = edu_map.loc[edu_map.index==j][0]



    # For those who are not getting mapped we will create a new category as None

    df['Education'] = df['Education'].fillna('None')

    

    #Communication

    df['Communication'] = df['Communication'].fillna('None')

    

    #Outcome

    df.loc[df['DaysPassed']==-1,'Outcome']='NoPrev'

    

    return df

Insurance_test_Org = Insurance_test.copy()
# Handle the missing values the same we had done for Train

Insurance_test = handle_missing_values(Insurance_test)
Insurance_test.isnull().sum()
# Convert Categorical to dummies

dummy_cols = pd.get_dummies(Insurance_test[non_num_cols])

dummy_cols.head()
# Append the columns

new_data = pd.concat([Insurance_test[num_cols], dummy_cols], axis=1)

print(new_data.shape)

new_data.head()
# Check if all the variables of train are present in test

# Variables in model

best_model.variables
# Variables missing in test data. This happens sometimes because of some categories not present in the new data

vars_missing = list(set(best_model.variables) - set(new_data.columns))

vars_missing
# Create the missing columns in the dataset and fill them with 0

# This will create columns bonly if there are missing values

for i in vars_missing:

    new_data[i] = 0

    

print(new_data.shape)

new_data.head()
# Get the new dataset in the same order of the variables used in train

new_data_final = new_data[best_model.variables]

new_data.head()
# Predict on the new data

new_data_final['Predicted'] = best_model.predict(new_data_final)

new_data_final.head()
# Export the results

new_data_final.to_csv('Predicted.csv', index=False)
# Define Model parameters to tune

model_parameters = {

        'n_estimators': [10, 50, 100, 200, 500],

        'max_depth': [3, 5, 10, None],

        'min_samples_leaf': [np.random.randint(1,10)]  

}
# Gridsearch the parameters to find the best parameters.

model = GradientBoostingClassifier(random_state=10)

## random_state  -- The random number seed so that same random numbers are generated every time.



gscv_GBM = GridSearchCV(estimator=model, 

                    param_grid=model_parameters, 

                    cv=5, 

                    verbose=1, 

                    n_jobs=-1,

                    scoring='f1')



gscv_GBM.fit(X_train, y_train)
print('The best parameter are -', gscv_GBM.best_params_)
# Re-fit the model with the best parameters

final_mod_GBM = GradientBoostingClassifier(max_depth = 10, min_samples_leaf = 5, n_estimators = 100)

final_mod_GBM.fit(X_train, y_train)

# Prediction

train_pred = final_mod_GBM.predict(X_train)

test_pred = final_mod_GBM.predict(X_test)



print('Classification report for train data is : \n',

      classification_report(y_train, train_pred))

print('Classification report for test data is : \n',

      classification_report(y_test, test_pred))
joblib.dump(final_mod_GBM, 'best_model_GBM.joblib')
# Generate ROC

plt.subplots(figsize=(10, 5))

train_prob = final_mod_GBM.predict_proba(X_train)[:, 1]

test_prob = final_mod_GBM.predict_proba(X_test)[:, 1]



plot_roc_curve(y_train, train_prob,

               y_test, test_prob)
# make predictions for test data

y_pred = final_mod_GBM.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Get the new dataset in the same order of the variables used in train

new_data_final = new_data[best_model.variables]

new_data.head()

# Predict on the new data

new_data_final['Predicted'] = best_model.predict(new_data_final)

new_data_final.head()
# Export the results

new_data_final.to_csv('Predicted_GBM.csv', index=False)