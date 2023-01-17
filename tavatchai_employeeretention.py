# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# NumPy for numerical computing

import numpy as np



# Pandas for DataFrames

import pandas as pd

pd.set_option('display.max_columns', 100)



# Matplotlib for visualization

from matplotlib import pyplot as plt

# display plots in the notebook

%matplotlib inline 



# Seaborn for easier visualization

import seaborn as sns

sns.set_style('darkgrid')



# (Optional) Suppress FutureWarning

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load employee data from CSV

df = pd.read_csv('/kaggle/input/empdata/employee_data.csv')

df.head()
# Dataframe dimensions

df.shape
# Column datatypes

df.dtypes
df.head()
df['salary_numeric'] = df.salary.replace({'low': 0, 'medium': 1, 'high': 2})
# Dataframe dimensions

df.shape
# Plot histogram grid

df.hist(figsize=(10,10), xrot=-45)



# Clear the text "residue"

plt.show()
# Summarize numerical features

df.describe()
# Summarize categorical features

df.describe(include=['object'])
# Plot bar plot for each categorical feature

for feature in df.dtypes[df.dtypes == 'object'].index:

    sns.countplot(y=feature, data=df)

    plt.show()
# Segment satisfaction by status and plot distributions

sns.violinplot(y='status', x='avg_monthly_hrs', data=df)

plt.show()
# Segment last_evaluation by status and plot distributions

sns.violinplot(y='status', x='last_evaluation', data=df)

plt.show()
# Segment by status and display the means within each class

df.groupby('status').mean()
# Plot bar plot for each categorical feature

for feature in df.dtypes[df.dtypes == 'object'].index:

    sns.countplot(y=feature, data=df)

    plt.show()
df.groupby('department').mean().sort_values(by='avg_monthly_hrs')
# Scatterplot of last_evaluation vs. avg_monthly_hrs

sns.lmplot(x='last_evaluation',

           y='avg_monthly_hrs',

           hue='status',

           data=df,

           fit_reg=False,

           scatter_kws={'alpha':0.1})

plt.show()
# Scatterplot of last_evaluation vs. avg_monthly_hrs for leavers

sns.lmplot(x='last_evaluation', y='avg_monthly_hrs', data=df[df.status=='Left'], fit_reg=False)

plt.show()
# Data Cleaning

# Drop duplicates

df.drop_duplicates(inplace=True)

print( df.shape )
# Fix Structural Errors

sns.countplot(y='department', data=df)

plt.show()
df.department.replace('information_technology', 'IT', inplace=True)
df = df[df.department != 'temp']
# Print unique values of 'filed_complaint'

print( df.filed_complaint.unique() )



# Print unique values of 'recently_promoted'

print( df.recently_promoted.unique() )
# Missing filed_complaint values should be 0

df.filed_complaint.fillna(0, inplace=True)



# Missing recently_promoted values should be 0

df.recently_promoted.fillna(0, inplace=True)
# Print unique values of 'filed_complaint'

print( df.filed_complaint.unique() )



# Print unique values of 'recently_promoted'

print( df.recently_promoted.unique() )
# Display number of missing values by feature

df.isnull().sum()
# Unique classes of 'department'

print( df.department.unique() )
# Fill missing values in department with 'Missing'

df['department'].fillna('Missing', inplace=True)
df.head()
# Indicator variable for missing last_evaluation

df['last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)
# Fill missing values in last_evaluation with 0

df.last_evaluation.fillna(0, inplace=True)
df.head()
# Display number of missing values by feature

df.isnull().sum()
# Feature Engineering

# Scatterplot of satisfaction vs. last_evaluation, only those who have left

sns.lmplot(x='satisfaction',

           y='last_evaluation',

           data=df[df.status == 'Left'],

           fit_reg=False)

plt.show()
# Create indicator features

df['underperformer'] = ((df.last_evaluation < 0.6) & 

                        (df.last_evaluation_missing == 0)).astype(int)



df['unhappy'] = (df.satisfaction < 0.2).astype(int)



df['overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)
# The proportion of observations belonging to each group

df[['underperformer', 'unhappy', 'overachiever']].mean()
 # Convert status to an indicator variable

df['status'] = pd.get_dummies( df.status ).Left
df.head()
# The proportion of observations who 'Left'

df.status.mean()
# Create new dataframe with dummy features

df = pd.get_dummies(df, columns=['department', 'salary'])



# Display first 10 rows

df.head(10)
# Save analytical base table

df.to_csv('analytical_base_table.csv', index=None)
# Pickle for saving model files

import pickle



# Import Logistic Regression

from sklearn.linear_model import LogisticRegression



# Import RandomForestClassifier and GradientBoostingClassifer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



# Function for splitting training and test set

from sklearn.model_selection import train_test_split



# Function for creating model pipelines

from sklearn.pipeline import make_pipeline



# StandardScaler

from sklearn.preprocessing import StandardScaler



# GridSearchCV

from sklearn.model_selection import GridSearchCV



# Classification metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score
for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load analytical base table from Module 2

abt = pd.read_csv('analytical_base_table.csv')
# Create separate object for target variable

y = abt.status



# Create separate object for input features

X = abt.drop('status', axis=1)
# Split X and y into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=1234,

                                                    stratify=abt.status)



# Print number of observations in X_train, X_test, y_train, and y_test

print( len(X_train), len(X_test), len(y_train), len(y_test) )
# Pipeline dictionary

pipelines = {

    'l1' : make_pipeline(StandardScaler(), 

                         LogisticRegression(penalty='l1' , random_state=123)),

    'l2' : make_pipeline(StandardScaler(), 

                         LogisticRegression(penalty='l2' , random_state=123)),

    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123)),

    'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=123))

}
# List tuneable hyperparameters of our Logistic pipeline

pipelines['l1'].get_params()
# Logistic Regression hyperparameters

l1_hyperparameters = {

    'logisticregression__C' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],

}



l2_hyperparameters = {

    'logisticregression__C' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],

}
# Random Forest hyperparameters

rf_hyperparameters = {

    'randomforestclassifier__n_estimators': [100, 200],

    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33],

    'randomforestclassifier__min_samples_leaf': [1, 3, 5, 10]

}
# Boosted Tree hyperparameters

gb_hyperparameters = {

    'gradientboostingclassifier__n_estimators': [100, 200],

    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],

    'gradientboostingclassifier__max_depth': [1, 3, 5]

}
# Create hyperparameters dictionary

hyperparameters = {

    'l1' : l1_hyperparameters,

    'l2' : l2_hyperparameters,

    'rf' : rf_hyperparameters,

    'gb' : gb_hyperparameters

}
# Create empty dictionary called fitted_models

fitted_models = {}



# Loop through model pipelines, tuning each one and saving it to fitted_models

for name, pipeline in pipelines.items():

    # Create cross-validation object from pipeline and hyperparameters

    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    

    # Fit model on X_train, y_train

    model.fit(X_train, y_train)

    

    # Store model in fitted_models[name] 

    fitted_models[name] = model

    

    # Print '{name} has been fitted'

    print(name, 'has been fitted.')
# Display best_score_ for each fitted model

for name, model in fitted_models.items():

    print( name, model.best_score_ )
# Predict classes using L1-regularized logistic regression 

pred = fitted_models['l1'].predict(X_test)



# Display first 10 predictions

print( pred[:10] )
# Display confusion matrix for y_test and pred

print( confusion_matrix(y_test, pred) )
# Predict PROBABILITIES using L1-regularized logistic regression

pred = fitted_models['l1'].predict_proba(X_test)



# Get just the prediction for the positive class (1)

pred = [p[1] for p in pred]



# Display first 10 predictions

print( np.round(pred[:10], 2) )
# Calculate ROC curve from y_test and pred

fpr, tpr, thresholds = roc_curve(y_test, pred)
# Initialize figure

fig = plt.figure(figsize=(9,9))

plt.title('Receiver Operating Characteristic')



# Plot ROC curve

plt.plot(fpr, tpr, label='l1')

plt.legend(loc='lower right')



# Diagonal 45 degree line

plt.plot([0,1],[0,1],'k--')



# Axes limits and labels

plt.xlim([-0.1,1.1])

plt.ylim([-0.1,1.1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# Calculate AUROC

print( roc_auc_score(y_test, pred) )
for name, model in fitted_models.items():

    pred = model.predict_proba(X_test)

    pred = [p[1] for p in pred]

    

    print( name, roc_auc_score(y_test, pred) )
# Save winning model as final_model.pkl

with open('final_model.pkl', 'wb') as f:

    pickle.dump(fitted_models['rf'].best_estimator_, f)
# Load final_model.pkl as model

with open('final_model.pkl', 'rb') as f:

    clf = pickle.load(f)
# Display model object

print( clf )
# Load analytical base table used in Module 4

abt = pd.read_csv('analytical_base_table.csv')
# Create separate object for target variable

y = abt.status



# Create separate object for input features

X = abt.drop('status', axis=1)



# Split X and y into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=1234,

                                                    stratify=abt.status)
# Predict X_test

pred = clf.predict_proba(X_test)



# Get just the prediction for the positive class (1)

pred = [p[1] for p in pred]



# Print AUROC

print( 'AUROC:', roc_auc_score(y_test, pred) )
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv('/kaggle/input/unseen-raw-data/unseen_raw_data.csv')



raw_data.head()
def clean_data(df):

    # Drop duplicates

    df = df.drop_duplicates()

    

    # Drop temporary workers

    df = df[df.department != 'temp']

    

    #Salary to numeric

    df['salary_numeric'] = df.salary.replace({'low': 0, 'medium': 1, 'high': 2})

    

    # Missing filed_complaint values should be 0

    df['filed_complaint'] = df.filed_complaint.fillna(0)



    # Missing recently_promoted values should be 0

    df['recently_promoted'] = df.recently_promoted.fillna(0)

    

    # 'information_technology' should be 'IT'

    df.department.replace('information_technology', 'IT', inplace=True)



    # Fill missing values in department with 'Missing'

    df['department'].fillna('Missing', inplace=True)



    # Indicator variable for missing last_evaluation

    df['last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)

    

    # Fill missing values in last_evaluation with 0

    df.last_evaluation.fillna(0, inplace=True)

    

    # Return cleaned dataframe

    return df
# Create cleaned_new_data 

cleaned_data = clean_data(raw_data)



# Display first 5 rows

cleaned_data.head()
def engineer_features(df):

    # Create indicator features

    df['underperformer'] = ((df.last_evaluation < 0.6) & 

                            (df.last_evaluation_missing == 0)).astype(int)



    df['unhappy'] = (df.satisfaction < 0.2).astype(int)



    df['overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)

        

    # Create new dataframe with dummy features

    df = pd.get_dummies(df, columns=['department', 'salary'])

    

    # Return augmented DataFrame

    return df
# Create augmented_new_data

augmented_data = engineer_features(cleaned_data)



# Display first 5 rows

augmented_data.head()
X_test.head()
augmented_data.shape
X_test.shape
# Predict probabilities

pred = clf.predict_proba(X_test)
X_test.head(10)
# Print first 10 predictions

print( pred[:10] )