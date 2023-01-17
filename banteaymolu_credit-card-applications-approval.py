# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# missing value matrix viz.

import missingno as mno



# Classifier libraries

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier





# Confusion Matrix, ROC curve, AUC

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve





# Import LabelEncoder, and scaler

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



# Import train_test_split

from sklearn.model_selection import train_test_split



# Model parameter: GridSearchCV

from sklearn.model_selection import GridSearchCV





# Cross-validation and learning curves

from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, learning_curve



sns.set()
filename = '/kaggle/input/exercicio-2704/credit.data'



# The original dataset has no header. 

data_df = pd.read_csv(filename, header=None)



# Data summary

print(data_df.info())
data_df.head()
# Following the blog, we will now change the column names of the dataset.

col_name = ['gender','age', 'dept', 'married','bank_customer', 'education_level','ethnicity','years_employed',

           'prior_default','employed','credit_score', 'drivers_license','citizen', 'zip_code', 'income', 

            'approval_status']



data_df.columns = col_name



print(data_df.info())
# Lets first check the number of missing labels.

print("Total number of missing values: ", data_df.isnull().values.sum())
# Group sample columns.

print(data_df.groupby('gender')['gender'].count())

print(data_df.groupby('married')['married'].count())

print(data_df.groupby('bank_customer')['bank_customer'].count())
# Replace "?" with NaN

data_df = data_df.replace(to_replace='?', value=np.nan)



# check the number of missing values again.

print("Total number of missing values: ", data_df.isnull().values.sum())
data_df.sample(10).head(10)
# Change age to float 

data_df['age'] = data_df.age.astype(float)



# For boolean data types, we can simply use a label encoder to change the labels to either 1 or 0

# (yes the datatype will be an int)

# Instantiate LabelEncoder

le = LabelEncoder()

data_df['employed'] = le.fit_transform(data_df['employed'])

data_df['prior_default'] = le.fit_transform(data_df['prior_default'])

data_df['drivers_license'] = le.fit_transform(data_df['drivers_license'])

data_df['approval_status'] = le.fit_transform(data_df['approval_status'])



print(data_df.info())
print('Not Approved', round(data_df['approval_status'].value_counts()[0]/len(data_df) * 100,2), '% of the dataset')

print('Approved', round(data_df['approval_status'].value_counts()[1]/len(data_df) * 100,2), '% of the dataset')



# Color plot of each class in approbal status.

colors = ["#0101DF", "#DF0101"]

sns.countplot('approval_status', data=data_df, palette=colors)
# Let's display % of missing values for each feature

def missing_values(dataset):

    total = dataset.isnull().sum().sort_values(ascending = False)

    percent = (dataset.isnull().sum()/dataset.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_values(data_df)
mno.matrix(data_df)
mno.heatmap(data_df, figsize=(18,6))
col_name = ['age','dept', 'years_employed', 'income']

lin_data = pd.DataFrame(columns = ["lin_age"])

lin_data['lin_age'] = data_df['age']



# Initialize model

linreg = LinearRegression()



# First copy original dataset

data_with_null = data_df[col_name].copy()



# Then drop rows with null values

data_without_null = data_with_null.dropna()



# train our model using values fron non missing age variable

linreg.fit(data_without_null.iloc[:,1:3], data_without_null.iloc[:,0])



# Using rows where age is null, predict and replace age null values

# Note: we maintain the index of the null values so we are only replacing those values.

lin_data.loc[data_df['age'].isnull(),'lin_age'] = linreg.predict(

    data_with_null[data_with_null['age'].isnull()].iloc[:,1:3])





# Finally display a distribution and box plots to see the effect of replacing NaN.

fig, ax = plt.subplots(1, 2, figsize=(16,5))

sns.distplot(lin_data['lin_age'], color='#0101DF',kde=False, label='Lin_age', ax=ax[0])

sns.distplot(data_df[data_df['age'].notnull()].age.values, color='#DF0101', kde=False, label='age', ax=ax[0])



# Box plots

sns.boxplot(data = pd.concat([data_df['age'], lin_data['lin_age']], axis = 1), ax=ax[1])
# Print summary stats for each.

pd.concat([data_df['age'], lin_data['lin_age']], axis = 1).describe().T
# Finally replace 'age' column with the updated values

data_df['age'] = lin_data['lin_age']

data_df.age.isnull().sum()
# Iterate over each column and replace missing values with mode

for col in data_df.columns.values:

    if data_df[col].dtypes == 'object':

        # Impute with the most frequent value

        data_df = data_df.fillna(data_df[col].value_counts().index[0])



        

# Finally diplay the nullity matrix one more time.

mno.matrix(data_df)
# Unique zip codes

print(data_df.zip_code.nunique())



final_dataset = data_df.drop(columns=['zip_code'], axis=1).copy()
# Separete target and independent variables

X_values = final_dataset.loc[:,'gender':'income'].copy()

Y_value = final_dataset.loc[:,'approval_status'].copy()



print(X_values.shape)



# Convert education level to numeric using label encoding

X_values['education_level'] = le.fit_transform(X_values['education_level'])



# For the rest of the catorical features, convert to dummy variables and drop the first column.

cat_cols = ['gender','married', 'bank_customer', 'ethnicity', 'citizen']



for col in cat_cols:

    X_values = pd.get_dummies(X_values, columns=[col], prefix=[col], drop_first=True)



# Print shape

print(X_values.shape)
# Initialize the min max scaler

scaler = MinMaxScaler(feature_range=(0, 1))



# Scale each column. Here, dummy variables won't be affected.

X = scaler.fit_transform(X_values)
# Confusion Matrix

def plot_confusion_matrix(y_test, y_pred, clf_name):

    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    fig, (ax1) = plt.subplots(ncols=1, figsize=(10,4))

    sns.heatmap(cm, 

                xticklabels=['Not Approved', 'Approved'],

                yticklabels=['Not Approved', 'Approved'],

                annot=True,

                ax=ax1,

                linewidths=.2,

                linecolor="Darkblue", 

                cmap="Blues")

    plt.title( str(clf_name + ' Confusion Matrix'), fontsize=14)

    plt.show()
# Split data into training and test sets.

X_train, X_test, y_train, y_test = train_test_split(X,

                                Y_value,

                                test_size=0.3,

                                random_state=42)
# Instantiate a LogisticRegression classifier with default parameter values

logreg = LogisticRegression(solver='liblinear')



# Fit logreg to the train set

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)

reports = classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])



print(reports)
# First let's implement simple classifiers

classifiers = {

    "Logreg": LogisticRegression(),

    "KNC": KNeighborsClassifier(),

    "DTC": DecisionTreeClassifier(),

    "SVC": SVC(probability=True)

}


for clf_name, clf in classifiers.items():

    # Fit clf to the training set

    clf.fit(X_train, y_train)    

   

    # Predict y_pred

    y_pred = clf.predict(X_test)

    y_pred_prob = clf.predict_proba(X_test)[:,1]

    

    reports = classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])

    auc =  roc_auc_score(y_test, y_pred_prob)

    print('{:s} reports:'.format(clf_name))

    print(reports)

    print("\n")

    

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title("{:s} ROC has : {} AUC score".format(clf_name, auc))

    plt.show()
# Logist Regression Grid Search parameters

penalty=['l1', 'l2']

tol = [0.01, 0.001, 0.0001]

c=[0.001,0.01, 0.1, 1, 10, 100, 400] 

max_iter = [100, 150, 200] 

solver=['liblinear']



logreg_param_grid = dict({'tol': tol, 'max_iter': max_iter, 'penalty': penalty,'C': c, 'solver': solver})



log_reg_grid_model = GridSearchCV(LogisticRegression(), param_grid=logreg_param_grid, cv=5, iid=True)

log_reg_grid_model.fit(X_train, y_train)

logreg_grid_model = log_reg_grid_model.best_estimator_

print("Logestic Best Scrore: ", log_reg_grid_model.best_score_, " gride: ", log_reg_grid_model.best_params_)





# KNeighborsClassifier Grid Search parameters

neighbors = list(range(2,6,1))

alg = ['auto', 'ball_tree', 'kd_tree', 'brute']

knc_param_grid = dict({'n_neighbors': neighbors, 'algorithm': alg})

knc_grid_model = GridSearchCV(KNeighborsClassifier(), knc_param_grid, cv=5, iid=True)

knc_grid_model.fit(X_train, y_train)

# KNears best estimator

knears_grid_model = knc_grid_model.best_estimator_

print("KNeighbors Best Scrore: ", knc_grid_model.best_score_, " gride: ", knc_grid_model.best_params_)





# Support Vector Classification Grid Search parameters

c=[0.001,0.01, 0.1, 1, 10] 

kernel=['rbf', 'poly', 'sigmoid', 'linear']

gamma = ['auto', 'scale']

svc_param_grid = dict({'C': c, 'kernel': kernel, 'gamma': gamma})

scv_grid_model = GridSearchCV(SVC(probability=True), svc_param_grid, cv=5)

scv_grid_model.fit(X_train, y_train)



# SVC best estimator

svc_g_model = scv_grid_model.best_estimator_

print("SVC Best Scrore: ", scv_grid_model.best_score_, " gride: ", scv_grid_model.best_params_)







# A decision tree classifier.

ctrn = ["gini", "entropy"]

max_depth = list(range(2,5,1))

min_samples_leaf = list(range(2,8,1))

splitter = ['best']

dtc_param_grid = dict({'criterion': ctrn, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 

                       'splitter': splitter})

dtc_grid_model = GridSearchCV(DecisionTreeClassifier(), dtc_param_grid, cv=5, iid=True)

dtc_grid_model.fit(X_train, y_train)



dtree_grid_model = dtc_grid_model.best_estimator_



print("Decision Tree Best Scrore: ", dtc_grid_model.best_score_, " gride: ", dtc_grid_model.best_params_)
log_reg_score = cross_val_predict(logreg_grid_model, X_train, y_train, cv=5, method="decision_function")

knears_score = cross_val_predict(knears_grid_model, X_train, y_train, cv=5)

svc_score = cross_val_predict(svc_g_model, X_train, y_train, cv=5, method="decision_function")

tree_score = cross_val_predict(dtree_grid_model, X_train, y_train, cv=5)



# Calculate AUC: Logistic regression performs better.

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_score))

print('Support Vector Classifier: ', roc_auc_score(y_train, svc_score))

print('KNears Neighbors: ', roc_auc_score(y_train, knears_score))

print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_score))
def plot_learning_curve(estimator, clf_name, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(10,5))

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    plt.title(str(clf_name + " Learning Curve"), fontsize=14)

    plt.xlabel('Training size (m)')

    plt.ylabel('Score')

    plt.grid(True)

    plt.legend(loc="best")

    

    plt.show()

    
n_classifiers = {'Logistic Regression': logreg_grid_model, 'KNeighbors Classifier': knears_grid_model,

                 'Support Vector Classifier': svc_g_model, 'Decision Tree Classifier': dtree_grid_model}

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

for clf_name, clf in n_classifiers.items(): 

    plot_learning_curve(clf, clf_name, X_train, y_train, cv=cv, n_jobs=4)

    
# Finally, we 

for clf_name, clf in n_classifiers.items():    

 

    # Predict y_pred

    y_pred = clf.predict(X_test)

    

    cfm = confusion_matrix(y_test, y_pred)

    reports = classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])

    print('{:s} reports:'.format(clf_name))

    plot_confusion_matrix(y_test, y_pred, clf_name)

    print("\n")

    print(reports)

    print("\n")

    

    

    y_pred_prob = clf.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title("{:s} AUC: {}".format(clf_name, roc_auc_score(y_test, y_pred_prob)))

    plt.show()