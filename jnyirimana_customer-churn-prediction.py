# loading necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

from sklearn import metrics

import missingno as msno

import math

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import scipy.sparse

import gc

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.metrics import roc_auc_score

import warnings 

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, matthews_corrcoef

from sklearn.metrics import f1_score

import scipy as sp

import os

import scikitplot as skplt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# to increase the display capacity

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
df_train = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
df_train.head()
df_train.info()
df_train.isnull().sum()
print(df_train.Exited.value_counts())

#Visualising non-churners and churners cases

plt.bar("Churn", df_train["Exited"].value_counts()[1], color="red")

plt.bar("No Churn", df_train["Exited"].value_counts()[0], color="green")

plt.ylabel("Count", fontsize=14)

plt.title("Churn VS No Churn")
df_train.groupby('Exited')['CreditScore'].mean() 
df_train.groupby('Exited')['Age'].mean() 
df_train.groupby('Exited')['Balance'].mean() 
df_train.groupby('Exited')['EstimatedSalary'].mean() 
df_train.describe()
df_train.Tenure.value_counts()
df_train.NumOfProducts.value_counts()
df_train.HasCrCard.value_counts()
df_train.IsActiveMember.value_counts()
df_train.describe(exclude = 'number')
df_train.Geography.value_counts()
df_train.Gender.value_counts()
df_train.groupby('Geography')['Exited'].value_counts()
df_train.groupby('Gender')['Exited'].value_counts()
df_train.groupby('NumOfProducts')['Exited'].value_counts()
df_train.groupby('Gender')['NumOfProducts'].value_counts()
df_train.groupby('Tenure')['Exited'].value_counts()
def describe(datatrain, feature):

    d = pd.DataFrame(columns=[feature,'Train','Train - Churn','Train - No Churn'])

    d[feature] = ['count','mean','std','min','25%','50%','75%','max','unique','NaN','NaNshare']

    for i in range(0,8):

        d['Train'].iloc[i] = datatrain[feature].describe().iloc[i]

        d['Train - Churn'].iloc[i]=datatrain[datatrain['Exited']==1][feature].describe().iloc[i]

        d['Train - No Churn'].iloc[i]=datatrain[datatrain['Exited']==0][feature].describe().iloc[i]

    d['Train'].iloc[8] = len(datatrain[feature].unique())

    d['Train - Churn'].iloc[8]=len(datatrain[datatrain['Exited']==1][feature].unique())

    d['Train - No Churn'].iloc[8]=len(datatrain[datatrain['Exited']==0][feature].unique())

    d['Train'].iloc[9] = datatrain[feature].isnull().sum()

    d['Train - Churn'].iloc[9] = datatrain[datatrain['Exited']==1][feature].isnull().sum()

    d['Train - No Churn'].iloc[9] = datatrain[datatrain['Exited']==0][feature].isnull().sum()

    d['Train'].iloc[10] = datatrain[feature].isnull().sum()/len(datatrain)

    d['Train - Churn'].iloc[10] = datatrain[datatrain['Exited']==1][feature].isnull().sum()/len(datatrain[datatrain['Exited']==1])

    d['Train - No Churn'].iloc[10] = datatrain[datatrain['Exited']==0][feature].isnull().sum()/len(datatrain[datatrain['Exited']==0])

    return d
BalanceAmtDescribe = describe(df_train,'Balance')
BalanceAmtDescribe
df_train[df_train['Balance']>222000]
AgeDescribe = describe(df_train,'Age')
AgeDescribe
df_train[df_train['Age']>=80]
train_age = (df_train.groupby(['Exited'])['Age']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index()

                     .sort_values('Age'))

                     

plt.figure(figsize=(20,20))

sns.barplot(x="Age", y="percentage", hue="Exited", data=train_age)
# Visualize the distribution of 'Day_Mins'

sns.distplot(df_train.CreditScore)



# Display the plot

plt.show()
# Visualize the distribution of 'Day_Mins'

sns.distplot(df_train.Age)



# Display the plot

plt.show()
# Visualize the distribution of 'Day_Mins'

sns.distplot(df_train.Balance)



# Display the plot

plt.show()
# Visualize the distribution of 'Day_Mins'

sns.distplot(df_train.EstimatedSalary)



# Display the plot

plt.show()
sns.boxplot(x = 'Exited',

           y = 'Balance',

           data = df_train)

plt.show()
sns.boxplot(x = 'Exited',

           y = 'Balance',

           data = df_train,

           hue = 'Geography')

plt.show()
sns.boxplot(x = 'Exited',

           y = 'CreditScore',

           data = df_train)

plt.show()
# Printing the filtered dataframe to verify to verify

df_train[df_train['CreditScore'] < 400]
sns.boxplot(x = 'Exited',

           y = 'Age',

           data = df_train)

plt.show()
df_train.head()
correlations = df_train.corr()

fig = plt.figure(figsize = (9, 6))



sns.heatmap(correlations, vmax = .8, square = True)

plt.show()
k = 10 #number of variables for heatmap

cols = correlations.nlargest(k, 'Exited')['Exited'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(15, 15))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Drop the unnecessary features

df_train = df_train.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
# Replacing 'Male' with 0 and 'Female' with 1 

df_train['Gender'] = df_train['Gender'].replace({'Male': 0, 'Female': 1})



# Print the results to verify

print(df_train['Gender'].head())
# Perform one hot encoding on 'Geography'

df_train = pd.get_dummies(data=df_train, columns=['Geography', 'Tenure', 'NumOfProducts'])
# Create feature variables

X = df_train.drop('Exited', axis=1)



# Create target variable

y = df_train['Exited']



# Create training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y)
# Import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier



# Instantiate the classifier

clf = RandomForestClassifier(class_weight='balanced_subsample', random_state=123)



# Fit to the training data

clf.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = clf.predict(X_test)



# Import roc_auc_score

from sklearn.metrics import roc_auc_score



# Compute accuracy

#print(clf.score(X_test, y_test))



# Generate the probabilities

y_pred_prob = clf.predict_proba(X_test)[:, 1]



# Print the AUC

print(roc_auc_score(y_test, y_pred_prob))



import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred)
from xgboost import XGBClassifier

# Instantiate the classifier

clf = XGBClassifier(random_state=123)



# Fit to the training data

clf.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = clf.predict(X_test)



# Generate the probabilities

y_pred_prob = clf.predict_proba(X_test)[:, 1]



# Print the AUC

print(roc_auc_score(y_test, y_pred_prob))



skplt.metrics.plot_confusion_matrix(y_test, y_pred)
from lightgbm import LGBMClassifier

# Instantiate the classifier

clf = LGBMClassifier(random_state=123)



# Fit to the training data

clf.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = clf.predict(X_test)



# Generate the probabilities

y_pred_prob = clf.predict_proba(X_test)[:, 1]



# Print the AUC

print(roc_auc_score(y_test, y_pred_prob))



skplt.metrics.plot_confusion_matrix(y_test, y_pred)
from catboost import CatBoostClassifier

# Instantiate the classifier

clf = CatBoostClassifier(random_state=123, logging_level='Silent')



# Fit to the training data

clf.fit(X_train, y_train)



# Predict the labels of the test set

y_pred = clf.predict(X_test)



# Generate the probabilities

y_pred_prob = clf.predict_proba(X_test)[:, 1]



# Print the AUC

print(roc_auc_score(y_test, y_pred_prob))



skplt.metrics.plot_confusion_matrix(y_test, y_pred)
clf = CatBoostClassifier()

params = {'iterations': [500],

          'depth': [4, 5, 6, 7],

          'loss_function': ['Logloss', 'CrossEntropy'],

          'l2_leaf_reg': np.logspace(-20, -19, 3),

          'eval_metric': ['AUC'],

#           'use_best_model': ['True'],

          'logging_level':['Silent'],

          'random_seed': [42]

         }

#scorer = make_scorer(accuracy_score)

clf_grid = GridSearchCV(estimator=clf, param_grid=params, cv=5)
clf_grid.fit(X_train, y_train)

best_param = clf_grid.best_params_

best_param
from sklearn.metrics import classification_report
y_pred = clf_grid.predict(X_test)



# Generate the probabilities

y_pred_prob = clf_grid.predict_proba(X_test)[:, 1]



# Print the AUC

print(roc_auc_score(y_test, y_pred_prob))



# Generate the probabilities

y_pred_prob = clf_grid.predict_proba(X_test)[:, 1]



# Print the AUC

print('The AUC score using GridSearchCV is :\n', roc_auc_score(y_test, y_pred_prob))



# Obtain the results from the classification report and confusion matrix 

print(skplt.metrics.plot_confusion_matrix(y_test, y_pred))

target_names = ['No Churn', 'Churn']

print(classification_report(y_test, y_pred, target_names=target_names))
# This is the pipeline module we need for this from imblearn

from imblearn.pipeline import Pipeline 

from imblearn.over_sampling import SMOTE



# Define which resampling method and which ML model to use in the pipeline

resampling = SMOTE()

model = CatBoostClassifier(logging_level = 'Silent')



# Define the pipeline, tell it to combine SMOTE with the CatBoost model

pipeline = Pipeline([('SMOTE', resampling), ('CatBoost ', model)])
# Split your data X and y, into a training and a test set and fit the pipeline onto the training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 

pipeline.fit(X_train, y_train) 

predicted = pipeline.predict(X_test)



# Generate the probabilities

y_pred_prob = pipeline.predict_proba(X_test)[:, 1]



# Print the AUC

print('The AUC score using SMOTE is :\n', roc_auc_score(y_test, y_pred_prob))



# Obtain the results from the classification report and confusion matrix 

conf_mat = skplt.metrics.plot_confusion_matrix(y_test, predicted)

print('Confusion matrix:\n', conf_mat)

print(classification_report(y_test, predicted, target_names=target_names))
# Import the package

from sklearn.ensemble import VotingClassifier



# Define the three classifiers to use in the ensemble

clf1 = LogisticRegression(class_weight={0:1, 1:15}, random_state=5)

clf2 = RandomForestClassifier(class_weight={0:1, 1:12}, criterion='gini', max_depth=8, max_features='log2',

            min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)

clf3 = DecisionTreeClassifier(random_state=5, class_weight="balanced")



# Combine the classifiers in the ensemble model

ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')
#fitting the ensemble model onto the training set

ensemble_model.fit(X_train, y_train)



#making predictions

y_pred = ensemble_model.predict(X_test)



# Obtain the results from the classification report and confusion matrix 

conf_mat = skplt.metrics.plot_confusion_matrix(y_test, y_pred)

print('Confusion matrix:\n', conf_mat)

print(classification_report(y_test, y_pred, target_names=target_names))