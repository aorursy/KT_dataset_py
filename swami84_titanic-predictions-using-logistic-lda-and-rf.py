# Import important libraries

import pandas as pd

import numpy as np

import os

import re

import math

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from six.moves import urllib

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

kfold = model_selection.KFold(n_splits=5,shuffle = True)
# Get the Data



df_train = pd.read_csv("../input/train.csv")

Survived = df_train['Survived']

df_test = pd.read_csv("../input/test.csv")

df_whole = pd.concat([df_train, df_test],ignore_index=True)

df_whole= df_whole.drop(columns=['Survived'])

df_whole.info()
# Prepare the Data



prep=df_whole.copy() # Copying the original data set 

mean_Fare = (prep.groupby('Pclass')['Fare'].mean()).values # Grouping Fare by Passenger Class which will be used to fill missing/0 values

# Removed prep['Age'] = prep['Age'].fillna(prep['Age'].mean()) # Filling missing Age values with mean 



prep.loc[(prep['Fare']==0) & (prep['Pclass']==1),'Fare']  = mean_Fare[0] 

prep.loc[(prep['Fare']==0) & (prep['Pclass']==2),'Fare']  = mean_Fare[1]

prep.loc[(prep['Fare']==0) & (prep['Pclass']==3),'Fare']  = mean_Fare[2]

prep['Fare'] = prep['Fare'].fillna(prep['Fare'].mean())



# Create Family Size Column

prep['FamilySize'] = prep['SibSp'] + prep['Parch'] + 1



# Map gender

prep['Sex'] = prep['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Replace missing Embarked with most common station (S)

prep['Embarked'] = prep['Embarked'].fillna('S')

# Map Embarked

prep['Embarked'] = prep['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

# Cabin has a lot of missing values. Making it binary

prep['Has Cabin'] = prep[['Cabin']].applymap(lambda x: 0 if pd.isnull(x) else 1)



# Title Search (inspired by https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset)

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

prep['Title'] = prep['Name'].apply(get_title)

# Grouping Titles together

prep['Title'] = prep['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royal')

prep['Title'] = prep['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev', ], 'Other')

prep['Title'] = prep['Title'].replace('Mlle', 'Miss')

prep['Title'] = prep['Title'].replace('Ms', 'Miss')

prep['Title'] = prep['Title'].replace('Mme', 'Mrs')

prep['Title'] = prep['Title'].map( {'Master': 1, 'Miss': 2, 'Mr': 3, 'Mrs':4, 'Other': 5, 'Royal':6} ).astype(int)



# Remove columns not be used in the model

prep = prep.drop(columns=['Name','Ticket','Cabin', 'Parch', 'SibSp', 'PassengerId'])

prep.info()
## Creating the traning dataset

train = prep.iloc[0:891]

# Add labels

train['Survived'] = Survived

train.info()
sns.catplot(x='Sex', y='Age',hue='Survived',col = 'Embarked',data=train,kind= 'swarm')

sns.catplot(x='Sex', y='Age',hue='Survived',col = 'Pclass',data=train,kind= 'swarm')
mean_Age = (train.groupby(['Pclass', 'Survived'])['Age'].mean()).values

train['Age'] = train['Age'].fillna(0)

train.loc[(train['Age']==0) & (train['Pclass']==1) & (train['Survived']==0),'Age']  = mean_Age[0]

train.loc[(train['Age']==0) & (train['Pclass']==1) & (train['Survived']==1),'Age']  = mean_Age[1] 

train.loc[(train['Age']==0) & (train['Pclass']==2) & (train['Survived']==0),'Age']  = mean_Age[2]

train.loc[(train['Age']==0) & (train['Pclass']==2) & (train['Survived']==1),'Age']  = mean_Age[3]

train.loc[(train['Age']==0) & (train['Pclass']==3) & (train['Survived']==0),'Age']  = mean_Age[4]

train.loc[(train['Age']==0) & (train['Pclass']==3) & (train['Survived']==1),'Age']  = mean_Age[5]
sns.catplot(x='Sex', y='Age',hue='Survived',col = 'Embarked',data=train,kind= 'swarm')

sns.catplot(x='Sex', y='Age',hue='Survived',col = 'Pclass',data=train,kind= 'swarm')
## Plot Pearson's Correlation Matrix

colormap = plt.cm.viridis

plt.figure(figsize=(14,14))

plt.title('Pearson Correlation of Features', y=1.05, size=18)

sns.heatmap(train.astype(float).corr(),linewidths=0.08,vmax=1, square=True, cmap=colormap, linecolor='white', annot=True)
## Survival Rate by Title

train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])

# Mapping {'Master': 1, 'Miss': 2, 'Mr': 3, 'Mrs':4, 'Other': 5, 'Royal':6}
## Survival Rate by Cabin

train[['Has Cabin', 'Survived']].groupby(['Has Cabin'], as_index=False).agg(['mean', 'count', 'sum'])

# Mapping {'No Cabin': 0, 'Yes Cabin': 1}
## Survival Rate by Sex

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])

# Mapping {'female': 0, 'male': 1}
## Survival Rate by Passenger Class



print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).agg(['mean', 'count', 'sum']))

## Roll back some features to text category for feature mapping done later

rev_prep = prep.copy()

y = (train['Survived']).values

rev_prep['Sex'] = rev_prep['Sex'].map( {0:'female', 1:'male'} ).astype(str)

rev_prep['Embarked'] = rev_prep['Embarked'].map( {0: 'C', 1: 'Q',2: 'S'} ).astype(str)

rev_prep['Has Cabin'] = rev_prep['Has Cabin'].map( {0: 'No Cabin', 1: 'Yes Cabin'} ).astype(str)

rev_prep['Title'] = rev_prep['Title'].map( {1: 'Master', 2: 'Miss',3: 'Mr', 4: 'Mrs', 5: 'Other', 6: 'Royal'} ).astype(str)

rev_prep['Pclass'] = rev_prep['Pclass'].apply(str)

rev_prep['Pclass'] = 'P' + rev_prep['Pclass']

rev_prep.info()
### Creating Training and Test Data Sets

train_pre = rev_prep.iloc[0:891]

test_pre = rev_prep.iloc[891:]

test_pre['Age'] = test_pre['Age'].fillna(test_pre['Age'].mean()) # Filling missing Age values with mean
### Pipelines for converting numerical values (useless)



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]

    def get_feature_names(self):

        return X.columns.tolist()

from sklearn.impute import SimpleImputer 

from sklearn.pipeline import Pipeline



num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["Age", "Fare", "FamilySize"])),

        ("imputer", SimpleImputer(strategy="mean")),

    ])

num_pipeline.fit_transform(train_pre)
### Pipelines for converting categorical values



OHE = OneHotEncoder(sparse = False)

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)

cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(["Embarked", "Pclass", "Sex", 'Has Cabin', 'Title'])),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OHE),

    ])

cat_pipeline.fit_transform(train_pre)
from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])

X =  preprocess_pipeline.fit_transform(train)
### Splitting the Data set to train and test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the data and evaluate accuracy scores for commonly used classifiers

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='scale')))

models.append(('RF', RandomForestClassifier()))

cv_res = pd.DataFrame()

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=3)

    model.fit(X_train,y_train)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    msg = "%s: %f %f (%f)" % (name, cv_results.mean(),acc_score, cv_results.std())

    print(msg)


# Randomized Search for Logistic Regression

from sklearn.model_selection import RandomizedSearchCV



LR = LogisticRegression()

param_grid_LR = {"solver": ["newton-cg", 'liblinear','sag'],

                 'max_iter' : [1,5,10,100,500,1000]

             }





rnd_search_LR = RandomizedSearchCV(LR, param_grid_LR,n_iter=500, cv=kfold, scoring = 'accuracy', return_train_score = True, random_state=42)

rnd_search_LR.fit(X_train, y_train)
## Best parameters for LR

rnd_search_LR.best_params_
cvres = rnd_search_LR.cv_results_

for accuracy, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(accuracy), params)
## List LR features with importances

cat_pipeline.fit_transform(train_pre)

LR  =  LogisticRegression(**rnd_search_LR.best_params_)

LR.fit(X_train, y_train)

weights_LR = np.round(np.transpose(LR.coef_),3)

attribs0 = list(OHE.categories_[0])

attribs1 = list(OHE.categories_[1])

attribs2 = list(OHE.categories_[2])

attribs3 = list(OHE.categories_[3])

attribs4 = list(OHE.categories_[4])

attributes = ["Age", "Fare", "FamilySize"] + attribs0 + attribs1 + attribs2 +  attribs3 + attribs4

sorted(zip(weights_LR, attributes), reverse=True)
## List LDA features with importances



LDA = LinearDiscriminantAnalysis(solver = 'svd')

LDA.fit(X_train,y_train)

weights_LDA = np.transpose(LDA.coef_)

sorted(zip(weights_LDA, attributes), reverse=True)
# Randomized Search for Random Forest Classifier



from sklearn.model_selection import RandomizedSearchCV



RF = RandomForestClassifier()



param_grid_RF = {'bootstrap': [True, False],

 'max_depth': [5,10,50,150],

 'max_features': [2,3,4,5,6],

 'min_samples_leaf': [2, 3, 4, 5],

 'min_samples_split': [7,8,9,10,11],

 'n_estimators': [100,150,200,250]}



rnd_search_RF = RandomizedSearchCV(RF, param_grid_RF,n_iter=500, cv=kfold, scoring = 'accuracy',random_state=42, return_train_score = True)

rnd_search_RF.fit(X_train, y_train)
rnd_search_RF.best_params_
## Best parameters for RF



rnd_search_RF.best_params_
feature_importance_list = rnd_search_RF.best_estimator_.feature_importances_

df_attr_imp = pd.DataFrame({'attribute_name': attributes, 'importance': feature_importance_list})

df_attr_imp.sort_values('importance', ascending=False)
# Final CV scores, accuracy and confusion matrix



from sklearn.metrics import confusion_matrix

final_models = []

final_models.append(('LR', LogisticRegression(**rnd_search_LR.best_params_, random_state=42)))

final_models.append(('LDA', LinearDiscriminantAnalysis(solver = 'svd')))

final_models.append(('RF', RandomForestClassifier(**rnd_search_RF.best_params_)))

cv_res = pd.DataFrame()

results = []

names = []

for name, model in final_models:

    model.fit(X_train,y_train)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    y_pred = model.predict(X_test)

    cv_score = np.round((cv_results.mean()),3)

    acc_score = np.round(accuracy_score(y_test, y_pred),3)

    

    print(name, ': CV accuracy is', cv_score,'Test Accuracy is',acc_score  )



    print ('\n The confusion matrix of ',name,'is \n',confusion_matrix( y_test,y_pred), "\n" )
# Final LDA predictions for test data set

test_final = preprocess_pipeline.fit_transform(test_pre)

y_pred_LDA = LDA.predict(test_final)

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred_LDA

    })

filename = 'Titanic Predictions LDA.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
# Final LR predictions for test data set



LR  =  LogisticRegression(**rnd_search_LR.best_params_)

LR.fit (X_train, y_train)

y_pred_LR = LR.predict(test_final)

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred_LR

    })

filename = 'Titanic Predictions LR.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
# Final RF predictions for test data set

RF  =  RandomForestClassifier(**rnd_search_RF.best_params_)

RF.fit (X_train, y_train)

y_pred_RF = RF.predict(test_final)

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred_RF

    })

filename = 'Titanic Predictions RF.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)