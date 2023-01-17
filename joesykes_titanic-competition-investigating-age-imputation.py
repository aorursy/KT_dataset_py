import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
print("Missing data points within the Training dataset:")

print(train_df.isnull().sum())

print("-"*25)

print("Missing data points within the Test dataset:")

print(test_df.isnull().sum())
train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df[test_df['Fare'].isnull() == True]
test_df[test_df['Pclass'] == 3]['Fare'].median()
test_df['Fare'] = test_df['Fare'].fillna(test_df[test_df['Pclass'] == 3]['Fare'].median())
test_df[test_df['PassengerId'] == 1044]
print("Unique 'Cabin' values in the Training dataset")

print(train_df['Cabin'].unique())

print("-" * 50)

print("Unique 'Cabin' values in the Test dataset")

print(test_df['Cabin'].unique())
train_df['Cabin'] = train_df['Cabin'].fillna('Missing')

test_df['Cabin'] = test_df['Cabin'].fillna('Missing')
train_df['Cabin'] = train_df['Cabin'].str[0]

test_df['Cabin'] = test_df['Cabin'].str[0]



ct = pd.crosstab(index=train_df['Survived'],columns=train_df["Cabin"],normalize="columns")

ct.T.plot(kind="bar",figsize=(15,10),stacked=True)
print("Missing data points within the Training dataset:")

print(train_df.isnull().sum())

print("-"*25)

print("Missing data points within the Test dataset:")

print(test_df.isnull().sum())

train_df['Name']
train_df['Title'] = train_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].replace(' ',''))

test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].replace(' ',''))
train_df['Title'].value_counts()
test_df['Title'].value_counts()
combine = [train_df,test_df]



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'theCountess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Misc')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
test_ids = list(test_df['PassengerId'])

y = train_df['Survived']

train_df = train_df.drop(['PassengerId','Name','Ticket','Survived'],axis=1)

test_df = test_df.drop(['PassengerId','Name','Ticket'],axis=1)
train_df.head()
test_df.head()
train_df.head()
train_df1 = train_df.copy(deep=True)

train_df2 = train_df.copy(deep=True)

train_df3 = train_df.copy(deep=True)

train_df4 = train_df.copy(deep=True)
train_df.groupby('Pclass').mean()['Age']
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 38

        elif Pclass == 2:

            return 30

        else:

            return 25

        

    else:

        return Age
train_df1['Age'] = train_df1[['Age','Pclass']].apply(impute_age,axis=1)
train_df1.isnull().sum()
train_df1.head()
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()

from sklearn.compose import ColumnTransformer

columnTransformer1 = ColumnTransformer([('encoder', onehot_encoder, [1,6,7,8])], remainder='passthrough')

train_df1 = columnTransformer1.fit_transform(train_df1)
train_df2 = train_df2.drop('Cabin',axis=1)
train_df2['Age'] = train_df2[['Age','Pclass']].apply(impute_age,axis=1)
columnTransformer2 = ColumnTransformer([('encoder', onehot_encoder, [1,6,7])],remainder='passthrough')

train_df2 = columnTransformer2.fit_transform(train_df2)
columnTransformer3 = ColumnTransformer([('encoder', onehot_encoder, [1,6,7,8])],remainder='passthrough')

train_df3 = columnTransformer3.fit_transform(train_df3)
from sklearn.impute import KNNImputer



imputer1 = KNNImputer()

train_df3 = imputer1.fit_transform(train_df3)
train_df4 = train_df4.drop("Cabin", axis=1)
columnTransformer4 = ColumnTransformer([('encoder', onehot_encoder, [1,6,7])],remainder='passthrough')

train_df4 = columnTransformer4.fit_transform(train_df4)
imputer2 = KNNImputer()

train_df4 = imputer2.fit_transform(train_df4)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df1,y,test_size=0.25,random_state=101)
from sklearn.ensemble import RandomForestClassifier

scen1_rf_model = RandomForestClassifier()

scen1_rf_model.fit(X_train,y_train)
scen1_rf_model_preds = scen1_rf_model.predict(X_test)
from xgboost import XGBClassifier

scen1_xgb_model = XGBClassifier()

scen1_xgb_model.fit(X_train, y_train)

scen1_xgb_model_preds = scen1_xgb_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print("RANDOM FOREST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen1_rf_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen1_rf_model_preds))

print("-"*50)

print("XGBOOST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen1_xgb_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen1_xgb_model_preds))
X_train, X_test, y_train, y_test = train_test_split(train_df2,y,test_size=0.25,random_state=101)
scen2_rf_model = RandomForestClassifier()

scen2_rf_model.fit(X_train, y_train)

scen2_rf_model_preds = scen2_rf_model.predict(X_test)
scen2_xgb_model = XGBClassifier()

scen2_xgb_model.fit(X_train, y_train)

scen2_xgb_model_preds = scen2_xgb_model.predict(X_test)
print("RANDOM FOREST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen2_rf_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen2_rf_model_preds))

print("-"*50)

print("XGBOOST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen2_xgb_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen2_xgb_model_preds))
X_train, X_test, y_train, y_test = train_test_split(train_df3, y, test_size=0.25, random_state=101)
scen3_rf_model = RandomForestClassifier()

scen3_rf_model.fit(X_train, y_train)

scen3_rf_model_preds = scen3_rf_model.predict(X_test)
scen3_xgb_model = XGBClassifier()

scen3_xgb_model.fit(X_train, y_train)

scen3_xgb_model_preds = scen3_xgb_model.predict(X_test)
print("RANDOM FOREST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen3_rf_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen3_rf_model_preds))

print("-"*50)

print("XGBOOST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen3_xgb_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen3_xgb_model_preds))
X_train, X_test, y_train, y_test = train_test_split(train_df4,y,test_size=0.25,random_state=101)
scen4_rf_model = RandomForestClassifier()

scen4_rf_model.fit(X_train, y_train)

scen4_rf_model_preds = scen4_rf_model.predict(X_test)
scen4_xgb_model = XGBClassifier()

scen4_xgb_model.fit(X_train,y_train)

scen4_xgb_model_preds = scen4_xgb_model.predict(X_test)
print("RANDOM FOREST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen4_rf_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen4_rf_model_preds))

print("-"*50)

print("XGBOOST CLASSIFIER")

print()

print("Confusion Matrix")

print()

print(confusion_matrix(y_test,scen4_xgb_model_preds))

print()

print("Classification Report")

print()

print(classification_report(y_test,scen4_xgb_model_preds))
test_df = test_df.drop("Cabin",axis=1)
test_df = columnTransformer4.transform(test_df)
test_df = imputer2.transform(test_df)
from sklearn.model_selection import GridSearchCV



final_xgb = XGBClassifier(random_state=11)

gscv = GridSearchCV(estimator=final_xgb,param_grid={

   "n_estimators":[100,500,1000,5000],

   "criterion":["gini","entropy"]

},cv=5,n_jobs=-1,scoring="accuracy")



model = gscv.fit(train_df4,y)

final_xgb_model = model.best_estimator_
final_predictions = final_xgb_model.predict(test_df)



submission = pd.DataFrame({"PassengerId":test_ids,"Survived":final_predictions})
submission.to_csv("Submission_Attempt_2.csv",index=False)