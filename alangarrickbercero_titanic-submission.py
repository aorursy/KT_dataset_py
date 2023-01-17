import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
testDf = pd.read_csv('../input/titanic/test.csv', index_col=0)

trainDf = pd.read_csv('../input/titanic/train.csv', index_col=0)
trainDf.head()
sns.heatmap(trainDf.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Lots of missing data in Cabin and Age

#No cabin data=they did not stay in cabins!

#Impute Age
sns.heatmap(testDf.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#same for test data, lots of missing Age and Cabin, even fare

#Impute Age and Fare
# Learned this from Niklas Donges, no cabin is U0

trainDf['Cabin'].fillna('U0', inplace=True)

testDf['Cabin'].fillna('U0', inplace=True)
import re

deck = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 8}



trainDf['Deck'] = trainDf['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

trainDf['Deck'] = trainDf['Deck'].map(deck)

trainDf['Deck'] = trainDf['Deck'].fillna(0)

trainDf['Deck'] = trainDf['Deck'].astype(int)

trainDf.drop('Cabin',axis=1,inplace=True)
deck = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 8}



testDf['Deck'] = testDf['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

testDf['Deck'] = testDf['Deck'].map(deck)

testDf['Deck'] = testDf['Deck'].fillna(0)

testDf['Deck'] = testDf['Deck'].astype(int)

testDf.drop('Cabin',axis=1,inplace=True)
# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(trainDf.drop(['Name','Survived','Ticket'],axis=1), trainDf['Survived'], test_size=0.3)
# Impute on training Data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train[['Age']]), index=X_train.index)

imputed_X_valid = pd.DataFrame(imputer.transform(X_valid[['Age']]), index=X_valid.index)

imputed_X_train.columns = ['Age']

imputed_X_valid.columns = ['Age']

imputed_X_train = pd.concat([X_train.drop('Age', axis=1), imputed_X_train], axis = 1)

imputed_X_valid = pd.concat([X_valid.drop('Age', axis=1), imputed_X_valid], axis = 1)
imputed_X_train.head()
# Impute on testing Data

imputed_X_test = pd.DataFrame(imputer.fit_transform(testDf[['Age', 'Fare']]), index=testDf.index)

imputed_X_test.columns = ['Age','Fare']

imputed_X_test = pd.concat([testDf.drop(['Name','Age','Fare','Ticket'], axis=1), imputed_X_test], axis = 1)
imputed_X_test.head()
#Create Labeled Data

object_cols = [col for col in imputed_X_train.columns if imputed_X_train[col].dtype == 'object']

good_label_cols = [col for col in object_cols if set(imputed_X_train[col])==set(imputed_X_valid[col])]

bad_label_cols = list(set(object_cols)-set(good_label_cols))

label_X_train = imputed_X_train.drop(bad_label_cols, axis=1)

label_X_valid = imputed_X_valid.drop(bad_label_cols, axis=1)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in good_label_cols:

    label_X_train[col] = le.fit_transform(imputed_X_train[col].astype(str))

    label_X_valid[col] = le.transform(imputed_X_valid[col].astype(str))

label_X_train
label_X_test = imputed_X_test.drop(bad_label_cols, axis=1)

for col in good_label_cols:

    label_X_test[col] = le.fit_transform(imputed_X_test[col].astype(str))

label_X_test
# Create Model

from xgboost import XGBClassifier

model = XGBClassifier(early_stopping_rounds=5,

                     eval_set=[(label_X_valid, y_valid)],

                     verbose=False)

model.fit(label_X_train,y_train)
pred = model.predict(label_X_valid)
# metrics

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_valid, pred))

print(confusion_matrix(y_valid, pred))
full_X = pd.concat([label_X_train,label_X_valid],axis=0)

full_X.sort_index(inplace=True)



full_y = pd.concat([y_train,y_valid],axis=0)

full_y.sort_index(inplace=True)
label_X_test = label_X_test[full_X.columns]
# train model on full training set

model = XGBClassifier(early_stopping_rounds=5,

                     verbose=False)

model.fit(full_X,full_y)

fullPred = model.predict(label_X_test)
presubmissionDf = pd.DataFrame(fullPred, columns=['Survived'])

presubmissionDf.index = np.arange(892, 892+len(presubmissionDf))

presubmissionDf = pd.concat([label_X_test,presubmissionDf], axis=1)

presubmissionDf['Name']=testDf['Name']

presubmissionDf
submissionDf=pd.DataFrame(presubmissionDf['Survived'], columns=['Survived'])

submissionDf = submissionDf.rename_axis('PassengerId')

submissionDf.to_csv('titanicSurvivorPredictions.csv')