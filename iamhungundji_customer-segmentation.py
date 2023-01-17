import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/janatahack-customer-segmentation/Train.csv')

test = pd.read_csv('../input/janatahack-customer-segmentation/Test.csv')

train.head()
combine = train.append(test)

combine.shape
combine.columns
combine.isnull().sum()
combine['Week'] = combine['ID'] % 7

combine['Month'] = combine['ID'] % 30

combine['Year'] = combine['ID'] % 365

combine['Quarter'] = combine['ID'] % 90



combine['NWeeks'] = combine['ID'] // 7

combine['NYear'] = combine['ID'] // 365

combine['NQuarter'] = combine['ID'] // 90

combine['NMonth'] = combine['ID'] // 30



combine.head()
combine['Gender'].value_counts()
combine['Ever_Married'].value_counts()
combine['Ever_Married'].fillna('Unknown', inplace=True)

combine['Ever_Married'].value_counts()
combine['Age'].describe()
bins= [17,30,40,50,60,90]

labels = ['Age_Tier1','Age_Tier2','Age_Tier3','Age_Tier4', 'Age_Tier5']

combine['Age'] = pd.cut(combine['Age'], bins=bins, labels=labels, right=False)

combine['Age'].value_counts()
combine['Graduated'].value_counts()
combine['Graduated'].fillna('Unknown', inplace=True)

combine['Graduated'].value_counts()
combine['Profession'].value_counts()
combine['Profession'].fillna('Unknown', inplace=True)

combine['Profession'].value_counts()
combine['Work_Experience'].describe()
combine['Work_Experience'].fillna(-1, inplace=True)

bins= [-1, 0, 3, 6, 9, 12, 15]

labels = [6, 5, 4, 3, 2, 1]

combine['Work_Experience'] = pd.cut(combine['Work_Experience'], bins=bins, labels=labels, right=False)

combine['Work_Experience'].value_counts()
combine['Spending_Score'].value_counts()
combine['Family_Size'].describe()
def get_family(years):

    switcher = {

        1: "F1",

        2: "F2",

        3: "F3",

        4: "F4",

        5: "F5",

        6: "F6",

        7: "F7",

        8: "F8",

    }

    return (switcher.get(years,"F9"))



combine['Family_Size'] = combine['Family_Size'].apply(lambda x: get_family(x))

combine['Family_Size'].value_counts()
combine['Var_1'].value_counts()
combine['Var_1'].fillna('Unknown', inplace=True)

combine['Var_1'].value_counts()
combine.isnull().sum()
train_cleaned = combine[combine['Segmentation'].isnull()!=True].drop(['ID'], axis=1)
train_cleaned.columns
Gender = pd.crosstab(train_cleaned['Gender'], train_cleaned['Segmentation'])

Ever_Married = pd.crosstab(train_cleaned['Ever_Married'], train_cleaned['Segmentation'])

Age = pd.crosstab(train_cleaned['Age'], train_cleaned['Segmentation'])

Graduated = pd.crosstab(train_cleaned['Graduated'], train_cleaned['Segmentation'])

Profession = pd.crosstab(train_cleaned['Profession'], train_cleaned['Segmentation'])

Work_Experience = pd.crosstab(train_cleaned['Work_Experience'], train_cleaned['Segmentation'])

Spending_Score = pd.crosstab(train_cleaned['Spending_Score'], train_cleaned['Segmentation'])

Var_1 = pd.crosstab(train_cleaned['Var_1'], train_cleaned['Segmentation'])







Gender.plot(kind="bar", figsize=(4, 4))

Ever_Married.plot(kind="bar", figsize=(4, 4))

Age.plot(kind="bar", figsize=(4, 4))

Graduated.plot(kind="bar", figsize=(4, 4))

Profession.plot(kind="bar", figsize=(4, 4))

Work_Experience.plot(kind="bar", figsize=(4, 4))

Spending_Score.plot(kind="bar", figsize=(4, 4))

Var_1.plot(kind="bar", figsize=(4, 4))



plt.show()
segmentation = {'A':1, 'B':2, 'C':3, 'D':4}

combine['Segmentation'] = combine['Segmentation'].apply(lambda x: segmentation.get(x))
from sklearn.preprocessing import LabelEncoder



cat_cols = ['Gender', 'Ever_Married', 'Spending_Score', 'Var_1', 'Age', 

            'Family_Size', 'Profession', 'Graduated']



encoder = LabelEncoder()

for col in cat_cols:

    combine[col] = encoder.fit_transform(combine[col])

#combine = pd.get_dummies(combine)

combine.shape
X = combine[combine['Segmentation'].isnull()!=True].drop(['ID','Segmentation'], axis=1)

y = combine[combine['Segmentation'].isnull()!=True]['Segmentation']



X_test = combine[combine['Segmentation'].isnull()==True].drop(['ID','Segmentation'], axis=1)



X.shape, y.shape, X_test.shape
X.head()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators=5000,

                       max_depth = 20,

                       max_features=0.85,

                       learning_rate=1.1)

model.fit(x_train, y_train)
pred_val = model.predict(x_val)

accuracy_score(y_val, pred_val)
confusion_matrix(y_val, pred_val)
segmentation = {1:'A', 2:'B', 3:'C', 4:'D'}

submission = pd.DataFrame()

submission['ID'] = test['ID']

submission['Segmentation'] = [segmentation.get(x) for x in model.predict(X_test)]

submission.head()
submission.to_csv('submission.csv', index=False)