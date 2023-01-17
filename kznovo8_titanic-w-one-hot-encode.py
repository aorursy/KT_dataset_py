import pandas as pd
import numpy as np
from pathlib import Path
CSV_DIR = '../input/'

train = pd.read_csv(str(Path(CSV_DIR, 'train.csv')))
test  = pd.read_csv(str(Path(CSV_DIR, 'test.csv')))
full_data = pd.concat([train, test])
# fillna
full_data['Age'].fillna(full_data['Age'].median(), inplace=True)
full_data['Embarked'].fillna(full_data['Embarked'].mode()[0], inplace=True)
full_data['Fare'].fillna(full_data['Fare'].median(), inplace=True)
# drop columns
drop_column = ['Cabin', 'Ticket']
full_data.drop(drop_column, axis=1, inplace=True)
# famsize
full_data['FamSize'] = full_data['SibSp'] + full_data['Parch'] + 1
full_data['IsAlone'] = 1
full_data.loc[full_data['FamSize'] > 1, 'IsAlone'] = 0
# title
full_data['Title'] = full_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# farebin
full_data['FareBin'] = pd.qcut(full_data['Fare'], 4)
# agebin
full_data['AgeBin'] = pd.cut(full_data['Age'].astype(int), 5)
# title
title_names = (full_data['Title'].value_counts() < 10)
full_data['Title'] = full_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
full_data.head()
from sklearn.preprocessing import LabelEncoder
# label encode
label = LabelEncoder()
full_data['AgeBinCode'] = label.fit_transform(full_data['AgeBin'])
full_data['FareBinCode'] = label.fit_transform(full_data['FareBin'])
# one-hot encode
Title = pd.get_dummies(full_data['Title'])
Embarked = pd.get_dummies(full_data['Embarked'])
Sex = pd.get_dummies(full_data['Sex'])
df = pd.concat([full_data, Title, Embarked, Sex], axis=1)
added_columns = list(Title.columns) + list(Embarked.columns) + list(Sex.columns) + ['AgeBinCode', 'FareBinCode']
added_columns
Target = ['Survived']
data_x = ['Pclass', 'IsAlone'] + added_columns
X_train = df.loc[np.logical_not(df[Target].isnull().values.T.tolist()[0]), data_x]
y_train = df.loc[np.logical_not(df[Target].isnull().values.T.tolist()[0]), Target].astype(int)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
logreg.score(X_train, y_train.values.ravel())
random_forrest = RandomForestClassifier(n_estimators=100)
random_forrest.fit(X_train, y_train.values.ravel())
random_forrest.score(X_train, y_train.values.ravel())
X_test = df.loc[df[Target].isnull().values.T.tolist()[0], data_x]
Y_pred = random_forrest.predict(X_test)
test_df = df.loc[df[Target].isnull().values.T.tolist()[0], 'PassengerId']
sub = pd.DataFrame({"PassengerId": test_df,"Survived": Y_pred})
sub.head()
# sub.to_csv('submission.csv', index=False)