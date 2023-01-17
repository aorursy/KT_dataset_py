# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#Importing the requisite libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Initialise the dataset

df = pd.read_csv('/kaggle/input/kobe-bryant-shot-selection/data.csv.zip')
#Getting a description of the dataset

df.describe()
#Getting the number of rows and columns in the dataset

np.shape(df)
#Scanning for the number of missing values in each column

df.isnull().sum()
#Getting the unique value for each column

for col in df:

    print(df[col].unique())
#Transforming the Date column inro a column  which houses the number of days it's been since the shot took place

df['days_since'] = (pd.to_datetime('2020-07-05') - pd.to_datetime(df['game_date'])).dt.days
#Dropping columns which are unneccesary



#The team's name and ID has no relation to whether the shot placed or not

#The game id also has no relation to if the shot placed or not

#The game date has been translated to the days since line prior to the execution of this line of code 

#The season number serves the sane purpose

df = df.drop(['team_name', 'team_id', 'game_id', 'game_date', 'season'], axis=1)
#Getting an updated description of the dataset

df.head()
#The Train data

train_data = pd.DataFrame(df)

test_data = []

shot_id_test = []
train_data = train_data.dropna(how='any', subset=['shot_made_flag'])
#I am label encoding all the columns which store categorial variable 

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

train_data['action_type'] = le.fit_transform(train_data['action_type'])

train_data['combined_shot_type'] = le.fit_transform(train_data['combined_shot_type'])

train_data['shot_type'] = le.fit_transform(train_data['shot_type'])

train_data['shot_zone_area'] = le.fit_transform(train_data['shot_zone_area'])

train_data['shot_zone_basic'] = le.fit_transform(train_data['shot_zone_basic'])

train_data['shot_zone_range'] = le.fit_transform(train_data['shot_zone_range'])

train_data['matchup'] = le.fit_transform(train_data['matchup'])

train_data['opponent'] = le.fit_transform(train_data['opponent'])
test_data = df[df.isnull().any(axis=1)]

shot_id = test_data['shot_id']

train_data = train_data.drop('shot_id', axis=1)

test_Data = test_data.drop('shot_id', axis=1)
test_data['action_type'] = le.fit_transform(test_data['action_type'])

test_data['combined_shot_type'] = le.fit_transform(test_data['combined_shot_type'])

test_data['shot_type'] = le.fit_transform(test_data['shot_type'])

test_data['shot_zone_area'] = le.fit_transform(test_data['shot_zone_area'])

test_data['shot_zone_basic'] = le.fit_transform(test_data['shot_zone_basic'])

test_data['shot_zone_range'] = le.fit_transform(test_data['shot_zone_range'])

test_data['matchup'] = le.fit_transform(test_data['matchup'])

test_data['opponent'] = le.fit_transform(test_data['opponent'])
print(test_data)
print(train_data['shot_made_flag'])
train_data.head()
X = pd.DataFrame(train_data.drop('shot_made_flag', axis=1))

y = train_data.iloc[:, 14].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)
from sklearn.svm import SVC

svc = SVC(kernel='rbf', random_state=42)

svc.fit(X_train, y_train)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

y_pred_svc = svc.predict(X_test)

y_pred_gnb = gnb.predict(X_test)

y_pred_xgb = xgb.predict(X_test)
print(train_data['shot_made_flag'])
from sklearn.metrics import r2_score

r2_score(y_test, y_pred_xgb)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



#Rest in Peace Kobe
test_data = test_data.drop(['shot_id', 'shot_made_flag'], axis = 1)
print(y_pred_xgb)
y_pred_test = xgb.predict(test_data)
submission_predictions = pd.DataFrame({"shot_id": shot_id, "shot_made_flag": y_pred_test})
submission_predictions.to_csv(r'C:\Users\adity.LAPTOP-F6A6F39F.000\Desktop\submission_predictions.csv', index = False, header=True)