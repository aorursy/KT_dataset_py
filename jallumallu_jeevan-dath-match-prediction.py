

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing other libraries

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
data=pd.read_csv('../input/speed-dating-experiment/Speed Dating Data.csv', encoding="ISO-8859-1")

data.head()
print(pd.crosstab(data.gender,data.match))

print(pd.crosstab(data.gender,data.met))

print(pd.crosstab(data.gender,data.dec))
#so for females the match ratio is:

F_match_ratio=690/3494

print("For females the match ratio is",F_match_ratio*100)

#Ratio of meeting a male for a female:

F_met=1-2018/(2018+183+1791+1+2+1+2+1)

print("Ratio of meeting a male for a female",F_met*100)

#decision_of_female_accpeting_male

F_Dec=2655/(2655+1529)

print("Decision_of_female_accpeting_male",F_Dec*100)

#so for males the match ratio is:

M_match_ratio=690/3504

print("For males the match ratio is:",M_match_ratio*100)

#Ratio of meeting a Female for a male:

M_met=1-2029/(2029+168+1806+1)

print("Ratio of meeting a Female for a male:",M_met*100)

#decision_of_male_accpeting_female

M_Dec=2205/(2205+1989)

print("Decision_of_male_accpeting_female",M_Dec*100)
plt.hist(data.age.values)

plt.xlabel('Age')

plt.ylabel('Frequency')
for column in data.columns:

    if(data[column].isnull().sum())>1000:

        data.drop([column],axis=1,inplace=True)
data.shape
data.dropna(inplace=True)

data.shape
data.dtypes
data.drop(['career','from','field'],axis=1,inplace=True)

data.shape
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("Correlation Heatmap")

corr = data.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#prepare the data

X=data[['like','dec']]

y=data['match']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression(random_state=0)

lrc = model.fit(X_train, y_train)

predict_train_lrc = lrc.predict(X_train)

predict_test_lrc = lrc.predict(X_test)

print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_lrc))

print('Production Accuracy:', metrics.accuracy_score(y_test, predict_test_lrc))
model = RandomForestClassifier()

rf_model = model.fit(X_train, y_train)

predict_train_rf = rf_model.predict(X_train)

predict_test_rf = rf_model.predict(X_test)

print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_rf))

print('Production Accuracy:', metrics.accuracy_score(y_test, predict_test_rf))
model = GradientBoostingClassifier()

xgb_model = model.fit(X_train, y_train)

predict_train_xgb = xgb_model.predict(X_train)

predict_test_xgb = xgb_model.predict(X_test)

print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_xgb))

print('Production Accuracy:', metrics.accuracy_score(y_test, predict_test_xgb))