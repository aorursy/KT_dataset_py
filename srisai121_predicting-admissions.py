# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Importing Random forest classifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df.head()
df = pd.read_csv('../input/Admission_Predict.csv')
df.columns

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit','LOR ':'LOR'})


print(df.info())
# TOEFL Score

plt.scatter(df["TOEFL Score"],df['Chance of Admit'])

plt.title("Chance of Admit for TOEFL Scores")

plt.xlabel("TOEFL Score")

plt.ylabel("Chance of Admit")

plt.show()

# GRE Score

plt.scatter(df["GRE Score"],df['Chance of Admit'])

plt.title("Chance of Admit for GRE Scores")

plt.xlabel("GRE Score")

plt.ylabel("Chance of Admit")

plt.show()

# Correlation between these score and  chance of admit 

print('Correlation between TOEFL Score and Chance of Admit : ',df['TOEFL Score'].corr(df['Chance of Admit']))

print('Correlation between GRE Score and Chance of Admit : ',df['GRE Score'].corr(df['Chance of Admit']))
df.head(10)
X = df.iloc[:, 1:8].values

y = df.iloc[:, 8].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

regressor = RandomForestRegressor(n_estimators =500, random_state = 42)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_pred))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)



y_pred_linReg = regressor.predict(X_test)

print("r_square score: ", r2_score(y_test,y_pred_linReg))
df['Admit'] = (df['Chance of Admit']>0.5).astype(int)

df[['Chance of Admit','Admit']].head(10)
X = df.iloc[:, 1:8].values

y = df.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators= 500,random_state = 0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score,f1_score

cm = confusion_matrix(y_test,y_pred)

print('Random Forrest metrics\n')

print('Confusion Matrix: \n',cm)

print('Precision Score: ',precision_score(y_test,y_pred))

print('Recall Scoree  : ',recall_score(y_test,y_pred))

print('F1 Score       : ',f1_score(y_test,y_pred))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('Logistics Regression metrics\n')

print('Confusion Matrix: \n',cm)

print('Precision Score: ',precision_score(y_test,y_pred))

print('Recall Scoree  : ',recall_score(y_test,y_pred))

print('F1 Score       : ',f1_score(y_test,y_pred))

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print('GaussianNB metrics\n')

print('Confusion Matrix: \n',cm)

print('Precision Score: ',precision_score(y_test,y_pred))

print('Recall Scoree  : ',recall_score(y_test,y_pred))

print('F1 Score       : ',f1_score(y_test,y_pred))
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('SVM metrics\n')

print('Confusion Matrix: \n',cm)

print('Precision Score : ',precision_score(y_test,y_pred))

print('Recall Scoree   : ',recall_score(y_test,y_pred))

print('F1 Score        : ',f1_score(y_test,y_pred))