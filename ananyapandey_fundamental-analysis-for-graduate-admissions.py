# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/Admission_Predict.csv')
df2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df=pd.concat([df1,df2])
df.head()
df.shape
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df.columns
# Visual manner to check the null values in the dataset
df.drop(['Serial No.'], inplace=True,axis='columns')
sns.heatmap(data=df.isnull(),cmap='viridis')
sns.set()
sns.distplot(tuple(df['Chance of Admit ']), color='green', bins=40)
sns.distplot(df['University Rating'])
sns.distplot(df['University Rating'],kde=False)
def modiffy(row):
    if row['Chance of Admit '] >0.7 :
        return 1
    else :
        return 0
df['Admit'] = df.apply(modiffy,axis=1)
dftemp = df.drop(['Chance of Admit '], axis=1)
sns.pairplot(dftemp,hue='Admit')
del dftemp
sns.heatmap(df.corr(),annot=True)
sns.scatterplot(data=df,x='GRE Score',y='TOEFL Score',hue='Research')
sns.scatterplot(data=df, y='Chance of Admit ', x='CGPA', hue='Research')
sns.boxplot(data=df,x='SOP',y='Chance of Admit ', hue ='Research')
from sklearn.model_selection import train_test_split
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']]
Y = df['Admit']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Fitting the model by using the training data
# Creating the predictions using our logistic regression model 
predictions = logmodel.predict(X_test)
# Creating the classigication report for the model to check the sensetivity and specificity
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.linear_model import LinearRegression
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']]
Y = df['Chance of Admit ']
lm = LinearRegression()
lm.fit(X_train,y_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# Pringing The Coefficients 
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=20)
