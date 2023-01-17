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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.isna().sum()
df.describe()
passing_marks = 40
df['Maths Status'] = np.where(df['math score']<passing_marks,'F','P')
df['Maths Status'].value_counts()
sns.countplot(df['Maths Status'],hue=df['gender'],palette='YlOrRd_r')
df['Reading Status'] = np.where(df['reading score']<passing_marks,'F','P')
df['Reading Status'].value_counts()
sns.countplot(df['Reading Status'],hue = df['gender'],palette='winter' )
df['Writing Status'] = np.where(df['writing score']<passing_marks,'F','P')
df['Writing Status'].value_counts()
sns.countplot(df['Writing Status'],hue=df['gender'],palette='bright')
df['Total Score'] = (df['math score']+ df['reading score']+ df['writing score'])/3
df['Pass Status'] = np.where(df['Total Score']<passing_marks,'F','P')
df['Pass Status'].value_counts()
sns.countplot(df['Pass Status'],hue=df['gender'])
def grades(pass_status,total_marks):
    if pass_status == 'F':
        return 'F'
    if total_marks >=90:
        return 'A+'
    if total_marks >=80:
        return 'A'
    if total_marks >=70:
        return 'B'
    if total_marks >=60:
        return 'C'
    if total_marks >=50:
        return 'D'
    if total_marks >=40:
        return 'E'

df['Total_Grade'] = df.apply(lambda x : grades(x['Pass Status'],x['Total Score']),axis=1)
df.head()
df['gender'].value_counts()
df['race/ethnicity'].value_counts()
df['parental level of education'].value_counts()
df['lunch'].value_counts()
df['test preparation course'].value_counts()
df.columns
df.drop(['math score', 'reading score','writing score','Maths Status', 'Reading Status', 'Writing Status','Pass Status'],axis = 1,inplace=True)
df.head()
X = df.drop('Total Score',axis = 1)
X = pd.get_dummies(X,drop_first=True)
plt.figure(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)
X.head()
y = df['Total Score']
x_train,y_train,x_test,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)
linear = LinearRegression()
linear.fit(x_train,x_test)
predict = linear.predict(y_train)
mse_linear = mean_squared_error(y_test,predict)
print(mse_linear)
rmse_linear = np.sqrt(mse_linear)
print(rmse_linear)
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(x_train,x_test)
predict_xgb = xgb.predict(y_train)
mse_xgb = mean_squared_error(y_test,predict_xgb)
print(mse_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print(rmse_xgb)
