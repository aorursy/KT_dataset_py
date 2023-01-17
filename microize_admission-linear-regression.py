import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
d=open('../input/graduate-admissions/Admission_Predict.csv')
for x in range(5):
    print(d.readline())
df=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv',index_col=['Serial No.'],sep=',')
df.head()
df.columns
df.columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research', 'Chance of Admit']
y=df['Chance of Admit']
X=df.drop(['Chance of Admit'],axis=1)
X.head()
ss=StandardScaler()
X=ss.fit_transform(X)
X=pd.DataFrame(X,columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research'])
X
df_corr=X.corr()
df_corr
sns.heatmap(df_corr)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=5)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)

r2_score(y_test,y_predict)