# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor



df=pd.read_csv('../input/insurance/insurance.csv',encoding='utf-8', engine='c')

df.head()

df.isnull().sum()

df.describe().transpose()



cols=[]

for col,value in df.iteritems():

    if value.dtype=='object':

        cols.append(col)



df_col=df[cols] 

df_col.head()



for i in cols:

    print(i,df[i].unique())



df.drop('region',axis=1,inplace=True)



sns.countplot('sex',data=df,hue='smoker')



plt.figure(figsize=(10,5))

sns.countplot('age',data=df,hue='smoker')



sns.barplot('smoker','charges',data=df)



plt.figure(figsize=(10,5))

sns.boxplot('smoker','charges',data=df)



sns.distplot(df['age']) #distribution of age

sns.lmplot('age','charges',hue='smoker',data=df)



sns.distplot(df['bmi']) #distribution of bmi

sns.distplot(df[(df.bmi>30)]['charges'])



sns.lmplot('bmi','charges',data=df,hue='smoker')

sns.scatterplot('bmi','charges',data=df,hue='smoker')



sns.countplot('children',data=df)

sns.countplot('smoker',data=df[df['children']>0])



sns.heatmap(df.corr(),annot=True)



df=pd.get_dummies(df,drop_first=True)



x=df.drop('charges',axis=1)

y=df['charges']



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)



model=LinearRegression()

model.fit(x_train,y_train)



pred=model.predict(x_test)

score=model.score(x_test,y_test)

print(score)

print(model.coef_)

print(model.intercept_)



print('Mean Absolute Error:',mean_absolute_error(y_test, pred))

print('Mean Squared Error:',mean_squared_error(y_test, pred))

print('Root mean squared erorr: ',np.sqrt(mean_squared_error(y_test,pred)))



#using Random forest

new_model=RandomForestRegressor(n_estimators=1000,random_state=1,n_jobs=1)

new_model.fit(x_train,y_train)

pred_new=new_model.predict(x_test)

score_new=new_model.score(x_test,y_test)

print(score_new)

print('Mean Absolute Error:',mean_absolute_error(y_test, pred_new))

print('Mean Squared Error:',mean_squared_error(y_test, pred_new))

print('Root mean squared erorr: ',np.sqrt(mean_squared_error(y_test,pred_new)))



values=pd.DataFrame({'y_test':y_test,'Pred':pred_new})

print(values)
