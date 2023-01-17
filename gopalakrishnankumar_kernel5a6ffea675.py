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



#pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')    

#pd.read_csv('/kaggle/working/submission.csv')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current se
import pandas as pd

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

df1=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

df2=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

df2['TargetValue']=df1['TargetValue']

df1.rename(columns={'Id':'ForecastId_Quantile'},inplace = True)

df2.rename(columns={'ForecastId':'ForecastId_Quantile'},inplace = True)

df2['ForecastId_Quantile']=df2['ForecastId_Quantile'].apply(str)

df2['ForecastId_Quantile']=df2['ForecastId_Quantile']+'_0.05'

df=pd.concat([df1, df2],ignore_index=True,axis=0)

#print(df)

y = df['TargetValue']

#df = df.drop(columns='TargetValue')

#df3=df[['Country_Region','Weight']] 

df3=df[['ForecastId_Quantile','Population','Weight','TargetValue']]

# define the target variable (dependent variable) as y

# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(df3, y, test_size=0.2,random_state=42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

# fit a model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

plt.xlabel("True Values")

plt.ylabel("Predictions")

print( "Score:", model.score(X_test, y_test))

submission=[]

submission=df3[['ForecastId_Quantile']]

submission['TargetValue']=df3[['TargetValue']]

#submission=submission[0:935010]

#submission.to_csv('submission.csv', index=False)

print(submission)

import pandas as pd

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

df1=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

df4=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

df4['TargetValue']=df1['TargetValue']

df1.rename(columns={'Id':'ForecastId_Quantile'},inplace = True)

df4.rename(columns={'ForecastId':'ForecastId_Quantile'},inplace = True)

df4['ForecastId_Quantile']=df4['ForecastId_Quantile'].apply(str)

df4['ForecastId_Quantile']=df4['ForecastId_Quantile']+'_0.5'

df_1=pd.concat([df1, df4],ignore_index=True,axis=0)

#print(df)

y_1 = df_1['TargetValue']

#df = df.drop(columns='TargetValue')

#df3=df[['Country_Region','Weight']] 

df4=df_1[['ForecastId_Quantile','Population','Weight','TargetValue']]

# define the target variable (dependent variable) as y

# create training and testing vars

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df4, y_1, test_size=0.2,random_state=42)

print(X_train_1.shape, y_train_1.shape)

print(X_test_1.shape, y_test_1.shape)

# fit a model

lm = linear_model.LinearRegression()

model = lm.fit(X_train_1, y_train_1)

predictions = lm.predict(X_test_1)

plt.scatter(y_test_1, predictions)

plt.xlabel("True Values")

plt.ylabel("Predictions")

print( "Score:", model.score(X_test_1, y_test_1))

submission_1=[]

submission_1=df4[['ForecastId_Quantile']]

submission_1['TargetValue']=df4[['TargetValue']]

#submission=submission[0:935010]

#submission.to_csv('submission.csv', index=False)

print(submission_1)

import pandas as pd

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

df1=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

df5=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

df5['TargetValue']=df1['TargetValue']

df1.rename(columns={'Id':'ForecastId_Quantile'},inplace = True)

df5.rename(columns={'ForecastId':'ForecastId_Quantile'},inplace = True)

df5['ForecastId_Quantile']=df5['ForecastId_Quantile'].apply(str)

df5['ForecastId_Quantile']=df5['ForecastId_Quantile']+'_0.95'

df_2=pd.concat([df1, df5],ignore_index=True,axis=0)

#print(df)

y_2 = df_2['TargetValue']

#df = df.drop(columns='TargetValue')

#df3=df[['Country_Region','Weight']] 

df5=df_2[['ForecastId_Quantile','Population','Weight','TargetValue']]

# define the target variable (dependent variable) as y

# create training and testing vars

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df5, y_2, test_size=0.2,random_state=42)

print(X_train_2.shape, y_train_2.shape)

print(X_test_2.shape, y_test_2.shape)

# fit a model

lm = linear_model.LinearRegression()

model = lm.fit(X_train_2, y_train_2)

predictions = lm.predict(X_test_2)

plt.scatter(y_test_2, predictions)

plt.xlabel("True Values")

plt.ylabel("Predictions")

print( "Score:", model.score(X_test_2, y_test_2))

submission_2=[]

submission_2=df5[['ForecastId_Quantile']]

submission_2['TargetValue']=df5[['TargetValue']]

#submission=submission[0:935010]

#submission.to_csv('submission.csv', index=False)

print(submission_2)

submission_3=pd.concat([submission[727230:1038900], submission_1[727230:1038900], submission_2[727230:1038900]],ignore_index=True,axis=0)

submission_3.to_csv('submission.csv', index=False)

print(submission_3)