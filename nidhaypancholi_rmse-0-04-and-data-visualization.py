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
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.set_index('Serial No.',inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

df.dtypes

plt.figure(figsize=(5,4))

sns.barplot(x='Research',y='Chance of Admit ',data=df)
plt.figure(figsize=(16,4))

sns.distplot(a=df[df['Chance of Admit ']>=0.5]['TOEFL Score'],label='greater than 0.5')

sns.distplot(a=df[df['Chance of Admit ']<0.5]['TOEFL Score'],label='less than 0.5')

plt.legend()
plt.figure(figsize=(16,4))

sns.distplot(a=df['GRE Score'])
plt.figure(figsize=(16,4))

sns.distplot(a=df[df['Chance of Admit ']>=0.5]['GRE Score'],label='greater than 0.5')

sns.distplot(a=df[df['Chance of Admit ']<0.5]['GRE Score'],label='less than 0.5')

plt.legend()
plt.figure(figsize=(16,4))

sns.scatterplot(x='GRE Score',y='TOEFL Score',data=df[df['Chance of Admit ']>=0.5],label='greater than 0.5')

sns.scatterplot(x='GRE Score',y='TOEFL Score',data=df[df['Chance of Admit ']<0.5],label='less than 0.5')
X_train=df.drop('Chance of Admit ',axis=1)[:400]

X_test=df.drop('Chance of Admit ',axis=1)[401:]

y_train=df['Chance of Admit '][:400]

y_test=df['Chance of Admit '][401:]

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error

for x in range(2,12):

    for y in [10,100,500,1000]:    

        reg=RandomForestRegressor(max_depth=x,n_estimators=y)

        reg.fit(X_train,y_train)

        y_pred=reg.predict(X_test)

        print("depth= ",x,'estimators= ',y,' rmse= ',np.sqrt(mean_squared_error(y_test,y_pred)))

final_reg=RandomForestRegressor(max_depth=4,n_estimators=1000)

final_reg.fit(X_train,y_train)

y_final_pred=final_reg.predict(X_test)

print("Rmse = ",np.sqrt(mean_squared_error(y_test,y_final_pred)))

print('Abs error= ',mean_absolute_error(y_test,y_final_pred))
list(zip(X_train.columns,final_reg.feature_importances_))
X_test.corrwith(y_test)