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
df=pd.read_csv('/kaggle/input/heights-and-weights/data.csv')
df
#now we have to create a column for bmi



#we know that bmi=weight/height*height
df['BMI']=df['Weight']/(df['Height']*df['Height'])
df
import seaborn as sns

import matplotlib.pyplot as plt

sns.lineplot(x=df['Height'],y=df['Weight'],data=df)
#clearly weight is directly proportional to the weight
y=df['BMI']

x=df.drop(['BMI'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

list_models=[]

list_errors=[]

list_scores=[]

lr=LinearRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=r2_score(y_test,pred_1)

error_1=mean_squared_error(y_test,pred_1)
list_scores.append(score_1)

list_models.append('linear regression')

list_errors.append(error_1)
score_1
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()

rfr.fit(x_train,y_train)

pred_2=rfr.predict(x_test)

score_2=r2_score(y_test,pred_2)

error_2=mean_squared_error(y_test,pred_2)
list_scores.append(score_2)

list_models.append('randomforest regressor')

list_errors.append(error_2)
score_2
from sklearn.svm import SVR

svm=SVR()

svm.fit(x_train,y_train)

pred_3=svm.predict(x_test)

score_3=r2_score(y_test,pred_3)

error_3=mean_squared_error(y_test,pred_3)

list_models.append('svr')

list_errors.append(error_3)

list_scores.append(score_3)
score_3
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()

gbr.fit(x_train,y_train)

pred_4=gbr.predict(x_test)

score_4=r2_score(y_test,pred_4)

error_4=mean_squared_error(y_test,pred_4)

list_models.append('gradientboosting regressor')

list_errors.append(error_4)

list_scores.append(score_4)
score_4
plt.figure(figsize=(12,4))

plt.bar(list_models,list_errors,width=0.2)

plt.xlabel('models')

plt.ylabel('mean squared error')

plt.show()
plt.figure(figsize=(12,4))

plt.bar(list_models,list_scores,width=0.2)

plt.xlabel('models')

plt.ylabel('r2 scores')

plt.show()