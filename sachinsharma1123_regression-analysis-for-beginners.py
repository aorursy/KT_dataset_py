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
df=pd.read_csv('/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv')
df
df.isnull().sum()
df.info()
#in this dataset all the columns are of object type ,so we have to convert them to float datatype
for i in list(df.columns):

    df[i]=df[i].str.replace(',','.')
#drop the date column

df=df.drop(['date'],axis=1)
#now convert all the columns into float datatype

for i in list(df.columns):

    df[i]=df[i].astype('float')
df
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(df.corr())
#now the dataset is ready for model building

y=df['% Silica Concentrate']

x=df.drop(['% Silica Concentrate'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

list_models=[]

list_score=[]

list_error=[]

lr=LinearRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=r2_score(y_test,pred_1)
list_score.append(score_1)

list_models.append('linear regression')



error_1=mean_squared_error(y_test,pred_1)

list_error.append(error_1)
sns.scatterplot(x=y_test,y=pred_1)
#seems to be a good git line
from sklearn.linear_model import Ridge

ridge=Ridge()
ridge.fit(x_train,y_train)

pred_2=ridge.predict(x_test)

score_2=r2_score(y_test,pred_2)

error_2=mean_squared_error(y_test,pred_2)

list_models.append('ridge')

list_error.append(error_2)

list_score.append(score_2)
score_2
sns.scatterplot(x=y_test,y=pred_2)
from sklearn.linear_model import Lasso

lasso=Lasso()

lasso.fit(x_train,y_train)

pred_3=lasso.predict(x_test)

score_3=r2_score(y_test,pred_3)

error_3=mean_squared_error(y_test,pred_3)
list_models.append('lasso')

list_error.append(error_3)

list_score.append(score_3)
score_3
sns.scatterplot(x=y_test,y=pred_3)
#plot bw mean squared error and models

sns.lineplot(x=list_models,y=list_error)
list_models,list_score
sns.barplot(x=list_models,y=list_score)
#from here we can conclude that linear and ridge regression has good r2 score and low mean squared error