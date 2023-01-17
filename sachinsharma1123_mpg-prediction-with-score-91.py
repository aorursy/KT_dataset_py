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
df=pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')
df
df.isnull().sum()
df['car name'].unique()
import matplotlib.pyplot as plt

import seaborn as sns
sns.scatterplot(x=df['displacement'],y=df['mpg'],data=df)
#from this figure we can conclude that mpg of a car decreases on increasing with displacement
sns.scatterplot(x=df['cylinders'],y=df['mpg'],data=df)
#not any fixed fattern visible
sns.scatterplot(x=df['horsepower'],y=df['mpg'],data=df)
#again not a clear pattern visible
sns.scatterplot(x=df['acceleration'],y=df['mpg'],data=df)
#here on increasing the acceleration the mpg increases
sns.scatterplot(x=df['weight'],y=df['mpg'],data=df)
#here the mpg vaires inversely with the weight
sns.heatmap(df.corr())
df=df.drop(['car name'],axis=1)
df
df.info()
#there are some unnecessary symbols in the horsepower columns so delete that row

df['horsepower']=df['horsepower'].str.replace(r'\D+','')
del_row=df[df['horsepower']==''].index

df=df.drop(del_row)
df['horsepower']=df['horsepower'].astype('float')
y=df['mpg']

x=df.drop(['mpg'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr=LinearRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=r2_score(y_test,pred_1)
score_1
sns.scatterplot(x=y_test,y=pred_1)
#seems to be a good fit line
from sklearn.ensemble import RandomForestRegressor

rfg=RandomForestRegressor()
rfg.fit(x_train,y_train)

pred_2=rfg.predict(x_test)

score_2=r2_score(y_test,pred_2)
score_2
sns.scatterplot(x=y_test,y=pred_2)
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(x_train,y_train)

pred_3=gbr.predict(x_test)

score_3=r2_score(y_test,pred_3)

score_3
plt.scatter(y_test,pred_3)

plt.show()
#all the models fits quite well but random forest gives the best r2 score