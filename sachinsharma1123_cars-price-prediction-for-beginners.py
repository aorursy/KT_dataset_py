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
df=pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
df
df.head(5)
df.info()
#this dataset has many irregularity in the columns,so lets separate out the categorical and numerical features

list_cate=[]

list_num=[]

for i in list(df.columns):

    

    if df[i].dtype=='object':

        

        list_cate.append(i)

    else:

        list_num.append(i)

        
list_cate
#there are some other symbols in the normalized-losses columns

df['normalized-losses'].unique()
#lets replace this ? with the value say 145

df['normalized-losses']=df['normalized-losses'].str.replace('?','145')
df['normalized-losses']=df['normalized-losses'].astype('int64')
df['make'].unique()
df['fuel-type'].unique()
df['bore'].unique()
#replace ? with a value say 3.35

df['bore']=df['bore'].str.replace('?','3.35')
df['bore']=df['bore'].astype('float')
df['stroke'].unique()
#similarly replace the symbol ? with a random values from the columns i.e 3.15
df['stroke']=df['stroke'].str.replace('?','3.15')
df['stroke']=df['stroke'].astype('float')
df['horsepower'].unique()
df['horsepower']=df['horsepower'].str.replace('?','130')
df['horsepower']=df['horsepower'].astype('int64')
df['peak-rpm'].unique()
df['peak-rpm']=df['peak-rpm'].str.replace('?','4500')

df['peak-rpm']=df['peak-rpm'].astype('int64')
df['price']=df['price'].str.replace('?','16000')

df['price']=df['price'].astype('int64')
df.info()
#now preprocess the categorical features 

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
df
y=df['price']

x=df.drop(['price'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr=LinearRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=r2_score(y_test,pred_1)
score_1
import matplotlib.pyplot as plt

import seaborn as sns

sns.scatterplot(x=y_test,y=pred_1)
#its a decent fit line
from sklearn.ensemble import RandomForestRegressor

rfg=RandomForestRegressor()

rfg.fit(x_train,y_train)

pred_2=rfg.predict(x_test)

score_2=r2_score(y_test,pred_2)
score_2
sns.scatterplot(x=y_test,y=pred_2)
#slightly better than previous one
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()

gbr.fit(x_train,y_train)

pred_3=gbr.predict(x_test)

score_3=r2_score(y_test,pred_3)
score_3
sns.scatterplot(x=y_test,y=pred_3)
from sklearn.svm import SVR

svm=SVR()

svm.fit(x_train,y_train)

pred_4=svm.predict(x_test)

score_4=r2_score(y_test,pred_4)
score_4
sns.scatterplot(x=y_test,y=pred_4)
#svm regressor gives the worst fit line

#from all the models randomforest gives the best fit line