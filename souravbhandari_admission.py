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

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

df=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")

print(df.head())

print(df.isnull().sum())

print(df.columns)

df.columns = (['serial_no','GRE','TOEFL','university_rating','SOP','LOR','CGPA','research','COA'])

df = df.drop('serial_no',axis=1)

df.head()

sns.distplot(df. GRE,bins=20, kde=False,color='purple')

plt.show()

con_feat = ['GRE','TOEFL','CGPA', 'COA']

sns.pairplot(df[con_feat])

plt.show()

y=df['COA']

x=df.drop(['COA'],axis=1)

cat_feature = [col for col in df.columns if df[col].dtypes == "O"]

print(cat_feature)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=10)

model= RandomForestRegressor()

model.fit(x_train,y_train)

predict=model.predict(x_test)

rmse=metrics.mean_squared_error(y_test,predict)

print(rmse)

model1 = LinearRegression()

model1.fit(x_train,y_train)

predict1 = model1.predict(x_test)

rmse1 =metrics.mean_squared_error(y_test,predict1)                   

print(rmse1)
