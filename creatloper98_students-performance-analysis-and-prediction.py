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
import pandas as pd

df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.sample(5)

df.columns = ['Gender','Race','Parent_Education','Lunch','Test_Prep','Math_Score','Reading_Score','Writing_Score']
df.isnull().sum()
df['total_score'] = (df['Math_Score'] + df['Reading_Score'] +df['Writing_Score'])/3

df.head(2)
df.groupby(['Race','Parent_Education']).mean()
df.groupby(['Gender']).mean()
df.groupby(['Parent_Education']).mean()
df.groupby(['Race']).mean()
df.groupby(['Test_Prep']).mean()
df['Test_Prep'].value_counts()
df.groupby(['Race','Test_Prep']).mean()
import seaborn as snb

race_gender = df.groupby(['Race','Test_Prep']).mean().reset_index()

snb.factorplot(x='Race', y='total_score', hue='Test_Prep', data = race_gender, kind='bar')
prep_gender = df.groupby(['Gender','Test_Prep']).mean().reset_index()

snb.factorplot(x='Gender', y='total_score', hue='Test_Prep', data=prep_gender, kind='bar')
parent_gender = df.groupby(['Test_Prep','Parent_Education']).mean().reset_index()

snb.factorplot(x='Test_Prep', y='total_score', hue='Parent_Education', data=parent_gender, kind='bar')
snb.boxplot( y = 'total_score' ,x ='Test_Prep' , data = df)
import matplotlib.pyplot as plt

bplot = snb.boxplot( y = 'total_score' ,x ='Parent_Education'  ,data = df  )

_ = plt.setp(bplot.get_xticklabels(), rotation=90)
bplot = snb.boxplot( y = 'total_score' ,x ='Test_Prep'  ,data = df  )
bplot = snb.boxplot( y = 'total_score' ,x ='Lunch'  ,data = df  )
passing_marks = 40

df['Math_Grade'] = np.where(df['Math_Score']<passing_marks ,'F','P')

df['Math_Grade'].value_counts()
df['Read_Grade'] = np.where(df['Reading_Score']<passing_marks ,'F','P')

df['Read_Grade'].value_counts()
df['Write_Grade'] = np.where(df['Writing_Score']<passing_marks ,'F','P')

df['Write_Grade'].value_counts()
snb.countplot(df['Math_Grade'],hue=df['Gender'],palette='YlOrRd_r')
snb.countplot(df['Read_Grade'],hue = df['Gender'],palette='winter' )
snb.countplot(df['Write_Grade'],hue = df['Gender'],palette='bright' )
df['Pass_Status'] = np.where(df['total_score']<passing_marks,'F','P')

df['Pass_Status'].value_counts()
snb.countplot(df['Pass_Status'],hue=df['Gender'],palette = 'winter')
snb.countplot(df['Pass_Status'],hue=df['Lunch'],palette = 'winter')
snb.countplot(df['Pass_Status'],hue=df['Parent_Education'],palette = 'winter')
df['Lunch'].value_counts()
def grades(pass_status,total_score):

    if pass_status == 'F':

        return 'F'

    if total_score >=90:

        return 'A+'

    if total_score >=80:

        return 'A'

    if total_score >=70:

        return 'B'

    if total_score >=60:

        return 'C'

    if total_score >=50:

        return 'D'

    if total_score >=40:

        return 'E'
df['Grade'] =df.apply(lambda x : grades(x['Pass_Status'] ,x['total_score']) ,axis =1)

df.head(2)
df['Grade'].value_counts()
snb.countplot(data = df , x = 'Grade')
p = snb.countplot(data =df , x ='Parent_Education' ,palette ='winter')

_ = plt.setp(p.get_xticklabels() ,rotation =90)
df.head(2)
df.head(2)
X = df.drop('total_score',axis = 1)

X = pd.get_dummies(X)
X.head()
plt.figure(figsize =(20,20))

snb.heatmap(X.corr(),annot=True)
y = df['total_score']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

x_train , y_train ,x_test ,y_test = train_test_split(X,y,test_size = 0.3 ,random_state =101)
lr = LinearRegression()

lr.fit(x_train,x_test)

predict = lr.predict(y_train)

mse = mean_squared_error(y_test ,predict)

print(mse)

print(lr.coef_)

print(lr.intercept_)
rmse = np.sqrt(mse)

rmse
from xgboost import XGBRegressor
xgb = XGBRegressor()

xgb.fit(x_train,x_test)

predict = xgb.predict(y_train)

mse = mean_squared_error(y_test ,predict)

print(mse)

rmse = np.sqrt(mse)

rmse