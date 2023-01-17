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



import matplotlib.pyplot as plt







df=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.describe()
df.shape

df.columns
university_ranks=df['University Rating'].value_counts()

gre_scores=df['GRE Score'].value_counts()



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(university_ranks.index,df['University Rating'].value_counts())

ax.set_ylabel('count of univerity')

ax.set_xlabel('Univerity Rating')



plt.show()
df =df.dropna(axis=0)
X = df.iloc[:,0:7]  #independent columns

y = df.iloc[:,-1]



import seaborn as sns

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

#plt.figure(figsize=(8,8))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corrmat
df[df['Chance of Admit ']>0.80].groupby('University Rating').mean()

y = df.iloc[:,-1]

y 
features=['GRE Score', 'TOEFL Score','CGPA','University Rating']

x =df[features]



x
x.describe()
x.head()
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
from sklearn.tree import DecisionTreeRegressor

#Specify a number for random_state to ensure same results each run

marks_model= DecisionTreeRegressor(random_state=1)

#fitting model

marks_model.fit(train_x, train_y)
print("The predictions are")

val_predictions = marks_model.predict(val_x)

print(val_predictions)

from sklearn.metrics import mean_absolute_error

print("error rate " + str(mean_absolute_error(val_y, val_predictions)))
from sklearn.metrics import mean_squared_error

mean_squared_error(val_y,val_predictions, squared=False)