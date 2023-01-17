# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.tail()
df['total']=df['math score']+df['reading score']+df['writing score']
df.isnull().sum()
list_col=list(df.select_dtypes('object').columns)

for i in list_col:

    print(i)

    print(df[i].unique())

    print('--------------------')
df.describe()
def eda(df_data):

    df_data['gender']=df_data['gender'].map({'female':0,'male':1})

    df_data['lunch']=df_data['lunch'].map({'standard':0,'free/reduced':1})

    df['test preparation course']=df['test preparation course'].map({'none':0,'completed':1})

    return df_data

    
df['test preparation course'].unique()
sns.boxplot(df['gender'],df['total'])

plt.show()
sns.boxplot(df['lunch'],df['total'])

plt.show()
sns.boxplot(df['test preparation course'],df['total'])

plt.show()
eda(df)
df=pd.get_dummies(df,drop_first=True)
df.info()
col=['math score','reading score','writing score','total']
fig,axes=plt.subplots(2,2,figsize=(18,5))

sns.kdeplot(df['math score'],ax=axes[0,0])

sns.kdeplot(df['reading score'],ax=axes[0,1])

sns.kdeplot(df['writing score'],ax=axes[1,0])

sns.kdeplot(df['total'],ax=axes[1,1])

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression
X=df.drop(['total'],axis=1)

y=df.total

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
def model_eval(model,X_train, X_test, y_train, y_test):

    model.fit(X_train,y_train)

    predit=model.predict(X_test)

    mse=mean_squared_error(y_test,predit)

    rmse=np.sqrt(mse)

    r2_scor=r2_score(y_test,predit)

    print('mse',mse)

    print('rmse',rmse)

    print('r2score',r2_scor)
model=LinearRegression()
model_eval(model,X_train, X_test, y_train, y_test)