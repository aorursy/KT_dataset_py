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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression,Ridge

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
data1=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data1
data2 = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

data2
print(data1.columns)

print(data2.columns)
# Since both the dataframes have same columns so we can combine them together



df = pd.concat([data1, data2])

df.head()
df.shape
df.isna().any()
df.describe()
sns.distplot(df['GRE Score'])
sns.distplot(df['TOEFL Score'])
sns.distplot(df['SOP'])
sns.distplot(df['CGPA'])
df.hist(figsize=(10,10),edgecolor="k")

plt.tight_layout()

plt.show()
# Heatmap



plt.figure(figsize=(12,12))

sns.heatmap(df.corr(),annot=True,linewidths=1.0)

plt.show()
# Find correlation between data columns

df.corr()
X=df.iloc[:,:-1]

Y=df.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=177)
sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
models=[LinearRegression(),

        RandomForestRegressor(n_estimators=200,max_depth=5),

        DecisionTreeRegressor(random_state=62,max_depth=5),GradientBoostingRegressor(),

        KNeighborsRegressor(n_neighbors=50),BaggingRegressor()]

model_names=['LinearRegression','RandomForestRegressor','DecisionTree','GradientBoostingRegressor','KNN',

             'BaggingReg']



R2_SCORE=[]

MSE=[]



for model in range(len(models)):

    print('')

    print("*"*40,"\n",model_names[model])

    reg=models[model]

    reg.fit(X_train,Y_train)

    pred=reg.predict(X_test)

    R2_score=r2_score(Y_test,pred)

    mse=mean_squared_error(Y_test,pred)

    R2_SCORE.append(R2_score)

    MSE.append(mse)

    print("R2 Score",R2_score)

    print("MSE",mse)