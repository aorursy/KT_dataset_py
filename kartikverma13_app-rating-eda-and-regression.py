# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/playstore-analysis/googleplaystore.csv")
df.head()
df.isnull().sum()
df.shape
df.dropna(inplace=True)
df.shape
def change(x):

    if 'M' in x:

        z=x[:-1]

        z=float(z)*1000

        return z

    

    elif 'k' in x:

        z=x[:-1]

        z=float(z)

        return z

    

    else : return None

    

df.Size = df.Size.map(change)    

    

    
df.Size
df["Size"].isnull().sum()
df["Size"].fillna(method='pad',inplace=True)

df["Size"].isnull().sum()
df["Reviews"]=df["Reviews"].astype('float')
df.Price = df.Price.apply(lambda x: x.replace('$',''))

df.Price=df.Price.astype('float')

df.Installs = df.Installs.apply(lambda x: x.replace(',','').replace('+',''))

df.Installs=df.Installs.astype('float')

df.dtypes
df["Rating"].shape
a=df.Rating>5
a.value_counts()
b=(df.Type=='Free')&(df.Price>0)
b.value_counts()
c=df.Reviews>df.Installs
c.value_counts()
df=df[df.Reviews<df.Installs].copy()

print(df.shape)
df=df[df.Price<200].copy()

print(df.shape)
d = df.Reviews>2000000
d.value_counts()
df=df[df.Reviews<=2000000].copy()

print(df.shape)
df.boxplot()
df.Installs=df.Installs.apply(func=np.log1p)

df.Reviews=df.Reviews.apply(func=np.log1p)



df.hist(column=['Installs','Reviews'])
plt.figure(figsize=(25,8))

sns.scatterplot(df.Price,df.Rating,hue=df.Rating)

plt.show()
plt.figure(figsize=(25,8))

sns.scatterplot(df.Size,df.Rating,hue=df.Rating)

plt.show()
plt.figure(figsize=(25,8))

sns.scatterplot(df.Reviews,df.Rating,hue=df.Rating)

plt.show()
plt.figure(figsize=(25,8))

sns.boxplot(df["Content Rating"],df["Rating"])

plt.show()
plt.figure(figsize=(25,8))

sns.boxplot(df.Category,df.Rating)

plt.xticks(fontsize=18,rotation='vertical')

plt.show()
df.drop(["App","Last Updated","Current Ver","Android Ver"],inplace=True,axis=1)
df=pd.get_dummies(df,drop_first=True)
x=df.iloc[:,1:]

y=df.iloc[:,:1]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.30, random_state=1)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

model=regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test)
from statsmodels.api import OLS

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error as ms
summ=OLS( y_train,x_train).fit()

summ.summary()

print('R2_Score=',r2_score(y_test,y_pred))

print('Root_Mean_Squared_Error(RMSE)=',np.sqrt(ms(y_test,y_pred)))