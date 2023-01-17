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
df=pd.read_csv("../input/titanic/train.csv",usecols=['Cabin','Survived'])

df.head()


### Replacing

df['Cabin'].fillna('Missing',inplace=True)

df.head()
df['Cabin'].unique()

df['Cabin']=df['Cabin'].astype(str).str[0]

df.head()
df.Cabin.unique()

prob_df=df.groupby(['Cabin'])['Survived'].mean()



prob_df=pd.DataFrame(prob_df)

prob_df
prob_df['Died']=1-prob_df['Survived']

prob_df['Probability_ratio']=prob_df['Survived']/prob_df['Died']

prob_df.head()

probability_encoded=prob_df['Probability_ratio'].to_dict()

df['Cabin_encoded']=df['Cabin'].map(probability_encoded)

df.head()
df.head(20)

df=pd.read_csv('../input/titanic/train.csv', usecols=['Pclass','Age','Fare','Survived'])

df.head()
df['Age'].fillna(df.Age.median(),inplace=True)

df.isnull().sum()

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

%matplotlib inline

# import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (14,8)
plt.hist(df['Fare'],bins=20)

from sklearn.preprocessing import MinMaxScaler

min_max=MinMaxScaler()

df_minmax=pd.DataFrame(min_max.fit_transform(df),columns=df.columns)

df_minmax.head()
plt.hist(df_minmax['Pclass'],bins=20)

plt.hist(df_minmax['Age'],bins=20)

plt.hist(df_minmax['Fare'],bins=20)

from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

df_robust_scaler=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

df_robust_scaler.head()
plt.hist(df_robust_scaler['Age'],bins=20)

df=pd.read_csv('../input/titanic/train.csv', usecols=['Age','Fare','Survived'])

df.head()
df['Age']=df['Age'].fillna(df['Age'].median())

df.isnull().sum()

import scipy.stats as stat

import pylab
#### If you want to check whether feature is guassian or normal distributed

#### Q-Q plot

def plot_data(df,feature):

    plt.figure(figsize=(14,8))

    plt.subplot(1,2,1)

    df[feature].hist()

    plt.subplot(1,2,2)

    stat.probplot(df[feature],dist='norm',plot=pylab)

    plt.show()
plot_data(df,'Age')



import numpy as np

df['Age_log']=np.log(df['Age'])

plot_data(df,'Age_log')
df['Age_reciprocal']=1/df.Age

plot_data(df,'Age_reciprocal')
df['Age_sqaure']=df.Age**(1/2)

plot_data(df,'Age_sqaure')
df['Age_exponential']=df.Age**(1/1.2)

plot_data(df,'Age_exponential')
df['Age_Boxcox'],parameters=stat.boxcox(df['Age'])

plot_data(df,'Age_Boxcox')

plot_data(df,'Fare')

df['Fare_log']=np.log1p(df['Fare'])

plot_data(df,'Fare_log')
df['Fare_Boxcox'],parameters=stat.boxcox(df['Fare']+1)

plot_data(df,'Fare_Boxcox')