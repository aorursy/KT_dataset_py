# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])

data.head()
data.isna().sum()
data['Age'].fillna(data['Age'].median(),inplace=True)
import scipy.stats as stat

import pylab
def plot_data(df,feature):

    plt.figure(figsize=(10,6))

    plt.subplot(1,2,1)

    df[feature].hist()   #1st plot

    plt.subplot(1,2,2)

    stat.probplot(df[feature],dist='norm',plot=pylab)

    plt.show()
plot_data(data,'Age')
data['Age_log'] = np.log(data['Age'])

plot_data(data,'Age_log')
data['fare_log'] = np.log(data['Fare'])

plot_data(data,'Fare')
data['Age_sq'] = np.sqrt(data['Age'])

plot_data(data,'Age_sq')
data['Age_reciprocal'] = 1/ data['Age']

plot_data(data,'Age_reciprocal')
data['Age_Sq'] = data['Age'] ** data['Age']

plot_data(data,'Age_sq')
data['Age_boxcox'],parameters = stat.boxcox(data['Age']) 
print(parameters)
plot_data(data,'Age_boxcox')
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])

data.head()
data['Age'].fillna(data.Age.median(),inplace=True)
mean_age = data.Age.mean()

max_min = data.Age.max() - data.Age.min()
data['age_scaled'] = (data['Age'] - mean_age) / max_min

data.head()
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])

#data.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(data)

ss = pd.DataFrame(df_scaled,columns=['Survived','Age','Fare'])

ss.head()
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])

#data.head()
from sklearn.preprocessing import MinMaxScaler

min_max = MinMaxScaler()
minmax_scaled = min_max.fit_transform(data)

minmax = pd.DataFrame(minmax_scaled)

minmax.head()
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])

#data.head()
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
rob_scale = scaler.fit_transform(data)

rs = pd.DataFrame(rob_scale)

rs.head()
data = pd.read_csv('/kaggle/input/titanic/train.csv',usecols=['Age','Fare','Survived'])

#data.head()
from sklearn.preprocessing import MaxAbsScaler

maxabs = MaxAbsScaler()
mx = maxabs.fit_transform(data)

mxb = pd.DataFrame(mx)

mxb.head()