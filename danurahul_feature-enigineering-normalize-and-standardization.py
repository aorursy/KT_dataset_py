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
df =pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Feature-Engineering-Live-sessions/master/titanic.csv',usecols=['Pclass','Age','Fare','Survived'])
df.head()
df['Age'].fillna(df.Age.median(),inplace=True)
df.isnull().sum()
##standardiztion we use standard scaler 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
##fit versrus fit_transform
df_scaled=scaler.fit_transform(df)
pd.DataFrame(df_scaled)
##it is happening along the columns
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(df_scaled[:,3],bins=20)
plt.hist(df['Fare'],bins=20)
##right skewed
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
df_minmax=pd.DataFrame(min_max.fit_transform(df),columns=df.columns)
df_minmax
plt.hist(df_minmax['Fare'],bins=20)
#right skewed
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df_robust_scaled=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df_robust_scaled.head()
plt.hist(df_robust_scaled['Fare'])
#more robust to outlier
df =pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Feature-Engineering-Live-sessions/master/titanic.csv',usecols=['Age','Fare','Survived'])
df.head()
df['Age']=df['Age'].fillna(df['Age'].median())
df.head()
df.isnull().sum()
import scipy.stats as stats
import pylab
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stats.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()
plot_data(df,'Age')
###logritmic transformation
df['Age_log']=np.log(df['Age'])
plot_data(df,'Age_log')
###reciprocal transformation
df['Age_reciprocal']=1/df.Age
plot_data(df,'Age_reciprocal')
###square root transformation
df['Age_sqrot']=df.Age**(1/2)
plot_data(df,'Age_sqrot')
###exponential transformation
df['Age_exp']=df.Age**(1/1.2)
plot_data(df,'Age_exp')

###boxcox transformation
#T(Y)=(Yexp(lambda)-1)/lambda
df['Age_boxcox'],parameters=stats.boxcox(df['Age'])
print(parameters)
plot_data(df,'Age_boxcox')
## fare log1p used for zero error in log
df['Fare_log']=np.log1p(df['Fare'])
plot_data(df,'Fare_log')
### +1 to prevent error
df['Fare_boxcox'],parameters=stats.boxcox(df['Fare']+1)
plot_data(df,'Fare_boxcox')
