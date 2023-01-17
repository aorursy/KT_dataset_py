import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data',header=None)
df.head()
df.shape[0] # Number of rows
df.shape[1] # Number of columns
df.info()
df.iloc[:,0].str.contains('\?').sum()
df.iloc[:,1].str.contains('\?').sum()
df.iloc[:,2].str.contains('\?').sum()
df.iloc[:,3].str.contains('\?').sum()
df.iloc[:,4].str.contains('\?').sum()
df.iloc[:,5].str.contains('\?').sum()

# Missing values are present here.
df.iloc[:,7].str.contains('\?').sum()
df.iloc[:,8].str.contains('\?').sum()
df.iloc[:,9].str.contains('\?').sum()
import numpy as np
# 2 columns have missing vales as ?

df.iloc[:,8].replace('?',np.nan,inplace=True) # inplace changes the df permanently
df.iloc[:,8].str.contains('\?').sum()
df.iloc[:,5].replace('?',np.nan,inplace=True)
df.iloc[:,5].str.contains('\?').sum()
df.isna().sum()
df.head()
# df.dropna(inplace=True)
df.isna().sum().sum()
df.shape
data = pd.read_csv('mtcars.csv')
data.head()
data.isna().sum()
data.info()
import numpy as np
np.asarray(data.hp).mean()
data.hp.dropna()
# mean

mean = np.asarray(data.hp.dropna()).mean()
data.hp.fillna(mean)
data.hp
data.hp.fillna(mean,inplace=True)
data.isna().sum()
data.gear.value_counts()
from scipy import stats
gear = np.asarray(data.gear.dropna())
mode = stats.mode(gear).mode[0]
data.gear.fillna(mode,inplace=True)
data.isna().sum()
df = pd.read_csv('StudentsPerformance.csv')
df.head()
df.isna().sum()
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='mean',missing_values='NaN',axis=0)

imputer = imputer.fit(df.loc[:,['math score']])
df.loc[:,['math score']] = imputer.transform(df.loc[:,['math score']])
imputer.fit_transform
df.isna().sum()
df.lunch
# forward fill

df.fillna(method='ffill')
# backword fill

df.fillna(method='bfill')
# Dropping the duplicates.

data.drop_duplicates()