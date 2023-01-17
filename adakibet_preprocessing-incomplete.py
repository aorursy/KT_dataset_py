import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn import impute



from scipy.stats import skew, norm, boxcox

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from scipy.stats.mstats import winsorize



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/insurance/insurance.csv')

data.head(8)
#Null values

data.isna().sum()
data.info()
#skewness in charges

sns.distplot(data['charges'], hist = True)

plt.title('Charges Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('Charges')
# log transformation

data['charges'] = np.log1p(data['charges'])
#After log transformation 

sns.distplot(data['charges'], hist = True)

plt.title('Charges Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('Charges')
#Skewness in other variables (before transformation)

skews = data.skew(axis = 0)

skews
#skewness in other features

#age

sns.distplot(data['age'], hist = True)

plt.title('age Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('age')
#transforming age

data['age'] = boxcox1p(data['age'], boxcox_normmax(data['age'] + 1))
#after transformation

sns.distplot(data['age'], hist = True)

plt.title('age Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('age')
#bmi

sns.distplot(data['bmi'], hist = True)

plt.title('bmi Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('bmi')
#skewness in other features

sns.distplot(data['children'], hist = True)

plt.title('children Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('childrem')
#transforming children

data['children'] = np.log1p(data['children'])
#after transformation

sns.distplot(data['children'], hist = True)

plt.title('children Frequency plot')

plt.ylabel('Frequency')

plt.xlabel('childrem')
#Skewness after transformation

skews = data.skew(axis = 0)

skews
f, ax = plt.subplots(figsize=(9, 8))

sns.boxplot(data = data)
#the clip function (replaces upper outliers with 95th quantile and lower with 5th quantiles)

data['bmi'] = data['bmi'].clip(lower=data['bmi'].quantile(0.05), upper=data['bmi'].quantile(0.95))
f, ax = plt.subplots(figsize=(9, 8))

sns.boxplot(data = data)
#Encoding categoricals

data = pd.get_dummies(data, drop_first = True)

data.head()