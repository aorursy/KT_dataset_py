import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#To hide Warning messages.

import warnings

warnings.filterwarnings('ignore')
traindf = pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')

testdf=pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')

traindf.head()
testdf.head()
traindf.describe()
traindf.shape , testdf.shape
sns.distplot(traindf.skew(),color='Green',axlabel ='Skewness')

plt.show()
sns.set_style('whitegrid')

sns.distplot(traindf['SalePrice'],bins=30,kde=True,color='Green')
traindf.info()
sum(traindf['Id'].duplicated()),sum(testdf['Id'].duplicated())
traindf.drop(columns=['Id'],inplace=True)
num_cols=[var for var in traindf.columns if traindf[var].dtypes != 'O']

cat_cols=[var for var in traindf.columns if traindf[var].dtypes != 'int64' and traindf[var].dtypes != 'float64']



print('No of Numerical Columns: ',len(num_cols))

print('No of Categorical Columns: ',len(cat_cols))

print('Total No of Cols: ',len(num_cols+cat_cols))
plt.figure(figsize=(30,12))

sns.heatmap(traindf.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')

plt.show()
var_with_na=[var for var in traindf.columns if traindf[var].isnull().sum()>=1 ]

for var in var_with_na:

    print(var, np.round(traindf[var].isnull().mean(),3), '% missing values')
var_with_na2=[var for var in testdf.columns if testdf[var].isnull().sum()>=1 ]

for var in var_with_na2:

    print(var, np.round(testdf[var].isnull().mean(),3), '% missing values')