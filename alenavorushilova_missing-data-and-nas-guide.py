import numpy as np

import pandas as pd 

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df = pd.read_csv('../input/bank-loan2/madfhantr.csv')

df.head()
df.info()
df.describe().T
def summary(df):

    

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nas = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing = (df.isnull().sum() / df.shape[0]) * 100

    sk = df.skew()

    krt = df.kurt()

    

    print('Data shape:', df.shape)



    cols = ['Type', 'Total count', 'Null Values', 'Distinct Values', 'Missing Ratio', 'Unique Values', 'Skewness', 'Kurtosis']

    dtls = pd.concat([types, counts, nas, distincts, missing, uniques, sk, krt], axis=1, sort=False)

  

    dtls.columns = cols

    return dtls
details = summary(df)

details
df.isnull().sum()
cols = df.columns 

colours = ['g', 'r'] 

f, ax = plt.subplots(figsize = (12,8))

sns.set_style("whitegrid")

plt.title('Missing Values Heatmap', )

sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours));
import missingno as msno

msno.matrix(df);
msno.bar(df,  color = 'y', figsize = (14,10));
msno.bar(df, log = True, color = 'g');
msno.heatmap(df,  cmap='GnBu_r');
ax = msno.dendrogram(df)
for col in df.columns:

    prct = np.mean(df[col].isnull())

    print('{}:{}%'.format(col, round(prct*100)))
df.dropna(subset = ['Loan_Amount_Term'], axis = 0, how = 'any', inplace = True)

df.isnull().sum()
df['Gender'].fillna((df['Gender'].mode()[0]),inplace=True)

df['Married'].fillna(df['Married'].mode()[0], inplace = True)

df['Dependents'].fillna((df['Dependents'].mode()[0]),inplace=True)

df.isnull().sum()
df['Credit_History'].fillna(method = 'ffill', inplace = True)

df['Self_Employed'].fillna(method = 'bfill', inplace = True)

df.isnull().sum()
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)

df.isnull().sum()