import numpy as np 

import pandas as pd 



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import datasets
boston = datasets.load_boston()

print(boston.DESCR)
X = boston.data

y = boston.target



columns = boston.feature_names
# create the dataframe

boston_df = pd.DataFrame(boston.data)



boston_df.columns = columns



boston_df.head()
# checking the data types

boston_df.dtypes
boston_df.describe()
boston_df.isnull().sum()
fig, (axes) = plt.subplots(ncols=13, figsize=(16,10))

sns.boxplot(data = boston_df[['CRIM']], palette = 'Set2', ax = axes[0]);

sns.boxplot(data = boston_df[['ZN']],   palette = 'Set2', ax = axes[1]);

sns.boxplot(data = boston_df[['INDUS']], palette = 'Set2', ax = axes[2]);

sns.boxplot(data = boston_df[['CHAS']], palette = 'Set2', ax = axes[3]);

sns.boxplot(data = boston_df[['NOX']],   palette='Set2', ax=axes[4]);

sns.boxplot(data = boston_df[['RM']]  ,  palette='Set2', ax=axes[5]);

sns.boxplot(data = boston_df[['AGE']],   palette='Set2', ax=axes[6]);

sns.boxplot(data = boston_df[['DIS']],   palette='Set2', ax=axes[7]);

sns.boxplot(data = boston_df[['RAD']],     palette='Set2', ax=axes[8]);

sns.boxplot(data = boston_df[['TAX']],     palette='Set2', ax=axes[9]);

sns.boxplot(data = boston_df[['PTRATIO']], palette='Set2', ax=axes[10]);

sns.boxplot(data = boston_df[['B']],     palette='Set2', ax=axes[11]);

sns.boxplot(data = boston_df[['LSTAT']],   palette='Set2', ax=axes[12]);
from scipy import stats

z = np.abs(stats.zscore(boston_df))

type(z)
boston_ndf = pd.DataFrame(data = z, columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',

       'PTRATIO', 'B', 'LSTAT'])

boston_ndf.head()
boston_ndf.describe()
fig, (axes) = plt.subplots(ncols=13, figsize=(16,10))

sns.boxplot(data = boston_ndf[['CRIM']], palette = 'Set2', ax = axes[0]);

sns.boxplot(data = boston_ndf[['ZN']],   palette = 'Set2', ax = axes[1]);

sns.boxplot(data = boston_ndf[['INDUS']], palette = 'Set2', ax = axes[2]);

sns.boxplot(data = boston_ndf[['CHAS']], palette = 'Set2', ax = axes[3]);

sns.boxplot(data = boston_ndf[['NOX']],   palette='Set2', ax=axes[4]);

sns.boxplot(data = boston_ndf[['RM']]  ,  palette='Set2', ax=axes[5]);

sns.boxplot(data = boston_ndf[['AGE']],   palette='Set2', ax=axes[6]);

sns.boxplot(data = boston_ndf[['DIS']],   palette='Set2', ax=axes[7]);

sns.boxplot(data = boston_ndf[['RAD']],     palette='Set2', ax=axes[8]);

sns.boxplot(data = boston_ndf[['TAX']],     palette='Set2', ax=axes[9]);

sns.boxplot(data = boston_ndf[['PTRATIO']], palette='Set2', ax=axes[10]);

sns.boxplot(data = boston_ndf[['B']],     palette='Set2', ax=axes[11]);

sns.boxplot(data = boston_ndf[['LSTAT']],   palette='Set2', ax=axes[12]);
boston_ndf1 = boston_ndf[boston_ndf>3].dropna(how='all').fillna(0.0)

boston_ndf1.shape
boston_ndf2 = boston_ndf1.T

boston_ndf3 = boston_ndf2[boston_ndf2>0].dropna(how ='all').T

boston_ndf3.shape
boston_ndf3.hist(figsize=(18,12), bins = 20);
boston_ndf3.notnull().sum()
boston_ndf3[boston_ndf3.notnull()].describe()