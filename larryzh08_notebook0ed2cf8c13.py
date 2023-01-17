# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

#df.drop('SalePrice', axis = 1, inplace = True)

#test = pd.read_csv('../input/test.csv')

#df = df.append(test, ignore_index = True)

df.head()
df.describe()
df.columns
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline  



print("Some Statistics of the Housing Price:\n")

print(df['SalePrice'].describe())

print("\nThe median of the Housing Price is: ", df['SalePrice'].median(axis = 0))
sns.distplot(df['SalePrice'], kde = False, color = 'b', hist_kws={'alpha': 0.9})
corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
cor_dict = corr['SalePrice'].to_dict()

del cor_dict['SalePrice']

print("List the numerical features decendingly by their correlation with Sale Price:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))
sns.regplot(x = 'OverallQual', y = 'SalePrice', data = df, color = 'Orange')
plt.figure(1)

f, axarr = plt.subplots(3, 2, figsize=(10, 9))

price = df.SalePrice.values

axarr[0, 0].scatter(df.GrLivArea.values, price)

axarr[0, 0].set_title('GrLiveArea')

axarr[0, 1].scatter(df.GarageArea.values, price)

axarr[0, 1].set_title('GarageArea')

axarr[1, 0].scatter(df.TotalBsmtSF.values, price)

axarr[1, 0].set_title('TotalBsmtSF')

axarr[1, 1].scatter(df['1stFlrSF'].values, price)

axarr[1, 1].set_title('1stFlrSF')

axarr[2, 0].scatter(df.TotRmsAbvGrd.values, price)

axarr[2, 0].set_title('TotRmsAbvGrd')

axarr[2, 1].scatter(df.MasVnrArea.values, price)

axarr[2, 1].set_title('MasVnrArea')

f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 12)

plt.tight_layout()

plt.show()
fig = plt.figure(2, figsize=(9, 7))

plt.subplot(211)

plt.scatter(df.YearBuilt.values, price)

plt.title('YearBuilt')



plt.subplot(212)

plt.scatter(df.YearRemodAdd.values, price)

plt.title('YearRemodAdd')



fig.text(-0.01, 0.5, 'Sale Price', va = 'center', rotation = 'vertical', fontsize = 12)



plt.tight_layout()

print(df.select_dtypes(include=['object']).columns.values)
plt.figure(figsize = (12, 6))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)

xt = plt.xticks(rotation=45)
plt.figure(figsize = (12, 6))

sns.countplot(x = 'Neighborhood', data = df)

xt = plt.xticks(rotation=45)