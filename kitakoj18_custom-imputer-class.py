import pandas as pd

from customimputerclass import CustomImputer
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_missing = train_df.isnull().sum()

train_miss_perct = train_missing/len(train_df)

train_miss_perct[train_miss_perct > 0]
test_missing = test_df.isnull().sum()

test_miss_perct = test_missing/len(test_df)

test_miss_perct[test_miss_perct > 0]
df1 = train_df[['LotFrontage', 'Alley', 'Fence', 'MSZoning', 'MSSubClass', 'Neighborhood']]

df2 = test_df[['LotFrontage', 'Alley', 'Fence', 'MSZoning', 'MSSubClass', 'Neighborhood']]
fill_vals = {'LotFrontage': 'mean', 'Alley': 'most_frequent', 'Fence': 'None', 'MSZoning': ('MSSubClass', 'most_frequent')}
print(df1['LotFrontage'].mean())

print(df1['Alley'].mode())

df1[df1['LotFrontage'].isnull()]
imputer = CustomImputer(fill_vals)

X = imputer.fit_transform(df1)
X.iloc[[7, 12, 1446], :]
df2[df2['MSZoning'].isnull()]
df2.groupby(['MSSubClass', 'MSZoning'])['Neighborhood'].count()
X2 = imputer.transform(df2)
X2.iloc[[455, 756, 790, 1444], :]