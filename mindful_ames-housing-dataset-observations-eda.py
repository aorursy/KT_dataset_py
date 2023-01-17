import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline



train = pd.read_csv('../input/train.csv')
plt.hist(train['SalePrice'], bins=30)

plt.xlabel("Sale Price")

plt.ylabel("Number of sales")

plt.show()
plt.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('Sale Price')

plt.xlabel('Above Ground Living Area (SF)')

plt.show()
train['PSF'] = train['SalePrice']/train['GrLivArea']

plt.hist(train['PSF'], bins=20)

plt.ylabel('Number of Sales')

plt.xlabel('Price per square foot')

plt.show()
train.boxplot(column=['PSF'], by=['YrSold'], figsize=(8,5))

plt.suptitle('$ per square foot by Year Sold')

plt.title('')

plt.ylabel('$/square foot')

plt.xlabel('Year Sold')

plt.show()
train['Age'] = train['YrSold'] - train['YearBuilt']

plt.scatter(train['Age'], train['PSF'])

plt.ylabel('$/square foot')

plt.xlabel('Home age (years)')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['Neighborhood'], figsize=(8,5), rot=90)

plt.suptitle('$ per square foot by neighborhood')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Neighborhood')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['BldgType'], figsize=(8,5), rot=90)

plt.suptitle('$ per square foot by bldg type')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Building Type')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['BedroomAbvGr'], figsize=(8,5))

plt.suptitle('$ per square foot by # of bedrooms')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Bedrooms')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['OverallQual'], figsize=(8,5))

plt.suptitle('$ per square foot by quality')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Quality')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['OverallCond'], figsize=(8,5))

plt.suptitle('$ per square foot by overall condition')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Quality')

plt.show()
train['Functional'].value_counts().plot(kind='bar')

plt.ylabel('$/square foot')

plt.xlabel('Functionality rating')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['Functional'], figsize=(8,5), rot=90)

plt.suptitle('')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Functionality rating')

plt.show()
train['SaleCondition'].value_counts().plot(kind='bar')

plt.ylabel('$/square foot')

plt.xlabel('Sale Condition')

plt.show()
train[train["YrSold"]== 2009].boxplot(column=['PSF'], by=['SaleCondition'], figsize=(8,5), rot=90)

plt.suptitle('')

plt.title('2009')

plt.ylabel('$/square foot')

plt.xlabel('Sale condition')

plt.show()