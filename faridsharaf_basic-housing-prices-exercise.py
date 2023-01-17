import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import mean_absolute_error as MAE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline

import matplotlib.pyplot as plt # Data visualization



import seaborn as sns
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')



print('Data shape  (row, col)')

print('Train Data:', train.shape)

print('Test Data: ', test.shape)
train.head()
target = train.SalePrice
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.info()
numeric_features.describe()
corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending=False)[:10], '\n')

print(corr['SalePrice'].sort_values(ascending=False)[:-5])

# should this be used after handling the skewness ?
train.OverallQual.unique()
# Relation between qualty and sale price median

quality_pivot = train.pivot_table(index = 'OverallQual', values = "SalePrice", aggfunc = np.median)
quality_pivot.plot(kind = 'bar', color = '#f44336')

plt.xlabel('Overall Quality')

plt.ylabel("Sale price")
plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel("Sale Price")

plt.xlabel("Ground Living Area")

# plt.figure(figsize=(16, 8))
# Mapping price vs Lotfrontage

sns.scatterplot(x='LotFrontage', y='SalePrice', data=train);

plt.title('House price versus lot frontage size');



# plt.figure(figsize=(16, 8))

# plt.subplot(1, 2, 1)

# plt.scatter(train['LotFrontage'], train['SalePrice'])

# plt.title('House price versus lot frontage size')
categ_features = train.select_dtypes(exclude = [np.number])

categ_features.describe()
print(train.SaleCondition.unique())
# Pivot table for the sale condition values occurance

sale_condition_pivot =  train.pivot_table(index = 'SaleCondition', values = 'SalePrice', aggfunc = np.median)

sale_condition_pivot.plot(kind = 'bar', color = '#f44336')

plt.xlabel('Sale Condition')

plt.ylabel("Sale price")
def encode(x):

    return 1 if x == 'Partial' else 0

train['enc_saleCondition'] = train.SaleCondition.apply(encode)

test['enc_saleCondition'] = test.SaleCondition.apply(encode)
# Pivot table for the new encoded sale condition values occurance

sale_condition_pivot =  train.pivot_table(index = 'enc_saleCondition', values = 'SalePrice', aggfunc = np.median)

sale_condition_pivot.plot(kind = 'bar', color = '#f44336')

plt.xlabel('Sale Condition')

plt.ylabel("Sale price")
# Visualizing missing values in train and test sets

with sns.axes_style(style= "whitegrid"):    

    for dataset in ["train", "test"]:

        fig, ax = plt.subplots(1, 1, figsize=(18, 10))

        bar = eval(dataset).isna().sum().plot(kind = "bar")

        bar.set_title(f"No. of missing values in {dataset}", fontsize = 17)
# Drop columns with many missing values and 

train.drop(['Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)

test.drop(['Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)
# look for columns with null values

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

sum(data.isnull().sum() != 0)
# Impute missing LotFrontage values with Median

train['LotFrontage'] = train['LotFrontage'].replace(np.NaN, train['LotFrontage'].median()) # Mean will be affected by scattered outliers

test['LotFrontage'] = test['LotFrontage'].replace(np.NaN, test['LotFrontage'].median())



train['GarageCars'] = train['GarageCars'].replace(np.NaN, train['GarageCars'].median())

test['GarageArea'] = test['GarageArea'].replace(np.NaN, test['GarageArea'].median())
y = train.SalePrice

X = data.drop(['SalePrice', 'Id'], axis=1)



X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=.2)
#test model on validation set

forest_model = RandomForestRegressor(random_state = 42, max_depth=10, n_estimators = 1000)

forest_model.fit(X_train,y_train)



prediction = forest_model.predict(X_val)



print('Mean Absolute Error: ', MAE(y_val, prediction))

print('Model Score: ', forest_model.score(X_val, y_val))
submission = pd.DataFrame()

submission['Id'] = test.Id



feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()



prediction = forest_model.predict(feats)

submission['SalePrice'] = prediction

submission.to_csv('submission1.csv', index=False)