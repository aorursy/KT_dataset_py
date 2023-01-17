import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



pd.pandas.set_option('display.max_columns', None)



train_data = pd.read_csv('../input/big-mart-sales-prediction/Train.csv')

train_data.head()
train_data.describe()
train_data.hist(figsize=(15,12))
train_data.info()
# Correlation between target and feature variables

corr_matrix = train_data.corr()

corr_matrix['Item_Outlet_Sales']
train_data.Item_Identifier.value_counts()
train_data.Item_Fat_Content.value_counts()
train_data.Item_Fat_Content = train_data.Item_Fat_Content.replace('LF', 'Low Fat')

train_data.Item_Fat_Content = train_data.Item_Fat_Content.replace('low fat', 'Low Fat')

train_data.Item_Fat_Content = train_data.Item_Fat_Content.replace('reg','Regular')



train_data.Item_Fat_Content.value_counts()
# Convert object column types to category type



train_data.Item_Identifier = train_data.Item_Identifier.astype('category')

train_data.Item_Fat_Content = train_data.Item_Fat_Content.astype('category')

train_data.Item_Type = train_data.Item_Type.astype('category')

train_data.Outlet_Identifier = train_data.Outlet_Identifier.astype('category')

train_data.Outlet_Size = train_data.Outlet_Size.astype('category')

train_data.Outlet_Location_Type = train_data.Outlet_Location_Type.astype('category')

train_data.Outlet_Type = train_data.Outlet_Type.astype('category')



train_data.info()
# Correlation strength of column Item_MRP with column Item_Outlet_Sales is very high

# Exploit Item_MRP column for further information about target column



fig, axes = plt.subplots(1, 1, figsize=(12,8))

sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', hue='Item_Fat_Content', size='Item_Weight', data=train_data)
fig, axes = plt.subplots(1, 1, figsize=(10,8))

sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', hue='Item_Fat_Content', size='Item_Weight', data=train_data)

plt.plot([69, 69],[0, 5000])

plt.plot([137, 137],[0, 5000])

plt.plot([203, 203],[0, 9000])
train_data.Item_MRP = pd.cut(train_data.Item_MRP, bins=[25, 69, 137, 203, 270], labels=['a', 'b', 'c', 'd'], right=True)

train_data.head()
# Explore other columns



fig, axes = plt.subplots(3, 1, figsize=(15, 12))

sns.scatterplot(x='Item_Visibility', y='Item_Outlet_Sales', hue='Item_MRP', ax=axes[0], data=train_data)

sns.boxplot(x='Item_Type', y='Item_Outlet_Sales', ax=axes[1], data=train_data)

sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', ax=axes[2], data=train_data)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.boxplot(x='Outlet_Establishment_Year', y='Item_Outlet_Sales', ax=axes[0,0], data=train_data)

sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', ax=axes[0,1], data=train_data)

sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', ax=axes[1,0], data=train_data)

sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', ax=axes[1,1], data=train_data)
# Columns for model training



attributes = ['Item_MRP','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year','Outlet_Identifier',

              'Item_Type','Item_Outlet_Sales']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.boxplot(x='Outlet_Establishment_Year', y='Item_Outlet_Sales', hue='Outlet_Size', ax=axes[0,0], data=train_data)

sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', hue='Outlet_Size', ax=axes[0,1], data=train_data)

sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', hue='Outlet_Size', ax=axes[1,0], data=train_data)

sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', hue='Outlet_Size', ax=axes[1,1], data=train_data)
data = train_data[attributes]

data.info()
sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', hue='Outlet_Type', data=data)
data[data.Outlet_Size.isnull()]
data.groupby(['Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type'])['Outlet_Identifier'].value_counts()
data.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].value_counts()
data.groupby('Outlet_Size').Outlet_Identifier.value_counts()
data['Outlet_Size'].isnull().value_counts()
# data.apply(lambda x: len(x.unique()))



data['Outlet_Size'].fillna((data['Outlet_Size'].mode()[0]), inplace=True)
data.head()
sns.boxplot(x='Item_MRP', y='Item_Outlet_Sales', data=data)
data[data.Item_MRP=='b'].Item_Outlet_Sales.describe()
data[data.Item_Outlet_Sales==7158.6816]
data.iloc[7796,7:] = data.groupby('Item_MRP').get_group('b')['Item_Outlet_Sales'].median()

data.iloc[7796,7:]
# Outliers fixed 

sns.boxplot(x='Item_MRP', y='Item_Outlet_Sales', data=data)
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=data)
sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=data)
data[data.Outlet_Location_Type == 'Tier 1'].Item_Outlet_Sales.describe()
data[data.Item_Outlet_Sales==9779.936200]
data.iloc[4289,7:] = data.groupby('Outlet_Location_Type').get_group('Tier 1')['Item_Outlet_Sales'].median()

data.iloc[4289,7:]
sns.boxplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', data=data)
data.head()
sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=data)
data[data.Outlet_Size=='High'].Item_Outlet_Sales.describe()
data[data.Item_Outlet_Sales==10256.649000]
data.iloc[4888,7:] = data.groupby('Outlet_Size').get_group('High')['Item_Outlet_Sales'].median()

data.iloc[4888,7:]
sns.boxplot(x='Outlet_Size', y='Item_Outlet_Sales', data=data)
sns.boxplot(x='Outlet_Establishment_Year', y='Item_Outlet_Sales', data=data)
data.Outlet_Establishment_Year = data.Outlet_Establishment_Year.astype('category')
data_label = data.Item_Outlet_Sales # y

data_dummy = pd.get_dummies(data.iloc[:,0:6]) # X

data_dummy['Item_Outlet_Sales'] = data_label

print(data_dummy.shape) # X

data_dummy.head()
from sklearn.model_selection import train_test_split



train, test = train_test_split(data_dummy, test_size=0.20, random_state=2019)

train.shape, test.shape
train_label = train['Item_Outlet_Sales']

test_label = test['Item_Outlet_Sales']



del train['Item_Outlet_Sales']

del test['Item_Outlet_Sales']
from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()

linear_reg.fit(train, train_label)

from sklearn.metrics import mean_squared_error



predict_lr = linear_reg.predict(test)

mse = mean_squared_error(test_label, predict_lr)

lr_score = np.sqrt(mse)

lr_score
# Cross validation for linear regression



from sklearn.model_selection import cross_val_score



score = cross_val_score(linear_reg, train, train_label, cv=10, scoring='neg_mean_squared_error')

lr_score_cross = np.sqrt(-score)



np.mean(lr_score_cross), np.std(lr_score_cross)
from sklearn.linear_model import Ridge



r = Ridge(alpha=0.05, solver='cholesky')

r.fit(train, train_label)



predict_r = r.predict(test)

mse = mean_squared_error(test_label, predict_r)

r_score = np.sqrt(mse)

r_score
# Cross validation Ridge

r = Ridge(alpha=0.05, solver='cholesky')

score = cross_val_score(r, train, train_label, cv=10, scoring='neg_mean_squared_error')

r_score_cross = np.sqrt(-score)

np.mean(r_score_cross), np.std(r_score_cross)
from sklearn.linear_model import Lasso



l = Lasso(alpha=0.01)

l.fit(train, train_label)



predict_l = l.predict(test)

mse = mean_squared_error(test_label, predict_l)

l_score = np.sqrt(mse)

l_score
# Cross validation Lasso



l = Lasso(alpha=0.01)

score = cross_val_score(l, train, train_label, cv=10, scoring='neg_mean_squared_error')

l_score_cross = np.sqrt(-score)

np.mean(l_score_cross), np.std(l_score_cross)
from sklearn.tree import DecisionTreeRegressor



dtr = DecisionTreeRegressor()

dtr.fit(train, train_label)



predict_r = dtr.predict(test)

mse = mean_squared_error(test_label, predict_r)

dtr_score = np.sqrt(mse)

dtr_score
# Cross validation Decision Tree



dtr = DecisionTreeRegressor()

score = cross_val_score(dtr, train, train_label, cv=10, scoring='neg_mean_squared_error')

dtr_score_cross = np.sqrt(-score)

np.mean(dtr_score_cross), np.std(dtr_score_cross)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor()

rf.fit(train, train_label)

predict_rf = rf.predict(test)

mse = mean_squared_error(test_label, predict_rf)

rf_score = np.sqrt(mse)

rf_score
# Cross validation Random Forest



rf = RandomForestRegressor()

score = cross_val_score(rf, train, train_label, cv=10, scoring='neg_mean_squared_error')

rf_score_cross = np.sqrt(-score)

np.mean(rf_score_cross), np.std(rf_score_cross)
from sklearn.ensemble import BaggingRegressor



br = BaggingRegressor(max_samples=70)

br.fit(train, train_label)
predict_br = br.predict(test)

br_score = mean_squared_error(test_label, predict_br)

br_score = np.sqrt(br_score)

br_score
# Cross validation Bagging



br = BaggingRegressor()

score = cross_val_score(br, train, train_label, cv=10, scoring='neg_mean_squared_error')

br_score_cross = np.sqrt(-score)

np.mean(br_score_cross), np.std(br_score_cross)
from sklearn.ensemble import AdaBoostRegressor



ada = AdaBoostRegressor()

ada.fit(train, train_label)

predict_ada = ada.predict(test)

ada_score = mean_squared_error(test_label, predict_ada)

ada_score = np.sqrt(ada_score)

ada_score
# Cross validation AdaBoostRegression



ada = AdaBoostRegressor()

score = cross_val_score(ada, train, train_label, cv=10, scoring='neg_mean_squared_error')

ada_score_cross = np.sqrt(-score)

np.mean(ada_score_cross), np.std(ada_score_cross)
from sklearn.ensemble import GradientBoostingRegressor



gbr = GradientBoostingRegressor()

gbr.fit(train, train_label)

predict_gbr = gbr.predict(test)

gb_score = mean_squared_error(test_label, predict_gbr)

gb_score = np.sqrt(gb_score)

gb_score
# Cross validation Gradient Boosting



gb = GradientBoostingRegressor()

score = cross_val_score(gb, train, train_label, cv=10, scoring='neg_mean_squared_error')

gb_score_cross = np.sqrt(-score)

np.mean(gb_score_cross), np.std(gb_score_cross)
techniques = ['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',

     'Lasso Regression CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost',

              'Ada Boost CV','Bagging','Bagging CV','Gradient Boost','Gradient Boost CV']



score_df = pd.DataFrame({'model': [lr_score,lr_score_cross,r_score,r_score_cross,l_score,l_score_cross,dtr_score,dtr_score_cross,

                                   rf_score,rf_score_cross,ada_score,ada_score_cross,br_score,br_score_cross, 

                                   gb_score,gb_score_cross]}, index=techniques)



score_df['model'] = score_df.applymap(lambda x: x.mean())

score_df.model.sort_values()
from sklearn.model_selection import GridSearchCV



gb = GradientBoostingRegressor(max_depth=7, n_estimators=200, learning_rate=0.01)



param = [{'min_samples_split':[5,9,13], 'max_leaf_nodes':[3,5,7,9], 'max_features':[8,10,15,18]}]



gs = GridSearchCV(gb, param, cv=5, scoring='neg_mean_squared_error')

gs.fit(train, train_label)



gs.best_estimator_
gb = gs.best_estimator_
total = pd.concat([train,test], axis=0, ignore_index=True)



total_label = pd.concat([train_label, test_label], axis=0, ignore_index=True)



total_label.shape, total.shape
gb.fit(total, total_label)
test = pd.read_csv('../input/big-mart-sales-prediction/Test.csv')

test.shape
# Test data Columns for model training



attributes = ['Item_MRP','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year','Outlet_Identifier',

              'Item_Type']



test = test[attributes]

test.shape
test.info()
test.Item_MRP = pd.cut(test.Item_MRP, bins=[25,75,140,205,270], labels=['a','b','c','d'],right=True)



test.Item_Type = test.Item_Type.astype('category')



test.Outlet_Size = test.Outlet_Size.astype('category')



test.Outlet_Identifier = test.Outlet_Identifier.astype('category')



test.Outlet_Type = test.Outlet_Type.astype('category')



test.Outlet_Location_Type = test.Outlet_Location_Type.astype('category')



test.Outlet_Establishment_Year = test.Outlet_Establishment_Year.astype('category')



test.info()
test.Outlet_Size.isnull().value_counts()
test['Outlet_Size'].fillna((test['Outlet_Size'].mode()[0]), inplace=True)
test.Outlet_Size.isnull().value_counts()
test_dummy = pd.get_dummies(test.iloc[:,0:6])



test_dummy.head()
predict = gb.predict(test_dummy)



predict.shape
sample = pd.read_csv('../input/big-mart-sales-prediction/Submission.csv')



sample.head()
del sample['Item_Outlet_Sales']
predict_df = pd.DataFrame({'Item_Outlet_Sales': predict})

pred_values = pd.concat([sample, predict_df], axis=1)

pred_values.head()
pred_values.to_csv('../sales-prediction-submission.csv')
output_df = pd.read_csv('../sales-prediction-submission.csv')

output_df.head()