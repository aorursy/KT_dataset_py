import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os,gc

import warnings 

warnings.filterwarnings("ignore")



from math import sqrt



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



import eli5

from eli5.sklearn import PermutationImportance
train = pd.read_csv('../input/restaurant-revenue-prediction/train.csv')

test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv')

sample = pd.read_csv('../input/restaurant-revenue-prediction/sampleSubmission.csv')
print(train.shape, test.shape)
train.head(n=10)
train.info()
train.describe()
plt.subplots(figsize=(6,6))

sns.distplot(train['revenue'], kde=True, bins=20)

plt.title('Number of Restaurants vs Revenue')

plt.xlabel('Revenue')

plt.ylabel('Number of Restaurants')
train['City'].nunique()
plt.subplots(figsize=(8,4))

train['City'].value_counts().plot(kind='bar')

plt.title('No of restaurants vs City')

plt.xlabel('City')

plt.ylabel('No of restaurants')
train[['City','revenue']].groupby('City').mean().plot(kind='bar')

plt.title('Mean Revenue Generated vs City')

plt.xlabel('City')

plt.ylabel('Mean Revenue Generated')
mean_revenue_per_city = train[['City', 'revenue']].groupby('City', as_index=False).mean()

mean_revenue_per_city['revenue'] = mean_revenue_per_city['revenue'].apply(lambda x: int(x/1e6)) 

mean_revenue_per_city



mean_dict = dict(zip(mean_revenue_per_city.City, mean_revenue_per_city.revenue))

mean_dict
train.replace({"City":mean_dict}, inplace=True)

test.replace({"City":mean_dict}, inplace=True)
test['City'] = test['City'].apply(lambda x: 6 if isinstance(x,str) else x)
train['City Group'].unique()
sns.countplot(train['City Group'])

plt.ylabel('No. of Restaurants')

plt.title('No of Restaurants vs City Group')
train[['City Group', 'revenue']].groupby('City Group').mean().plot(kind='bar')

plt.ylabel('Mean Revenue Generated')

plt.title('Mean Revenue Generated vs City Group')
lr = LabelEncoder()

train['City Group'] = lr.fit_transform(train['City Group'])

test['City Group'] = lr.transform(test['City Group'])
train['Type'].unique()
sns.countplot(train['Type'])
train[['Type', 'revenue']].groupby('Type').mean().plot(kind='bar')

plt.title('Mean Revenue per Type')
test['Type'] = lr.fit_transform(test['Type'])

train['Type'] = lr.transform(train['Type'])
train.info()
train_correlations = train.drop(["revenue"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_correlations, color="Red", label="train")

sns.distplot(test_correlations, color="Green", label="test")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
plt.figure(figsize=(20,20))

sns.heatmap(train.corr(), annot=True)
X = train.drop(['revenue', 'Id', 'Open Date'],axis=1)

y = train['revenue']
X.head()
model = LinearRegression(normalize=True)

model.fit(X,y)
perm = PermutationImportance(model, random_state=1).fit(X,y)

eli5.show_weights(perm, feature_names = X.columns.to_list())
important_features = ['P26', 'P9', 'P16', 'P36', 'P8', 'P18']



f, axes = plt.subplots(3,2, figsize=(12,12), sharex=True)

f.suptitle('Distribution Plots of Important Features')



for ax,feature in zip(axes.flatten(), important_features):

    sns.distplot(X[feature], ax=ax)
sns.pairplot(train[important_features])
important_features
train['P26_to_City_mean'] = train.groupby('City')['P26'].transform('mean')

train['P9_to_City_mean'] = train.groupby('City')['P9'].transform('mean')

train['P16_to_City_mean'] = train.groupby('City')['P16'].transform('mean')

train['P36_to_City_mean'] = train.groupby('City')['P36'].transform('mean')

train['P8_to_City_mean'] = train.groupby('City')['P8'].transform('mean')

train['P18_to_City_mean'] = train.groupby('City')['P18'].transform('mean')



test['P26_to_City_mean'] = test.groupby('City')['P26'].transform('mean')

test['P9_to_City_mean'] = test.groupby('City')['P9'].transform('mean')

test['P16_to_City_mean'] = test.groupby('City')['P16'].transform('mean')

test['P36_to_City_mean'] = test.groupby('City')['P36'].transform('mean')

test['P8_to_City_mean'] = test.groupby('City')['P8'].transform('mean')

test['P18_to_City_mean'] = test.groupby('City')['P18'].transform('mean')
train['P26_to_City_group_mean'] = train.groupby('City Group')['P26'].transform('mean')

train['P9_to_City_group_mean'] = train.groupby('City Group')['P9'].transform('mean')

train['P16_to_City_group_mean'] = train.groupby('City Group')['P16'].transform('mean')

train['P36_to_City_group_mean'] = train.groupby('City Group')['P36'].transform('mean')

train['P8_to_City_group_mean'] = train.groupby('City Group')['P8'].transform('mean')

train['P18_to_City_group_mean'] = train.groupby('City Group')['P18'].transform('mean')



test['P26_to_City_group_mean'] = test.groupby('City Group')['P26'].transform('mean')

test['P9_to_City_group_mean'] = test.groupby('City Group')['P9'].transform('mean')

test['P16_to_City_group_mean'] = test.groupby('City Group')['P16'].transform('mean')

test['P36_to_City_group_mean'] = test.groupby('City Group')['P36'].transform('mean')

test['P8_to_City_group_mean'] = test.groupby('City Group')['P8'].transform('mean')

test['P18_to_City_group_mean'] = test.groupby('City Group')['P18'].transform('mean')
X = train.drop(['revenue', 'Id', 'Open Date'],axis=1)

y = train['revenue']
X.head()
cv = KFold(n_splits=10, shuffle=True, random_state=108)

model = LGBMRegressor(n_estimators=200, learning_rate=0.01, subsample=0.7, colsample_bytree=0.8)



scores = []

for train_idx, test_idx in cv.split(X):

    X_train = X.iloc[train_idx]

    X_val = X.iloc[test_idx]

    y_train = y.iloc[train_idx]

    y_val = y.iloc[test_idx]

    

    model.fit(X_train,y_train)

    preds = model.predict(X_val)

    

    rmse = sqrt(mean_squared_error(y_val, preds))

    print(rmse)

    scores.append(rmse)



print("\nMean score %d"%np.mean(scores))
test.head()
predictions = model.predict(test.drop(['Id', 'Open Date'], axis=1))

sample['Prediction'] = predictions
sns.distplot(predictions, bins=20)
sample.to_csv('submission.csv', index=False)