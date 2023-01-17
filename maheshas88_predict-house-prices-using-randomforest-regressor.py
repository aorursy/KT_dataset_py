import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings("ignore")

#set random seed for the whole session
random.seed(123)

#import the train dataset
house_train=pd.read_csv("../input/train.csv")
house_train.head()
#test data
house_test = pd.read_csv("../input/test.csv")
house_test.head()
#the dataset contains several numeric features as well as categorical.
house_train.describe()
house_train.dtypes
nums = house_train.select_dtypes(include=['float64','int64']).columns
print(nums)
fig, ax = plt.subplots()
ax.scatter(x = house_train['LotFrontage'], y = house_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('LotFrontage', fontsize=11)
plt.show()

house_train = house_train.drop(house_train[(house_train['LotFrontage']>300) & (house_train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(x = house_train['LotArea'], y = house_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('LotArea', fontsize=11)
plt.show()
house_train = house_train.drop(house_train[(house_train['LotArea']>200000) & 
                                           (house_train['SalePrice']<=400000)].index)
fig, ax = plt.subplots()
ax.scatter(x = house_train['GrLivArea'], y = house_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('GrLivArea', fontsize=11)
plt.show()
house_train = house_train.drop(house_train[(house_train['GrLivArea']>4000) & 
                                           (house_train['SalePrice']<=200000)].index)
fig, ax = plt.subplots()
ax.scatter(x = house_train['OverallQual'], y = house_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('OverallQual', fontsize=11)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = house_train['GarageArea'], y = house_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('GarageArea', fontsize=11)
plt.show()
house_train = house_train.drop(house_train[(house_train['GarageArea']>1300) & 
                                           (house_train['SalePrice']<=200000)].index)
#take dependent variable in aseprate list and drop it from train dataframe
train_y = house_train['SalePrice']
house_train.drop(columns='SalePrice', inplace=True)

#concatenate train and test data to perform common operations on both
entire_data = pd.concat([house_train, house_test])

##column Id is not needed
entire_data.drop(columns='Id')
print(entire_data.shape)
#list the null values and their count in each col
null_cols = entire_data.columns[entire_data.isnull().any()]
entire_data[null_cols].isnull().sum()
#take the columns that are of type object and replace null values with "Negative". "Negative"
#could mean missing or not available according to each column. We cant afford to drop rows with
#null vals as the training dataset is not too much already.
objects = entire_data.select_dtypes(include=['object']).columns
for col in objects:
    entire_data[col] = entire_data[col].fillna("Negative")

#replace numerical cols missing vals with the mean of the column
nums = entire_data.select_dtypes(include=['float64','int64']).columns
for col in nums:
    entire_data[col] = entire_data[col].fillna(entire_data[col].mean()) 
null_cols = entire_data.columns[entire_data.isnull().any()]
entire_data[null_cols].isnull().sum()
entire_data = pd.get_dummies(entire_data)
#the dependent variable saleprice is right skewed with a peak. It makes sense to unskew it. 
#Log transfroming the column values gives it a more normal distribution. Other transformations like 
#BoxCox can also be used here.

#Below graph shows the actual skew before tansformation of saleprice
sns.distplot(train_y, color="c", kde=False)
plt.title("Skewness of Sale Price")
plt.ylabel("Total Number")
plt.xlabel("Sale Price")
#logtransofrm 
train_y = np.log1p(train_y)

#skewness after transformation for saleprice
sns.distplot(train_y, color="c", kde=False)
plt.title("Skewness of Sale Price")
plt.ylabel("Total Number")
plt.xlabel("Sale Price")
from scipy.stats import skew

#in the dataset several features are skewed. This not augur well for a model to predict the right values.
#Here the features which are numeric alone are taken and its skew measured. If skew is more than a threshold
#it is logtransformed. 

numerics = entire_data.dtypes[entire_data.dtypes != "object"].index
skewed_feats = entire_data[numerics].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.70]
skewed_feats = skewed_feats.index
entire_data[skewed_feats] = np.log1p(entire_data[skewed_feats])
#train test split
train_x = entire_data[:house_train.shape[0]]
test_x = entire_data[house_train.shape[0]:]
print(len(train_x))
print(len(test_x))
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#Model
model = RandomForestRegressor(random_state=123, n_estimators=300, verbose=1, 
                              n_jobs=-1, oob_score=True)

param_grid = {
    'bootstrap': [True],
    'max_features': ['auto','sqrt','log2'],
    'n_estimators': [300,400,500,600,700],
    'random_state': [0,1,42,123]
}

grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(train_x, train_y)

best_rf = grid_search.best_estimator_
best_rf.feature_importances_
#predict using the model
y_pred = best_rf.predict(test_x)
#submission
sub_file = pd.read_csv('../input/sample_submission.csv',sep=',')
sub_file.SalePrice=y_pred
sub_file.to_csv('Submission1.csv',index=False)