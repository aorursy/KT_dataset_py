# Remove categorical variables
# Take the log on sales price
# Use decision Tree
# Best depth with CV
# only use columns 
# ['Id','LotArea', 'OverallQual','OverallCond','YearBuilt','TotRmsAbvGrd','GarageCars','WoodDeckSF',
# 'PoolArea','SalePrice']

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
columns_to_use = ['Id', 'LotArea', 'OverallQual','OverallCond','YearBuilt',
                  'TotRmsAbvGrd','GarageCars','WoodDeckSF','PoolArea','SalePrice']
columns_in_test = columns_to_use.copy()
columns_in_test.remove("SalePrice")
columns_in_test
df = pd.read_csv("../input/train.csv", usecols=columns_to_use)
df.set_index('Id', inplace=True)
pd.options.display.max_rows=5
df
df.isna().sum().sum()
y = np.log(df.SalePrice)
X = df.drop(['SalePrice'], 1)
from sklearn import tree
# make an array of depths to choose from, say 1 to 20
depths = np.arange(1, 21)
depths
from sklearn.model_selection import train_test_split
X_train_and_validate, X_test, y_train_and_validate, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 1
)
print(f"X_train_and_validate shape is {X_train_and_validate.shape}")
print(f"X_test shape is {X_test.shape}")
# print(f"X_validate shape is {X_validate.shape}")
print(f"y_train_and_validate shape is {y_train_and_validate.shape}")
# print(f"y_validate shape is {y_validate.shape}")
print(f"y_test shape is {y_test.shape}")
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    ''' Root mean squared error regression loss
    
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    '''
    return np.sqrt(mean_squared_error(y_true, y_pred))
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 10, random_state=1).split(X_train_and_validate, y_train_and_validate)
train_scores = np.zeros((10,20))
validate_scores = np.zeros((10,20))
for k, (train, validate) in enumerate(kfold):
    print(k)
    # Get train and validate
    X_train = X_train_and_validate.iloc[train, :]
    X_validate = X_train_and_validate.iloc[validate, :]
    y_train = y_train_and_validate.iloc[train]
    y_validate = y_train_and_validate.iloc[validate]
    
    # Get best depth in each fold
    for j in depths:
        print(f"max-depth: {j}")
        my_model = tree.DecisionTreeRegressor(max_depth=j)
        my_model.fit(X_train, y_train)
        
        # Train Scores
        y_train_predicted = my_model.predict(X_train)
        train_scores[k, j-1] = root_mean_squared_error(y_train, y_train_predicted)

        # Validate Scores
        y_validate_predicted = my_model.predict(X_validate)
        validate_scores[k, j-1] = root_mean_squared_error(y_validate, y_validate_predicted)
train[0:10]
validate[0:10]
X_train_and_validate
train_scores[0]
validate_scores[0]

validate_scores.shape
results = pd.DataFrame([
    depths, 
    train_scores.mean(axis=0),
    train_scores.std(axis=0),
    validate_scores.mean(axis=0), 
    validate_scores.std(axis=0)
]).transpose()
results.columns = ['depth', 'train_score_mean', 'train_score_std', 'validate_score_mean', 'validate_score_std']
results
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
fig, ax = plt.subplots()
sns.lineplot(x = 'depth', y = 'train_score_mean', label = 'train', data = results)
sns.lineplot(x = 'depth', y = 'validate_score_mean', label = 'validate', data = results)
ax.legend()
fig, ax = plt.subplots()
plt.plot(results.depth, results.train_score_mean, color='blue', marker='o', markersize=5, label='training')
plt.fill_between(results.depth, 
                 results.train_score_mean-results.train_score_std, 
                 results.train_score_mean+results.train_score_std, 
                 alpha=0.35, color='blue')

plt.plot(results.depth, results.validate_score_mean, color='orange', marker='o', markersize=5, label='validate')
plt.fill_between(results.depth, 
                 results.validate_score_mean-results.validate_score_std, 
                 results.validate_score_mean+results.validate_score_std, 
                 alpha=0.35, color='orange')
ax.legend()
ax.set_title("Training and validation scores of CV:10")
results.iloc[results.validate_score_mean.idxmin, :]

my_model = tree.DecisionTreeRegressor(max_depth=5)
my_model.fit(X_train, y_train)
y_predicted = my_model.predict(X_test)
root_mean_squared_error(y_test, y_predicted)

# ここに来て、以下のすべてをtestではなく、submitと呼ぶことにする
submit = pd.read_csv('../input/test.csv', usecols=columns_in_test)
submit.set_index('Id', inplace=True)
# Treat the test data in the same way as training data. In this case, pull same columns.
submit_X = submit.replace(np.nan, 0)
# submit_X = encoder.transform(submit_X)
# test_X = test.replace(np.nan, 0, inplace=True)
# test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(submit_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
# Get the exponent of prices
# The current predicted prices are the log of the prices
predicted_prices = np.exp(predicted_prices)
my_submission = pd.DataFrame({'Id': submit_X.index, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('07_decisiontree_CV_maxdepth5.csv', index=False)
import matplotlib.pyplot as plt
df.SalePrice.hist()
np.log10(df.SalePrice).hist()

