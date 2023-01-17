# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mobile_file_path_train = '../input/mobile-price-classification/train.csv'

mobile_train_data = pd.read_csv(mobile_file_path_train)

mobile_file_path_test = '../input/mobile-price-classification/test.csv'

mobile_test_data = pd.read_csv(mobile_file_path_test)
mobile_train_data.head()
mobile_train_data.describe()
mobile_train_data.columns
#removing missing values, btw we don't have missing values here !

mobile_train_data = mobile_train_data.dropna()
y = mobile_train_data.price_range

y.unique()

sns.countplot(mobile_train_data['price_range'])

plt.show()
mobile_test_data.columns
from sklearn.model_selection import train_test_split



mobile_features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',

       'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',

       'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',

       'touch_screen', 'wifi']

X = mobile_train_data[mobile_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

train_X.describe()
train_X.head()
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

mobile_decision_tree_model = DecisionTreeRegressor(random_state=1)



# Fit model

mobile_decision_tree_model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error



# get predicted prices on validation data

val_predictions = mobile_decision_tree_model.predict(val_X)



val_mae = mean_absolute_error(val_y, val_predictions)

print(mobile_decision_tree_model.tree_.node_count)

print("Validation MAE: {:,.0f}".format(val_mae))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

print(scores)

print(str(best_tree_size) + ' leaves is the best tree size')
candidate_max_leaf_nodes = range(75,350,25)

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

print(scores)

print(str(best_tree_size) + ' leaves is the best tree size with MAE of ' + str(scores[best_tree_size]))
# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)



# fit the final model and uncomment the next two lines

final_model.fit(X, y)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



mobile_forest_model = RandomForestRegressor(random_state=1)

mobile_forest_model.fit(train_X, train_y)

mobile_forest_predicts = mobile_forest_model.predict(val_X)

print(mean_absolute_error(val_y, mobile_forest_predicts))
test_X = mobile_test_data[mobile_features]

predicted_prices = mobile_forest_model.predict(test_X)

my_submission = pd.DataFrame({'Id': mobile_test_data.id, 'price_range': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
X.describe()
fig = plt.figure(figsize=(15,12))

sns.heatmap(mobile_train_data.corr())
sns.boxplot(mobile_train_data['price_range'],mobile_train_data['talk_time'])

sns.countplot(mobile_train_data['dual_sim'])

plt.show()
sns.boxplot(mobile_train_data['dual_sim'],mobile_train_data['price_range'])

sns.boxplot(mobile_train_data['fc'],mobile_train_data['price_range'])

plt.show()
sns.boxplot(mobile_train_data['n_cores'],mobile_train_data['price_range'])

plt.show()

sns.boxplot(mobile_train_data['wifi'],mobile_train_data['price_range'])

plt.show()
sns.boxplot(mobile_train_data['ram'],mobile_train_data['price_range'])

plt.show()
sns.stripplot(data=mobile_train_data, x="fc", y="pc" ,palette="vlag")

plt.show()
sns.stripplot(data=mobile_train_data, x="sc_h", y="sc_w" ,palette="vlag")

plt.show()
# Add screen size field to dataset

mobile_train_data["sc"] = mobile_train_data["sc_h"] * mobile_train_data["sc_w"]

mobile_train_data["camera"] = (mobile_train_data["fc"] + mobile_train_data["pc"]) / 2

mobile_train_data['internet'] = mobile_train_data[['three_g','four_g','wifi']].max(axis=1)

mobile_train_data["ram_g"] = round(mobile_train_data["ram"]/1000) 
mobile_train_data.describe()
mobile_features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',

       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',

       'ram_g', 'sc', 'talk_time',

       'touch_screen', 'internet']

X_new = mobile_train_data[mobile_features]

train_X_new, val_X_new, train_y_new, val_y_new = train_test_split(X_new, y, random_state=1)

train_X_new.describe()

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: get_mae(leaf_size, train_X_new, val_X_new, train_y_new, val_y_new) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

print(scores)

print(str(best_tree_size) + ' leaves is the best tree size with MAE of ' + str(scores[best_tree_size]))
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(train_X,train_y)

lm.score(train_X,train_y)

preds_val = lm.predict(val_X)

mae = mean_absolute_error(val_y, preds_val)

print('Linear Regression MAE is: '+ str(mae))
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



svm = SVC()

svm.fit(train_X,train_y)

preds_val = svm.predict(val_X)

mae = mean_absolute_error(val_y, preds_val)

print('SVM MAE is: '+ str(mae))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(train_X,train_y)

preds_val = knn.predict(val_X)

mae = mean_absolute_error(val_y, preds_val)

print('KNN MAE is: '+ str(mae))
error_rate = {}

for i in range(3,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(train_X,train_y)

    preds_val = knn.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    error_rate[i] = mae

best_n_size = min(error_rate, key=error_rate.get)

print(error_rate)

print(str(best_n_size) + ' neighbors is the best n size with MAE of ' + str(error_rate[best_n_size]))