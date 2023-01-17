# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_data_original = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train_data = train_data_original
train_data.info()
train_data.head()
print(type(train_data))
train_data.describe()
# fig, ax = plt.subplots(figsize=(30,30)) 
# train_data_corr = train_data.corr()
# sns.heatmap(train_data_corr, annot=True, cmap="YlGnBu", ax=ax)
train_data_corr = train_data.corr(method='spearman')
fig, ax = plt.subplots(figsize=(30,30)) 
sns.heatmap(train_data_corr, annot=True, cmap="YlGnBu", ax=ax)
td_best_corr = train_data_corr.SalePrice.sort_values(ascending=False)[train_data_corr.SalePrice > 0.4]
td_best_corr.axes
feature_columns = list(td_best_corr.to_dict().keys())
high_corr_train_data = train_data[feature_columns]
                                  
high_corr_train_data.info()
train_data_corr.LotFrontage.drop(labels=['LotFrontage'], inplace=True)
# train_data_corr.LotFrontage.values.index(min(train_data_corr.LotFrontage.values))
corr_index = train_data_corr.LotFrontage.to_list().index(max(train_data_corr.LotFrontage.to_list()))
LotFrontage_corr = (train_data_corr.LotFrontage.index[corr_index], train_data_corr.LotFrontage[corr_index])
LotFrontage_corr
lot_df = high_corr_train_data[['LotArea', 'LotFrontage']]
lot_df = lot_df[lot_df['LotArea'].notna()]
lot_df
lot_area_frontage_ratio = lot_df.sum().LotArea / lot_df.sum().LotFrontage
lot_area_frontage_ratio
high_corr_train_data['LotFrontage'].fillna(high_corr_train_data['LotArea'] / lot_area_frontage_ratio, inplace=True)
high_corr_train_data.info()
GarageYrBlt_mean = high_corr_train_data['GarageYrBlt'].mean()
MasVnrArea_mean = high_corr_train_data['MasVnrArea'].mean()

high_corr_train_data[['GarageYrBlt']] = high_corr_train_data[['GarageYrBlt']].fillna(value=GarageYrBlt_mean)
high_corr_train_data[['MasVnrArea']] = high_corr_train_data[['MasVnrArea']].fillna(value=MasVnrArea_mean)

high_corr_train_data.info()
y = high_corr_train_data.SalePrice
X = high_corr_train_data.drop(['SalePrice'], axis=1)

X.info()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

X.head()
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def get_best_nodes_candidate(candidate_list, train_X, val_X, train_y, val_y):
    candidate_results = {}
    max_leaf_champion = 0
    
    for max_leaf in candidate_list:
        candidate_results[str(max_leaf)] = get_mae(max_leaf, train_X, val_X, train_y, val_y)
        if max_leaf_champion == 0 or candidate_results[str(max_leaf_champion)] > candidate_results[str(max_leaf)]:
            max_leaf_champion = max_leaf;
        
    return (candidate_results, max_leaf_champion)
        
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_prediction = rf_model.predict(val_X)
rf_model_mae = mean_absolute_error(val_y, rf_prediction)

candidate_results, decision_tree_max_leaf_nodes = get_best_nodes_candidate([80, 100, 120, 130, 140], train_X, val_X, train_y, val_y)
candidate_dataframe = pd.DataFrame(candidate_results.items(), columns=['max_leaf', 'mea'])
print(candidate_dataframe)
print("Best leaf candidate: {}".format(decision_tree_max_leaf_nodes))

dt_model = DecisionTreeRegressor(max_leaf_nodes=decision_tree_max_leaf_nodes, random_state=1)
dt_model.fit(train_X, train_y)
dt_predictions = dt_model.predict(val_X)
dt_model_mae = mean_absolute_error(dt_predictions, val_y)

print("\n")
print("RandomForest MAE: {}".format(rf_model_mae))
print("DecisionTree MAE: {}".format(dt_model_mae))
model_comparison = {}
# rf_prediction
# val_y
model_comparison['Prediction'] = rf_prediction
model_comparison['Actual'] = val_y
model_comparison['Diff'] = []

for i in range(len(rf_prediction)):
    diff = rf_prediction[i] - val_y.index[i];
    model_comparison['Diff'].append(diff)
    
df = pd.DataFrame(model_comparison)
df.head()
test_data.info()
test_feature_columns = feature_columns[1:]
test_feature_columns
high_corr_test_features = test_data[test_feature_columns]

# Fill LotFrontage using lotArea ratio
high_corr_test_features['LotFrontage'].fillna(high_corr_test_features['LotArea'] / lot_area_frontage_ratio, inplace=True)
# Fill the rest
high_corr_test_features.fillna(high_corr_test_features.mean(), inplace=True)
high_corr_test_features.info()
test_preds = rf_model.predict(high_corr_test_features)

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output
output.to_csv('submission_2.csv', index=False)