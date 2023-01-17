import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import xgboost as xgb

from sklearn import ensemble

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.options.display.max_columns = 999
import os

os.listdir("../input/mercedes-benz-greener-manufacturing")
train_df = pd.read_csv("../input/mercedes-benz-greener-manufacturing/train.csv.zip")

test_df = pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv.zip")
print(f'Train Shape: {train_df.shape}')

print(f'Test Shape: {test_df.shape}')
train_df.head()
plt.figure(figsize=(8, 6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
upper_limit = 180

train_df.loc[train_df['y'] > upper_limit, 'y'] = upper_limit



plt.figure(figsize=(8, 6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(train_df.y.values, bins=50, kde=False)

plt.xlabel('y value', fontsize=12)

plt.show()
train_df.dtypes.reset_index()
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Column", "Column Type"]

dtype_df['Column Type'].value_counts()
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type")["Count"].count().reset_index()
dtype_df.loc[:10, :]
train_df.isnull().sum(axis=0).reset_index()
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count'] > 0]

missing_df = missing_df.sort_values(by="missing_count")

missing_df
unique_values_dict = {}



for col in train_df.columns:

    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        unique_value = str(np.sort(train_df[col].unique()).tolist())

        if unique_value not in unique_values_dict:

            unique_values_dict[unique_value] = [col]

        else:

            unique_values_dict[unique_value].append(col)



for unique_val in unique_values_dict:

    print("Columns containing the unique values : ",unique_val)

    print(unique_values_dict[unique_val])

    print("--------------------------------------------------")
train_df['X1'].value_counts()
var_name = "X1"



plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train_df, order= train_df[var_name].value_counts().index)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X2"



plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train_df, order= train_df[var_name].value_counts().index)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X3"



plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train_df, order= train_df[var_name].value_counts().index)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
one_count_list = []

zero_count_list = []



cols_list = unique_values_dict['[0, 1]']



# Now to store total no. of 0's & 1's in each col

for col in cols_list:

    zero_count_list.append((train_df[col] == 0).sum())

    one_count_list.append((train_df[col] == 1).sum())



N = len(cols_list)

ind = np.arange(N)

width = 0.35



plt.figure(figsize=(6,100))

p1 = plt.barh(ind, zero_count_list, width, color='red')

p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color="green")

plt.yticks(ind, cols_list)

plt.legend((p1, p2), ('Zero count', 'One Count'))

plt.title("Count Distribution", fontsize=15)

plt.show()
zero_mean_list = []

one_mean_list = []



cols_list = unique_values_dict['[0, 1]']



for col in cols_list:

    zero_mean_list.append(train_df.loc[train_df[col] == 0, 'y'].mean())

    one_mean_list.append(train_df.loc[train_df[col] == 1, 'y'].mean())



temp_df = pd.DataFrame({"column_name": cols_list + cols_list, "value": [0]*len(cols_list) + [1]*len(cols_list), "y_mean": zero_mean_list + one_mean_list})

temp_df = temp_df.pivot(index = 'column_name', columns = 'value', values = 'y_mean')

temp_df.head()
plt.figure(figsize=(8, 80))

sns.heatmap(temp_df, cmap="YlGnBu")

plt.title("Mean of y val across binary variables", fontsize=15)

plt.show()
var_name = "ID"



plt.figure(figsize=(12,6))

sns.regplot(x = var_name, y = 'y', data = train_df, scatter_kws = {'alpha':0.5, 's':30})

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
train_df['eval_set'] = "train"

test_df['eval_set'] = "test"

train_df['eval_set'].head()
full_df = pd.concat([train_df[["ID", "eval_set"]], test_df[["ID", "eval_set"]]])

full_df.head()
plt.figure(figsize=(12,6))

sns.stripplot(x="eval_set", y='ID', data=full_df)

plt.xlabel("eval_set", fontsize=12)

plt.ylabel('ID', fontsize=12)

plt.title("Distribution of ID variable with evaluation set", fontsize=15)

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x="eval_set", y='ID', data=full_df)

plt.xlabel("eval_set", fontsize=12)

plt.ylabel('ID', fontsize=12)

plt.title("Distribution of ID variable with evaluation set", fontsize=15)

plt.show()
for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))

        

train_y = train_df['y'].values

train_X = train_df.drop(["ID", "y", "eval_set"], axis=1)



def xgb_r2_score(preds, dtrain):

    labels = dtrain.getLabel()

    return 'r2', r2_score(labels, preds)



xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'silent': 1

}



dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)
fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

model.fit(train_X, train_y)
feat_names = train_X.columns.values



importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()