import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
print("Train rows and columns : ", train.shape)

print("Test rows and columns : ", test.shape)
unique_df = train.nunique().reset_index()

unique_df.columns = ["col_name", "unique_count"]

constant_df = unique_df[unique_df["unique_count"]==1]

constant_df.shape
str(constant_df.col_name.tolist())
X = train.drop(constant_df.col_name.tolist() + ["ID", "target"], axis=1)

y = np.log1p(train["target"].values) # Our Evaluation metric for the competition is RMSLE. So let us use log of the target variable to build our models.



test_2 = test.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
# from scipy.stats import spearmanr

# import warnings

# warnings.filterwarnings("ignore")



# labels = []

# values = []

# for col in train.columns:

#     if col not in ["ID", "target"]:

#         labels.append(col)

#         values.append(spearmanr(train[col].values, train["target"].values)[0])

# corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

# corr_df = corr_df.sort_values(by='corr_values')

 

# corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]

# ind = np.arange(corr_df.shape[0])

# width = 0.9

# fig, ax = plt.subplots(figsize=(12,30))

# rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')

# ax.set_yticks(ind)

# ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

# ax.set_xlabel("Correlation coefficient")

# ax.set_title("Correlation coefficient of the variables")

# plt.show()
from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=99)

model.fit(X, y)



## plot the importances ##

feat_names = X.columns.values

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
feat_names[indices]
X[feat_names[indices]].head()
X_top_20_f = X[feat_names[indices]]
X_top_20_f.shape
# Doign the same for test dataset.

test_top_20_f = test_2[feat_names[indices]]

test_top_20_f.shape
from sklearn.model_selection import train_test_split 

  

X_train, X_valid, y_train, y_valid = train_test_split(X_top_20_f, y, test_size = 0.2, random_state = 0) 
from xgboost import XGBRegressor
clf_xgb = XGBRegressor()

clf_xgb.fit(X_train, y_train)
y_pred = abs(clf_xgb.predict(X_valid))
from sklearn.metrics import mean_squared_log_error

np.sqrt(mean_squared_log_error( y_valid, y_pred ))
pred_test_full = abs(clf_xgb.predict(test_top_20_f))

len(pred_test_full)
subm_df = pd.DataFrame({"ID":test["ID"].values})

subm_df["target"] = pred_test_full

subm_df.to_csv("XGBReg_v2.csv", index=False)
subm_df.head()