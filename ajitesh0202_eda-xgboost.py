import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/train.csv")
train.head()
sns.boxplot(train['popularity'])
#For detecting outliners wrt Revenue

sns.jointplot(train.budget, train.revenue);

plt.show()
indexNames = train[ train['budget'] >= 200000000].index

train.drop(indexNames , inplace=True)
sns.jointplot(train.popularity, train.revenue);

plt.show()
indexNames = train[ train['popularity'] >= 70].index

train.drop(indexNames , inplace=True)
sns.jointplot(train.runtime, train.revenue);

plt.show()
indexNames = train[ train['runtime'] <= 50 ].index

train.drop(indexNames , inplace=True)

indexNames = train[ train['runtime'] >= 225 ].index

train.drop(indexNames , inplace=True)
#Percentage of missing values in each column

columns = train.columns

percent_missing = train.isnull().sum() * 100 / len(train)

missing_value_df = pd.DataFrame({'percent_missing': percent_missing})

print(missing_value_df)
train = train.drop(['belongs_to_collection'],axis=1)
f, ax = plt.subplots(figsize=(10, 10))

sns.barplot(x='original_language', y='revenue', data=train);
f, ax = plt.subplots(figsize=(10, 10))

sns.barplot(x='status', y='revenue', data=train);
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

train['status']=labelencoder.fit_transform(train['status'])
train['original_language']=labelencoder.fit_transform(train['original_language'])
features = ['original_language','runtime', 'budget','popularity','status']

X = train[features]

y = train.revenue

#splitting the data into training and validation to check validity of the model



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=1)
#Using Xgbost for Regression

import xgboost

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)

best_xgb_model.fit(X_train,y_train)
test = pd.read_csv("../input/test.csv")

test.head()

test.isnull().sum()
cleanup_nums = {"status":     {"Released": 0, "Post Production": 1,"Rumored":2} }

test.replace(cleanup_nums, inplace=True)

test['status'] = test['status'].astype(float)

test['status'].dtypes
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

test['original_language']=labelencoder.fit_transform(test['original_language'])

test['status']=labelencoder.fit_transform(test['status'])

y_pred=best_xgb_model.predict(test[features])

test['revenue'] = best_xgb_model.predict(test[features])

filename = 'submission.csv'

pd.DataFrame({'id': test.id, 'Revenue': test.revenue}).to_csv(filename, index=False)
sub= pd.read_csv("submission.csv")