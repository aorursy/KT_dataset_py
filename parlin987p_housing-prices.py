
import numpy as np # linear algebra, ps=[0.001,0.01, ps=[0.001,0.01], ], 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing  # standartize the data
from sklearn.model_selection import train_test_split
#from sklearn import metrics

# import path
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))
# path=Path("../input/home-data-for-ml-course/")
# path.ls()
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
train.tail() 
train.info()
train = train.dropna(thresh=0.8*len(train), axis=1) # Drop any column with more than 80% missing values
test = test.dropna(thresh=0.8*len(test), axis=1)
print(train.shape) ;
print(test.shape)
train.isna().sum()  ; test.isna().sum()  
#encoding
df = pd.get_dummies(train)
#filling NA's with the mean of the column:
df = df.fillna(df.mean())
col = train.corr().nlargest(10, 'SalePrice')['SalePrice'].index
corr_matrix = np.corrcoef(df[col].values.T)
plt.figure(figsize = (10,8))
sns.heatmap(corr_matrix, cmap = 'coolwarm', annot = True, xticklabels= col.values, yticklabels= col.values);
# Visulazing the distibution of the data for every feature
train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));
X = df.drop('SalePrice', axis=1)
y=df['SalePrice']
#standartize data
normalized_X = preprocessing.normalize(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
def metrics (predict, target):
    mae = (abs(predict-target)).mean()
    mse = ((predict-target)**2).mean()
    rmse = np.sqrt(((predict - target) ** 2).mean())
    #return mae,mse, rmse
    print("the mae is {} \n the mse is {} \n the rmse is {} " .format(mae, mse,rmse))
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_val)

metrics(predictions,y_val)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier (n_estimators = 200)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_val)
metrics(predictions,y_val)
# from sklearn.ensemble import GradientBoostingClassifier
# gbm_model = GradientBoostingClassifier()
# gbm_model.fit(X_train, y_train)
# predictions = gbm_model.predict(X_train)
# metrics(predictions,y_val)
from xgboost import XGBRegressor
                     

my_model = XGBRegressor(n_estimators=500, learning_rate=0.01, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_val, y_val)],
             verbose=False)
predictions = my_model.predict(X_val)
metrics(predictions,y_val)
import lightgbm as lgb

# feature_cols = train.columns.drop('SalePrice')

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_val, label=y_val)

param = {'num_leaves': 120, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=64, verbose_eval=False)
predictions = bst.predict(X_val)
metrics(predictions,y_val)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)


predictions = neigh.predict(X_val)
metrics(predictions,y_val)

# Predicted class
# print(neigh.predict(X_valid))
error_rate = []
for i in range(1,80):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_val)
    error_rate.append(np.mean(pred_i != y_val))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,80),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# NOW WITH K=2
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
pred = knn.predict(X_val)

predictions = knn.predict(X_val)
metrics(predictions,y_val)

submit = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submit.head()