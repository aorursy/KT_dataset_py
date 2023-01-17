
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
import lightgbm as gbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("../input/train.csv",index_col=0)
test = pd.read_csv("../input/test.csv",index_col=0)
train.head(5)
train.info()
train.describe()
plt.figure(figsize=(8,6))
sns.countplot(train['type'])
plt.xticks(fontsize=20)
plt.show()
sns.countplot(train['color'])
plt.xticks(fontsize=15)
plt.show()
print(train.groupby('color')['type'].count())
g = sns.pairplot(train.loc[:,'bone_length':],hue="type",palette= 'husl')
train.groupby(['type','color'])['color'].count()
train_x=pd.get_dummies(train,columns=['color'],drop_first=True)
test_x = pd.get_dummies(test,columns=['color'],drop_first=True)
train_label = train_x['type']
train_x.drop('type',axis=1,inplace=True)
train_val = train_x.values
model = RandomForestClassifier()
model.fit(train_val,train_label)
imp=model.feature_importances_
feature_importances = pd.DataFrame(imp,index = train_x.columns,columns=['importance']).sort_values('importance',ascending=False)
plt.figure(figsize=(8,8))
plt.barh(feature_importances.index,feature_importances.importance,color='r')
plt.show()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
lb.fit(train_label)
train_nc=lb.transform(train_label)
train_nc
mod = gbm.LGBMClassifier()
train_lgb = gbm.Dataset(data = train_val,label = train_nc)
default = mod.get_params()
default
N_FOLDS = 3
MAX_EVALS=5
del default['n_estimators']
cv_results = gbm.cv(default, train_lgb, num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = N_FOLDS, seed = 42)
cv_results
test_val =test.values
mod.n_estimators = len(cv_results['auc-mean'])
mod.fit(train_val,train_nc)
predictions=mod.predict(test_x)
ans=lb.inverse_transform(predictions)
submission = pd.DataFrame({'id':test_x.index,'type':ans})
submission.to_csv('submit.csv',index=False)
