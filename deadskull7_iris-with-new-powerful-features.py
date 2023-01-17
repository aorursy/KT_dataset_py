# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Iris.csv")
df.head()
df.Species.value_counts()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.Species)
df["Species"]=le.transform(df.Species)
print(le.classes_)
temp=df.apply(lambda row: row['PetalLengthCm'] + row['PetalWidthCm'], axis=1)
print(df.Species.corr(pd.Series(temp)))
df['new1']=temp
temp=df.apply(lambda row: abs(row['SepalLengthCm'] - row['PetalLengthCm']), axis=1)
print(df.Species.corr(pd.Series(temp)))
df['new2']=temp
import xgboost as xgb
import matplotlib.pyplot as plt

train_y = df["Species"].values
train_X = df.drop(['Species','Id'], axis=1)

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'multi:softmax',
    'num_class':3 ,
    'eval_metric': 'merror',
   # 'base_score': np.mean(train_y), # base prediction = mean(target)
    'silent': 1
}

final = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params), final, num_boost_round=200, maximize=True)
fig, ax = plt.subplots(figsize=(6,2))
xgb.plot_importance(model, max_num_features=7, height=0.8, ax=ax, color = 'coral')
print("Feature Importance by XGBoost")
plt.show()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=420)
model.fit(train_X, train_y)
feat_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:7]

plt.subplots(figsize=(6,2))
plt.title("Feature importances by RandomForestRegressor")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), feat_names[indices], rotation='horizontal')
plt.ylim([-1, len(indices)])
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
from sklearn import svm
from sklearn.metrics import accuracy_score
clf = svm.SVC()
clf.fit(x_train, y_train)
pred=clf.predict(x_valid)
accuracy_score(y_valid, pred)
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.3, random_state=42)
from sklearn import svm
from sklearn.metrics import accuracy_score
clf = svm.SVC()
clf.fit(x_train, y_train)
pred=clf.predict(x_valid)
accuracy_score(y_valid, pred)
