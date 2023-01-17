# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/hw03-dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import lightgbm as lgb
df = pd.read_csv("/kaggle/input/hw03-dataset/hw03_train.csv")
print("df data:", df.shape)
df.dtypes

def create_one_hot_encodings(df):
    '''
    Creates one-hot encodings for the non-numerical columns in the input dataframe
    '''
    return pd.get_dummies(df, columns=[c for c in df.columns if df[c].dtype == 'object'])
df = create_one_hot_encodings(df)
print("Shape after converting categorical columns into 1-hot encodings:", df.shape)
df.isna().sum()*100/len(df)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
def scale_data(input_df):
    scaler = StandardScaler()
    return scaler.fit_transform(input_df)
y = df[df['target'].notna()]['target']
X_train, X_holdout, y_train, y_holdout = train_test_split(
    df[df['target'].notna()].values, y, test_size=0.3, random_state=20)

#Scale features
X_train = scale_data(X_train)
X_holdout = scale_data(X_holdout)
print("X_train_scaled shape:", X_train.shape)
print("X_train_scaled shape:", X_holdout.shape)
d_train = lgb.Dataset(X_train, label=y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 30
params['min_data'] = 50
params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)
print("light gbm model:", clf)
#Prediction
y_pred=clf.predict(X_holdout)

#convert into binary values
for i in range(0,len(y_pred)):
    if y_pred[i]>=.3:       # setting threshold to .3
       y_pred[i]=1
    else:  
       y_pred[i]=0
y_pred
y_holdout
#Accuracy calculation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_pred,y_holdout.values))
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_holdout.values)
cm
from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_preds = xgb.predict(X_holdout)
#Accuracy calculation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_preds,y_holdout.values))
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_holdout.values)
cm
