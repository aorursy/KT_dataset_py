# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train=pd.read_csv('/kaggle/input/instagram-fake-spammer-genuine-accounts/train.csv')
data_train
data_test=pd.read_csv('/kaggle/input/instagram-fake-spammer-genuine-accounts/test.csv')
data_test
data = pd.concat([data_train, data_test], axis=0)
data
data.info()
data.nunique()
data.describe()
from scipy.stats import skew
data['fake'].skew()
corr_mat=data.corr().round(2)
import seaborn as sns
sns.heatmap(corr_mat,annot=True)
data_train.skew()
data_train['#followers'] = np.log1p(data_train['#followers'])
data_train['#posts'] = np.log1p(data_train['#posts'])
X=data_train.drop(columns=['fake'])
Y=data_train['fake']
Y.value_counts()
X.drop(columns=['nums/length fullname','name==username', 'external URL'], inplace=True)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
Y_train.value_counts()
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
X_train, Y_train = rus.fit_resample(X_train, Y_train)
#from imblearn.over_sampling import SMOTE
#smote = SMOTE(sampling_strategy='auto')
#X_train, Y_train = smote.fit_sample(X_train, Y_train)
#print(X_train.shape,Y_train.shape)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
r_estimator=LogisticRegression()

parameters={'penalty' : ['l1', 'l2', 'elasticnet' ],
               'class_weight' : ['balanced'],
           'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid=GridSearchCV(estimator=r_estimator,param_grid=parameters,verbose=True,cv=7,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_
from  sklearn.linear_model import RidgeClassifier
model =RidgeClassifier()
model.fit(X_train, Y_train)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 500,max_features='auto')

model.fit(X_train, Y_train)
a = model.feature_importances_
for i in range(1, len(a)):
    if a[i] < 0.01:
        print('the columns ', X.columns[i], ' has feature value ', a[i])
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight = 'balanced',
                          penalty = 'l1',
                          solver = 'liblinear',max_iter = 1000)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
pred
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, auc, roc_curve
confusion_matrix(pred, Y_test)
precision_score(Y_test, pred)
recall_score(Y_test, pred)
f1_score(pred,Y_test)
print(classification_report(pred, Y_test))
roc_curve(pred,Y_test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
dtr = RandomForestRegressor()
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(Y_test,y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(Y_test,y_pred)
print("R2 Score:", r2)
dtr = xgb.XGBRegressor()
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(Y_test, y_pred)
print("R2 Score:", r2)
a = dtr.feature_importances_
for i in range(1, len(a)):
        print('the columns ', X.columns[i], ' has feature value ', a[i])
import seaborn as sns
plt.figure(figsize=(10,7))
corr_mat=data.corr()
sns.heatmap(data=corr_mat>0.2,annot=True)
X=data.drop(columns=['profile pic','fullname words','fake'])
Y=data['fake']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
dtr = RandomForestRegressor()
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)                                         

mse = mean_squared_error(Y_test,y_pred)
print("RMSE Error:", np.sqrt(mse))                           #good rmse
r2 = r2_score(Y_test,y_pred)
print("R2 Score:", r2)
dtr = xgb.XGBRegressor()
dtr.fit(X_train,Y_train)
y_pred = dtr.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))
r2 = r2_score(Y_test, y_pred)
print("R2 Score:", r2)
data["fake_followers_count"] = data.groupby(['#followers'])['fake'].transform('count')

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
fpr, tpr, thres = roc_curve(Y_test,  pred)
plt.scatter(fpr, tpr)
thres
roc_auc_score(pred, Y_test)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba)
auc = roc_auc_score(Y_test, pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
plt.scatter(fpr, tpr)
plt.show()
import numpy as np
plt.plot(fpr,tpr)
plt.show() 
auc = np.trapz(tpr,fpr)
print('AUC:', auc)