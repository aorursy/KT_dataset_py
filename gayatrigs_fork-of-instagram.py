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
corr_mat=data_train.corr().round(2)

import seaborn as sns
sns.heatmap(corr_mat, annot=True)
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


# from imblearn.over_sampling import SMOTE
# smote = SMOTE(sampling_strategy='auto')
# X_train, Y_train = smote.fit_sample(X_train, Y_train)
# print(X_train.shape,Y_train.shape)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import Ridge
r_estimator=LogisticRegression(max_iter = 1000)

parameters={'penalty' : ['l1', 'l2', 'elasticnet' ],
               'class_weight' : ['balanced'],
           'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid=GridSearchCV(estimator=r_estimator,param_grid=parameters,cv=7,verbose=True,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_

from  sklearn.linear_model import RidgeClassifier

model =RidgeClassifier()

model.fit(X_train, Y_train)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 500)

model.fit(X_train, Y_train)
a = model.feature_importances_
for i in range(1, len(a)):
    if a[i] < 0.01:
        print('the columns ', X.columns[i], ' has feature value ', a[i])
    

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight = 'balanced',
                          penalty = 'l1',
                          solver = 'liblinear',
                          max_iter = 1000)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
pred
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, auc, roc_curve

confusion_matrix(pred, Y_test)

recall_score(Y_test, pred)
f1_score(pred,Y_test)
print(classification_report(pred, Y_test))
import matplotlib.pyplot as plt
fpr, tpr, thres = roc_curve(Y_test,  pred)
plt.scatter(fpr, tpr)
thres
from sklearn.metrics import roc_auc_score
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
# import plotly.graph_objects as go
# import numpy as np


# N = 70

# fig = go.Figure(data=[go.Mesh3d(x=200*data['#follows'],
#                    y=200*data['nums/length username'],
#                    z=200*data['#posts'],
#                    opacity=0.5,
#                    color='rgba(244,22,100,0.6)'
#                   )])

# fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[-100,100],),
#                      yaxis = dict(nticks=4, range=[-50,100],),
#                      zaxis = dict(nticks=4, range=[-100,100],),),
#     width=700,
#     margin=dict(r=20, l=10, b=10, t=10))

# fig.show()