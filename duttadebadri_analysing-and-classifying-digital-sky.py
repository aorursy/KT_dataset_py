# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

SMALL_SIZE = 10
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150
# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
df.head()
df.describe()
df.info()
columns = df.columns
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing')
sns.heatmap(df.corr())
df.corr()
df.drop(['specobjid','fiberid'],axis=1,inplace=True)
cnt_srs = df['class'].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Class distribution',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Ratings")
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(df[df['class']=='STAR'].redshift, bins = 30, ax = axes[0], kde = False)
ax.set_title('Star')
ax = sns.distplot(df[df['class']=='GALAXY'].redshift, bins = 30, ax = axes[1], kde = False)
ax.set_title('Galaxy')
ax = sns.distplot(df[df['class']=='QSO'].redshift, bins = 30, ax = axes[2], kde = False)
ax = ax.set_title('QSO')
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))
ax = sns.lvplot(x=df['class'], y=df['dec'], palette='coolwarm')
ax.set_title('dec')

di={'STAR':1,'GALAXY':2,'QSO':3}
df.replace({'class':di},inplace=True)

y=df['class']
df.drop(['objid','class'],axis=1,inplace=True)
dx=df[['ra','dec','u','g','r','i','z','run','rerun','camcol','field','redshift','plate','mjd']]
for i in dx.columns:
    plt.figure(figsize=(12,8))
    sns.boxplot(y=i, data=df)
    plt.ylabel(i+'Distribution', fontsize=12)
    plt.title(i+"Distribution", fontsize=14)
    plt.xticks(rotation='vertical')
    plt.show()
import umap

embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(df.iloc[:20000, 1:])

plt.figure(figsize=(12,12))
plt.scatter(embedding[:20000, 0], embedding[:20000, 1], 
            c=df.iloc[:20000, 0], 
            edgecolor='none', 
            alpha=0.80, 
            s=10)
plt.axis('off');
df.head(1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
sdss = scaler.fit_transform(df)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, 
                                                    y, test_size=0.33)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100
print("Accuracy of KNN: ", acc_knn)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
acc_gv_rbf = (grid_predictions == y_test).sum().astype(float) / len(grid_predictions)*100
print("Accuracy of KNN: ", acc_gv_rbf)
'''param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['linear']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)'''
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,y_train)
preds2=gnb.predict(X_test)
acc_gnb=(preds2==y_test).sum().astype(float)/len(preds)*100
print("Accuracy of Naive Bayes: ",acc_gnb)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
preds3=rf.predict(X_test)
acc_rf=(preds3==y_test).sum().astype(float)/len(preds)*100
print("Accuracy of Random Forest Classifier: ",acc_rf)
import xgboost as xgb

xgb=xgb.XGBClassifier()
xgb.fit(X_train,y_train)
preds4=xgb.predict(X_test)
acc_xgb=(preds4==y_test).sum().astype(float)/len(preds)*100
print("Accuracy of XGBoost Classifier: ",acc_xgb)
import lightgbm as lgb

lgb=lgb.LGBMClassifier()
lgb.fit(X_train,y_train)
preds5=lgb.predict(X_test)
acc_lgb=(preds5==y_test).sum().astype(float)/len(preds)*100
print("Accuracy of LightGBM Classifier: ",acc_lgb)
trace1 = go.Bar(
    x=['KNN','Naive Bayes','Random Forest','XGBoost','LightGBM'],
    y=[acc_knn,acc_gnb,acc_rf,acc_xgb,acc_lgb],
    name = 'Accuracy Comparisons of the 4 algorithms',
        marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic'
    ),
)

layout = go.Layout(
    title='Accuracy Score Ratio'
)

data = [trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Ratio")
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['multi:softmax'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb, parameters, n_jobs=5, 
                   cv=StratifiedKFold(y_train, n_folds=5, shuffle=True),
                   verbose=2, refit=True)

clf.fit(X_train, y_train)
preds6=clf.predict(X_test)

acc_xgbpt=(preds6==y_test).sum().astype(float)/len(preds)*100
print("Accuracy of XGBoost Classifier after parameter tuning: ",acc_xgbpt)
print("Accuracy decreased by =",(acc_xgb-acc_xgbpt),"% after parameter tuning with GridSearchCV")
from keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
from keras.models import Sequential
from keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(50, activation = "relu", input_shape=(14, )))
# Hidden - Layers
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(50, activation = "relu"))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(50, activation = "relu"))
# Output- Layer
model.add(Dense(4, activation = "softmax"))
model.summary()
model.compile(
 optimizer = "RMSProp",
 loss = "categorical_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 X_train, y_train,
 epochs= 10,
 batch_size = 32,
 validation_data = (X_test, y_test)
)

