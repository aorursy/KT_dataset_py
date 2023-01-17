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
import pandas as pd
import seaborn as sn
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns',None)
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
def unique(df):
    for i in df.columns:
        print(f' feature <{i}> has {df[i].unique()} values')
        print('='*75)
def valuecounts(df):
    for i in df.columns:
        print(f' feature <{i}> has {df[i].value_counts()} value counts')
        print('='*75)
unique(df)
pd.set_option('display.max_rows',None)
valuecounts(df)
df = df.drop(['PassengerId'],axis = 1)
df.head()
px.bar(data_frame=df,x = 'Country',y = 'Survived',labels={'x':'Country','y':'Survived'})
age_df = df.sort_values(['Age'],ascending=False)
age_df.head()
px.bar(data_frame=age_df,x = age_df['Age'],y = age_df['Survived'],labels={'x':'Age','y':'Survived'})
px.bar(data_frame=age_df,x = age_df['Sex'],y = age_df['Survived'],labels={'x':'Age','y':'Survived'})
df = df.drop(['Firstname','Lastname'],axis = 1)
df.head()
df['Sex'] = pd.get_dummies(df['Sex'])
df['Category'] = pd.get_dummies(df['Category'])
df.head()
mean_map = df.groupby(['Country'])['Survived'].mean()
mean_map
data = df.loc[(df['Country']!='Belarus') & (df['Country']!='Canada')]
data.head()
mean_map1 = data.groupby(['Country'])['Survived'].mean()
mean_map1
data['Country'] = data['Country'].map(mean_map1)
data.head()
test = data['Survived']
train = data.drop(['Survived'],axis = 1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(train,test,test_size = 0.2)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
plt.figure(figsize=(16,9))
tree.plot_tree(clf,filled=True,feature_names=train.columns,class_names=['Survived','Not Survived'])
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas = path.ccp_alphas
alpha_list = []
for i in  ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0,ccp_alpha=i)
    clf.fit(X_train,y_train)
    alpha_list.append(clf)
train_score = [clf.score(X_train,y_train) for clf in alpha_list]
test_score =  [clf.score(X_test,y_test) for clf in alpha_list]

plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.plot(ccp_alphas,train_score,marker = 'o',label = 'training',color = 'magenta',drawstyle = 'steps-post')
plt.plot(ccp_alphas,test_score,marker = '+',label = 'testing',color = 'red',drawstyle = 'steps-post')
plt.legend()
plt.show()
dt = DecisionTreeClassifier(random_state=0,ccp_alpha=0.0075)
dt.fit(X_train,y_train)
plt.figure(figsize=(16,9))
tree.plot_tree(dt,filled=True,feature_names=train.columns,class_names=['Survived','Not Survived'])
params = {
    'RandomForest':{
        'model':RandomForestClassifier(),
        'params':{
            'criterion':['gini','entropy'],
            'n_estimators':[int(x) for x in np.linspace(100,1200,10)],
            'max_depth':[int(x) for x in np.linspace(1,30,5)],
            'max_features':['auto','sqrt','log2'],
            'ccp_alpha':[x for x in np.linspace(0.0050,0.0090,5)],
            'min_samples_split':[2,5,10,14],
            'min_samples_leaf':[2,5,10,14],
        }
    },
    'logistic':{
        'model':LogisticRegression(),
        'params':{
            'penalty':['l1', 'l2', 'elasticnet'],
            'C':[0.25,0.50,0.75,1.0],
            'tol':[1e-10,1e-5,1e-4,1e-3,0.025,0.25,0.50],
            'solver':['lbfgs','liblinear','saga','newton-cg'],
            'multi_class':['auto', 'ovr', 'multinomial'],
            'max_iter':[int(x) for x in np.linspace(start=1,stop=250,num=10)],
        }
    },
    'D-tree':{
        'model':DecisionTreeClassifier(),
        'params':{
            'criterion':['gini','entropy'],
            'splitter':['best','random'],
            'min_samples_split':[1,2,5,10,12],
            'min_samples_leaf':[1,2,5,10,12],
            'max_features':['auto','sqrt'],
            'ccp_alpha':[x for x in np.linspace(0.0050,0.0090,5)],
        }
    },
    'SVM':{
        'model':SVC(),
        'params':{
            'C':[0.25,0.50,0.75,1.0,5,10,100],
            'gamma':['scale',1,0.1,0.01,0.001,0.0001],
            'tol':[1e-10,1e-5,1e-4,0.025,0.50,0.75],
            'kernel':['linear','poly','sigmoid','rbf'],
            'max_iter':[int(x) for x in np.linspace(start=1,stop=250,num=10)],
        }
    },
    'G-Boost':{
        'model':GradientBoostingClassifier(),
        'params':{
            'n_estimators':[int(x) for x in np.linspace(100,1200,10)],
            'max_depth':[int(x) for x in np.linspace(1,30,5)],
            'max_features':['auto','sqrt','log2'],
            'ccp_alpha':[x for x in np.linspace(0.0050,0.0090,5)],
            'min_samples_split':[2,5,10,14],
            'min_samples_leaf':[2,5,10,14],
            'loss':['deviance', 'exponential'],
            'learning_rate':[0.1,0.5,1.0,1.5],
            'tol':[1e-10,1e-5,1e-4,0.025,0.50,0.75],
        }
    }
    
    
}
scores = []
for model_name,mp in params.items():
    clf = RandomizedSearchCV(mp['model'],param_distributions=mp['params'],cv = 5,n_iter=10,scoring='accuracy',verbose=2)
    clf.fit(X_train,y_train)
    scores.append({
        'model_name':model_name,
        'best_score':clf.best_score_,
        'best_estimator':clf.best_estimator_,
    })
scores_df = pd.DataFrame(scores,columns=['model_name','best_score','best_estimator'])
scores_df
for i in scores_df['best_estimator']:
    print(i)
    print('='*100)
rf = RandomForestClassifier(ccp_alpha=0.006, criterion='entropy', max_depth=8,
                       max_features='sqrt', min_samples_leaf=2,
                       min_samples_split=10, n_estimators=711)
lr = LogisticRegression(max_iter=167, multi_class='ovr', solver='liblinear',
                   tol=0.025)
dt = DecisionTreeClassifier(ccp_alpha=0.005, criterion='entropy',
                       max_features='sqrt', splitter='random')
svc = SVC(C=0.5, gamma=0.001, max_iter=250, tol=0.025)

gb = GradientBoostingClassifier(ccp_alpha=0.006, learning_rate=0.5,
                           loss='exponential', max_depth=1, max_features='sqrt',
                           min_samples_leaf=5, n_estimators=711, tol=1e-05)
xgb = XGBClassifier(booster='dart', gamma=0.8346938775510204, learning_rate=0.325,
              max_depth=22, n_estimators=344, reg_alpha=1.0, reg_lambda=3)
model_list = [rf,lr,dt,svc,gb,xgb]
for i in model_list:
    i.fit(X_train,y_train)
train_score = []
for i in model_list:
    print(f' model is <{i}_classifier> is \n training score is :{i.score(X_train,y_train)}')
    print('='*100)
train_score = []
for i in model_list:
    print(f' model is <{i}_classifier> is \n testing score is :{i.score(X_test,y_test)}')
    print('='*100)
rf = RandomForestClassifier(ccp_alpha=0.006, criterion='entropy', max_depth=8,
                       max_features='sqrt', min_samples_leaf=2,
                       min_samples_split=10, n_estimators=711)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
rf.score(X_test,y_test)
metrics.plot_confusion_matrix(rf,X_test,y_test,display_labels=['Survived','Not Survived'],cmap = 'plasma')
clf = DecisionTreeClassifier(ccp_alpha=0.006)
clf.fit(X_train,y_train)
plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True,feature_names=train.columns,class_names=['Survived','Not Survived'])
temp_train = np.array(train)
predict = []
for i in range(0,len(temp_train)):
    predict.append(rf.predict([temp_train[i]]))
predict = np.array(predict)
data['predict'] = predict
compare_list = ['Survived','predict']
for i in compare_list:
    print(f' feature <{i}> has {data[i].value_counts()}')
    print('='*100)
y_pred = rf.predict(X_test)
y_pred = np.array(y_pred)
y_test = np.array(y_test)
sn.distplot(y_test,hist = True,kde = False,color = 'magenta',label = 'Actual')
sn.distplot(y_pred,hist = True,kde = False,color='red',label='Predicted')
plt.legend()
plt.show()
report = metrics.classification_report(y_test,y_pred)
print(report)
