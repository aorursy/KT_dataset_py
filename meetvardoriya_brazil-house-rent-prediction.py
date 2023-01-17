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
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')
df.head()
def unique(df):
    for i in df.columns:
        print(f' feature <{i}> has {df[i].unique()} values')
        print("="*100)
        
def valuecounts(df):
    for i in df.columns:
        print(f' feature <{i}> has {df[i].value_counts()} value counts')
        print("="*100)
pd.set_option('display.max_rows',None)
valuecounts(df)
df['property tax'].value_counts()
df['hoa'] = df['hoa'].replace('Sem info',0)
df['hoa'] = df['hoa'].replace('Incluso',0)
df['property tax'] = df['property tax'].replace('Incluso',0)
df['property tax'] = df['property tax'].replace('Sem info',0)
num_vars = df.select_dtypes(include=['int64','float64']).columns
cat_vars = df.select_dtypes(include=['object']).columns
print(f' num vars are {num_vars}\n\n cat vars are {cat_vars}')

to_use_list = ['hoa', 'rent amount', 'property tax',
       'fire insurance', 'total']
for i in to_use_list:
    df[i] = df[i].astype('str')
df.head(3)
def replacements(value):
    out = value.replace('R$','')
    out1 = out.replace(',','')
    out1 = float(out1)
    return out1
for i in to_use_list:
    df[i] = df[i].apply(lambda x: replacements(x))
df.head(2)
num_vars = df.select_dtypes(include=['int64','float64']).columns
cat_vars = df.select_dtypes(include=['object']).columns
print(f' num vars are {num_vars}\n\n cat vars are {cat_vars}')
df['floor'] = df['floor'].replace('-',0)
df['floor'] = df['floor'].astype('int64')
floor_median = df['floor'].median()
df['floor']  = df['floor'].replace(0,floor_median)
df.head()
df = pd.get_dummies(data=df,columns=['animal','furniture'],drop_first=True)
df.head(3)
df = df.drop(['Unnamed: 0'],axis = 1)
df.columns = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'hoa',
       'rent amount', 'property tax', 'fire insurance', 'total',
       'furniture', 'animals']
col_list = ['city', 'area', 'rooms', 'bathroom', 'furniture', 'animals','parking spaces', 'floor', 'hoa',
       'rent amount', 'property tax', 'fire insurance', 'total']
df = df.reindex(columns=col_list)
df.head(3)
import scipy.stats as stats
def plots(df,var):
    plt.figure(figsize=(16,9))
    plt.subplot(1,3,1)
    plt.hist(df[var],color = 'aqua')
    
    plt.subplot(1,3,2)
    stats.probplot(df[var],dist='norm',plot=plt)
    
    plt.subplot(1,3,3)
    sn.boxplot(y = df[var],color='blue')
    
    plt.show()
for i in df.columns:
    print(f' plots of feature <{i}> are shown below ↓')
    plots(df,i)
    print("="*75)
temp = df.copy()
for i in to_use_list:
    temp[i],impure = stats.boxcox(temp[i]+1)
for i in to_use_list:
    print(f' plots of feature <{i}> is shown below ↓')
    plots(temp,i)
    print('='*75)
upper_list = []
lower_list = []
for i in df.columns:
    upper_list.append(df[i].mean() + (df[i].std())*3)
    lower_list.append(df[i].mean() - (df[i].std())*3)
    
    
j = 0
for i in df.columns:
    temp = df.loc[(df[i]>upper_list[j]) & (df[i]<lower_list[j])];j+=1
temp
iqr1 = df.quantile(0.25)
iqr2 = df.quantile(0.75)
iqr = iqr2-iqr1
iqr
temp = df[df['total']<=5622.5]
print(f' shape of dataframe df is {df.shape}\n\n shape of dataframe after outlier removal of total is {temp.shape}')
temp.columns
sn.boxplot(data=temp,x = 'bathroom',y = 'total',palette=['aqua','magenta'])
plt.figure(figsize=(16,9))
sn.heatmap(temp.corr(),annot=True,linewidths=3,linecolor='red',cmap='plasma')
test = temp['total']

train = temp.drop(['total'],axis = 1)

from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import metrics
X_train, X_test, y_train, y_test =  train_test_split(train,test,test_size = 0.2)
print(f' training shape is {X_train.shape}\n\n testing shape is {X_test.shape}')
clf = DecisionTreeRegressor(random_state=0)
clf.fit(X_train,y_train)
plt.figure(figsize = (25,12.5))
tree.plot_tree(clf,filled=True,feature_names=train.columns,class_names=['Price'])
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alpha = path.ccp_alphas
alpha_list = []
for i in ccp_alpha:
    clf = DecisionTreeRegressor(ccp_alpha=i,random_state=0)
    clf.fit(X_train,y_train)
    alpha_list.append(clf)
train_score = [clf.score(X_train,y_train) for clf in alpha_list]
test_score = [clf.score(X_test,y_test) for clf in alpha_list]
plt.plot(ccp_alpha,train_score,label = 'Traning',color = 'red',marker = 'o',drawstyle = 'steps-post')
plt.plot(ccp_alpha,test_score,label = 'Testing',color = 'black',marker = '+',drawstyle = 'steps-post')
plt.legend()
plt.show()
params = {
    'RandomForest':{
        'model':RandomForestRegressor(),
        'params':{
            'n_estimators':[int(x) for x in np.linspace(start=100,stop=1200,num=10)],
            'max_features':['auto','sqrt','log2'],
            'max_depth':[int(x) for x in np.linspace(start=1,stop=30,num=5)],
            'min_samples_split':[2,5,10,12],
            'min_samples_leaf':[2,5,10,12],
            'ccp_alpha':[int(x) for x in np.linspace(0,200000,5)],
        }
    },
    'DecisionTree':{
        'model':DecisionTreeRegressor(),
        'params':{
            #'criterion':['gini','entropy'],
            'max_depth':[int(x) for x in np.linspace(start=1,stop=30,num=5)],
            'min_samples_split':[2,5,10,12],
            'min_samples_leaf':[2,5,10,12],
            'ccp_alpha':[int(x) for x in np.linspace(0,200000,5)],
            'splitter':['best','random'],
        }
    },
   
     'Linearreg':{
        'model': LinearRegression(),
        'params':{},
    },
    'Lasso':{
        'model':Lasso(),
        'params':{
            'alpha':[0.25,0.50,0.75,1.0,1.5,2.0],
            'max_iter':[int(x) for x in np.linspace(100,1500,10)],
            'tol':[1e-10,1e-5,1e-4,1e-3,0.05,0.25,0.50],
            'selection':['cyclic', 'random'],
        }
    },
    'SVR':{
        'model':SVR(gamma = 'auto'),
        'params':{
            'kernel':['rbf','linear','poly','sigmoid'],
            'tol':[1e-10,1e-5,1e-4,1e-3,0.05,0.25,0.50],
            'C':[0.005,0.025,0.25,0.50,0.75,1.0],
            'max_iter':[int(x) for x in np.linspace(1,250,5)],
        }
    },
    'Ridge':{
        'model':Ridge(),
        'params':{
            'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'alpha':[0.25,0.50,0.75,1.0,1.5,2.0],
            'max_iter':[int(x) for x in np.linspace(100,1500,10)],
            'tol':[1e-10,1e-5,1e-4,1e-3,0.05,0.25,0.50],
        }
    }

}
scores = []
for model_name,mp in params.items():
    clf = RandomizedSearchCV(mp['model'],param_distributions=mp['params'],cv = 5,n_iter = 10,scoring = 'neg_mean_squared_error',verbose = 2)
    clf.fit(X_train,y_train)
    scores.append({
        'model_name':model_name,
        'best_score': clf.best_score_,
        'best_estimator':clf.best_estimator_,
    })
scores_df = pd.DataFrame(scores,columns = ['model_name','best_score','best_estimator'])
scores_df
for i in scores_df['best_estimator']:
    print(i)
    print("="*60)
rf = RandomForestRegressor(ccp_alpha=0, max_depth=22, min_samples_leaf=10,
                      n_estimators=711)
dt = DecisionTreeRegressor(ccp_alpha=100000, max_depth=15, min_samples_leaf=2,
                      min_samples_split=12)
lr = LinearRegression()

ls = Lasso(alpha=0.25, max_iter=1188, selection='random', tol=1e-05)

svr = SVR(C=0.005, gamma='auto', kernel='linear', max_iter=250, tol=0.05)

rid = Ridge(alpha=1.5, max_iter=100, solver='lsqr')
model_list = [rf,dt,lr,ls,svr,rid]
for i in model_list:
    i.fit(X_train,y_train)
    print(f' model <{i}>\n\n training score is {i.score(X_train,y_train)}')
    print('='*100)
for i in model_list:
    print(f' model <{i}> \n\n testing score is {i.score(X_test,y_test)}')
    print("="*100)
rd = Ridge(alpha=1.5, max_iter=100, solver='lsqr')
rd.fit(X_train,y_train)
rd.score(X_train,y_train)
predict =  []
train_test = np.array(train)
for i in range(0,len(train_test)):
    predict.append(rd.predict([train_test[i]]))
predict = np.array(predict)
temp['Ridge_predicted'] = predict
temp.head()
rf = RandomForestRegressor(ccp_alpha=0, max_depth=22, min_samples_leaf=10,
                      n_estimators=711)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
predict =  []
train_test = np.array(train)
for i in range(0,len(train_test)):
    predict.append(rf.predict([train_test[i]]))
predict = np.array(predict)
temp['RF_predicted'] = predict
temp.head()
sn.set_style(style='darkgrid')
sn.distplot(temp['total'],kde = True,hist=False,label = 'Actual Price',color = 'red')
sn.distplot(temp['Ridge_predicted'],kde = True,hist = False,label = 'Predicted Price',color = 'blue')
plt.legend()
plt.show()
sn.set_style(style='darkgrid')
sn.distplot(temp['total'],kde = True,hist=False,label = 'Actual Price',color = 'red')
sn.distplot(temp['RF_predicted'],kde = True,hist = False,label = 'Predicted Price',color = 'blue')
plt.legend()
plt.show()
y_pred = rd.predict(X_test)
print(f' mean absolute error is : {metrics.mean_absolute_error(y_test,y_pred)}')
print("="*60)
print(f' mean squared error is :{metrics.mean_squared_error(y_test,y_pred)}')
print("="*60)
print(f' root mean squared error is :{np.sqrt(metrics.mean_squared_error(y_test,y_pred))}')
print("="*60)
