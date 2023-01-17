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
df = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
df.head()
df.isnull().sum()
cat_vars = df.select_dtypes(include = ['object']).columns
num_vars = df.select_dtypes(include = ['int64','float64']).columns
print(f' num_vars are {num_vars}\n cat_vars are {cat_vars}')
df['Type 2'].fillna('others',inplace = True)
df.isnull().sum()
mean_map = []
for i in df[cat_vars].columns:
    mean_map.append(df.groupby([i])['Total'].mean())
j = 0
for i in df[cat_vars].columns:
    df[i] = df[i].map(mean_map[j]);j+=1
df.head()
df.head()
df = df.drop(['#'],axis = 1)
df['Legendary'] = pd.get_dummies(df['Legendary'])
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
df.head()
plt.figure(figsize=(16,9))
sn.heatmap(df.corr(),annot=True,linewidths=3,linecolor='red')
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV,cross_val_score,train_test_split
temp = df.copy()
upper_limit = temp.HP.mean() + (temp.HP.std()*3)
lower_limit = temp.HP.mean() - (temp.HP.std()*3)
temp[(temp.HP > upper_limit) | (temp.HP < lower_limit)]
data_new = temp[(temp.HP<upper_limit) & (temp.HP>lower_limit)]
data_new.head()
data_new[(data_new.HP>upper_limit)|(data_new.HP<lower_limit)]
test = data_new['Total']

train = data_new.drop(['Total'],axis = 1)
mm = MinMaxScaler()
train_scaled = mm.fit_transform(train)
params = {
    'Random_forest':{
        'model':RandomForestRegressor(),
        'params':{
            'n_estimators':[int(x) for x in np.linspace(start=1,stop=1200,num = 10)],
            'max_depth':[int(x) for x in np.linspace(start=1,stop=30,num = 5)],
            'min_samples_split':[2,5,10,12],
            'min_samples_leaf':[2,5,10,12],
            'max_features':['auto','sqrt'],
            'ccp_alpha':[0.015,0.010,0.005]
        }
    },
    'SVR':{
        'model': SVR(gamma='auto'),
        'params':{
            'kernel':['rbf','poly','linear','sigmoid'],
            'C':[0.25,0.50,0.75,1.0],
            'tol':[1e-10,1e-5,0.005],
            
        }
    }
}
scores = []
for model_name,mp in params.items():
    clf = RandomizedSearchCV(mp['model'],mp['params'],cv = 5,verbose = 2,n_iter=10,scoring='neg_mean_squared_error')
    clf.fit(train_scaled,test)
    scores.append({
        'model_name':model_name,
        'best_score':clf.best_score_,
        'best_estimator':clf.best_estimator_,
    })
score_df = pd.DataFrame(scores,columns = ['model_name','best_score','best_estimator'])
score_df
X_train, X_test, y_train, y_test = train_test_split(train_scaled,test,test_size = 0.2)
from sklearn.model_selection import cross_val_score
rf_score = cross_val_score(RandomForestRegressor(ccp_alpha=0.015, max_depth=22, min_samples_leaf=2,
                      min_samples_split=5, n_estimators=533),X_train,y_train,cv = 10)
svr_score = cross_val_score(SVR(gamma='auto', kernel='linear', tol=1e-10),X_train,y_train,cv = 10)
print(f' mean score of rf is {rf_score.mean()}\n mean score of svr is {svr_score.mean()}')
rf_score = cross_val_score(RandomForestRegressor(ccp_alpha=0.015, max_depth=22, min_samples_leaf=2,
                      min_samples_split=5, n_estimators=533),X_test,y_test,cv = 10)
print(f' This is testing score:...\n mean score of rf is {rf_score.mean()}')
rf = RandomForestRegressor(ccp_alpha=0.015, max_depth=22, min_samples_leaf=2,
                      min_samples_split=5, n_estimators=533)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
y_pred = rf.predict(X_test)
sn.distplot(y_pred-y_test)
predict = []
for i in range(0,len(train_scaled)):
    predict.append(rf.predict([train_scaled[i]]))
predict = np.array(predict)
data_new['predict'] = predict
sn.distplot(data_new['Total'],label = 'Total',kde= True,hist=False)
sn.distplot(data_new['predict'],label = 'predicted',kde = True,hist=False)
plt.legend()
plt.show()
data_new.head()
