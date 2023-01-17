# %% [markdown]
# This Python 3 environment comes with many helpful analytics libraries installed
#It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#For example, here's several helpful packages to load in 
  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
from sklearn.model_selection import cross_val_score, train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from collections import Counter
 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
  
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
  
HouseID = df_test['Id']
df_train = df_train.drop(['Id'], axis=1)
df_test = df_test.drop(['Id'], axis=1)
target = df_train['SalePrice']
warnings.simplefilter(action='ignore', category=FutureWarning)
categorical = np.array([])
numerical = np.array([])
c_test = np.array([])
n_test = np.array([])
c_labels = np.array([])
n_labels = np.array([])

for i in df_train:
    if df_train[i].dtypes=='object':
        categorical = np.append(categorical, df_train[i])
        c_labels = np.append(c_labels, i)
    else:
        numerical = np.append(numerical, df_train[i])
        n_labels = np.append(n_labels, i)

categorical = np.array(np.array_split(categorical, len(categorical)/len(df_train)))
numerical = np.array(np.array_split(numerical, len(numerical)/len(df_train)))

for i in df_test:
    if df_test[i].dtypes=='object':
        c_test = np.append(c_test, df_test[i])
    else:
        n_test = np.append(n_test, df_test[i])
        
c_test = np.array(np.array_split(c_test, len(c_test)/len(df_test)))
n_test = np.array(np.array_split(n_test, len(n_test)/len(df_test)))
n_missing = (pd.DataFrame(numerical).T.isnull().sum()/numerical.shape[1])*100
c_missing = np.array((pd.DataFrame(categorical).T.isnull().sum()/categorical.shape[1])*100)

for i in c_missing:
    if i > 40:
        c_labels = np.delete(c_labels, list(c_missing).index(i), 0)
        categorical = np.delete(categorical, list(c_missing).index(i), 0)
        c_test = np.delete(c_test, list(c_missing).index(i), 0)
        c_missing = np.delete(c_missing, list(c_missing).index(i), 0)

c_missing = pd.DataFrame(c_missing)
categorical = np.array(pd.DataFrame(categorical).fillna('nan'))
c_test = np.array(pd.DataFrame(c_test).fillna('nan'))
numerical = numerical[:36]
a=0

while a < len(numerical):
    b = pd.DataFrame(numerical[a]).fillna(float(pd.DataFrame(numerical[a]).mean(skipna=True)))
    c = pd.DataFrame(n_test[a]).fillna(float(pd.DataFrame(n_test[a]).mean(skipna=True)))
    numerical[a] = np.array(b).T[0]
    n_test[a] = np.array(c).T[0]
    a += 1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_test = LabelEncoder()
num = np.array([])
num_test = np.array([])

for i in range(categorical.shape[0]):
    le.fit(categorical[i])
    num = np.append(num, le.transform(categorical[i]))
    
for i in range(c_test.shape[0]):
    le_test.fit(c_test[i])
    num_test = np.append(num_test, le_test.transform(c_test[i]))
    
num = np.array(np.array_split(num, categorical.shape[0]))
num_test = np.array(np.array_split(num_test, c_test.shape[0]))
X = pd.DataFrame(np.array_split(np.append(numerical, num), 74)).T
test = pd.DataFrame(np.array_split(np.append(n_test, num_test), 74)).T
y = target
plt.figure(figsize=(9, 5))

heatmap = sns.heatmap(X.corr())

for i in range(categorical.shape[0]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    ax1.bar(Counter(categorical[i]).keys(), list(Counter(num[i]).values()))
    ax2.pie(list(Counter(num[i]).values()), labels=Counter(categorical[i]).keys())
    ax3.scatter(num[i], target)
    fig.suptitle(c_labels[i], fontsize=16)

for i in range(numerical.shape[0]):
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 3))
    ax1.scatter(numerical[i], target)
    fig.suptitle(n_labels[i], fontsize=16)
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.075, random_state=1, shuffle=True)

algorithms = {
    'Ridge': Ridge(),
    'Lasso': Lasso(tol=0.01),
    'ElasticNet': ElasticNet(tol=0.1),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100),
    'XGB Regressor': XGBRegressor(colsample_bylevel=0.9, learning_rate=0.1, reg_lambda=0.1, n_estimators=150, max_depth=5, objective='reg:squarederror'),
    'SVR': SVR(gamma='auto'),
    'Decision Tree': DecisionTreeRegressor()
}

scoring = np.array([])
error = np.array([])

i = 0
while i < len(list(algorithms.keys())):
    print(list(algorithms.keys())[i])
    model = list(algorithms.values())[i]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model score:            ' + str(model.score(X_test, y_test)*100) + '%')
    scores = cross_val_score(model, X_test, y_test, cv=5).mean()
    print('Cross validation score: ' + str(scores*100) + '%')
    mae_error = mae(y_test, y_pred)
    print('Mean absolute error:    ' + str(mae_error))
    scoring = np.append(scoring, (model.score(X_test, y_test)+cross_val_score(model, X_test, y_test, cv=5).mean())/2)
    error = np.append(error, mae_error)
    print('')
    i+=1

plt.figure(figsize=(15, 5))
plt.bar(list(algorithms.keys()), scoring)
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Preformance of ML regression models')
plt.show()
model = list(algorithms.values())[list(error).index(error.min())]
model.fit(X_train, y_train)
predictions = model.predict(test)

print('Selected model: ' + str(list(algorithms.keys())[list(error).index(error.min())]))

output = pd.DataFrame({'Id': HouseID, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)