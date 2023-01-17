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
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('/kaggle/input/bank-loan-classification/UniversalBank.csv')
df.head()
df.isnull().sum()
df.drop(['ID','ZIP Code'], axis = 1, inplace = True)
cols = ['Family','Education', 'Securities Account', 'CD Account',
       'Online', 'CreditCard']

df = pd.get_dummies(df, columns = cols,drop_first = True)
y = df.loc[:,['Personal Loan']]
X = df.drop('Personal Loan', axis =1)

train_x, test_x, train_y, test_y = train_test_split(X,y, test_size = 0.3, random_state = 42)
scaler = StandardScaler()
scaler.fit(train_x)
scaler.transform(train_x)
scaler.transform(test_x)
knn_classifier = KNeighborsClassifier(n_neighbors=1)
cross_val = cross_val_score(knn_classifier,train_x,train_y, cv = 10)
print('Cross Val Score: {}%'.format(round(cross_val.mean()*100,2)))
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(knn,train_x,train_y, cv = 10)
    error_rate.append(1 - score.mean())
    
plt.plot(range(1,40), error_rate, color = 'red', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 10)
plt.title('K value vs Error rate')
plt.xlabel('K value')
plt.ylabel('Error rate')
knn1 = KNeighborsClassifier(n_neighbors = 21)
cross_val = cross_val_score(knn1, train_x, train_y,cv = 10)
print('Cross Val Score: {}%'.format(round(cross_val.mean()*100,2)))
k_range = list(range(1,40))
weight_options = ["uniform","distance"]
param_grid = {'n_neighbors' : k_range, 'weights' : weight_options}

knn = KNeighborsClassifier()
grid = GridSearchCV(knn,param_grid,cv = 8)
grid.fit(train_x,train_y)
print(grid.best_score_)
print(grid.best_params_)
knn1 = KNeighborsClassifier(n_neighbors = 20, weights = 'uniform')
cross_val = cross_val_score(knn1, train_x, train_y,cv = 10)
print('Cross Val Score: {}%'.format(round(cross_val.mean()*100,2)))
