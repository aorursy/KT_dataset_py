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
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df = df.drop(['sl_no'], axis=1)
# df['status'][df['status']=='Placed']=1.0
# df['status'][df['status']=='Not Placed']=0.0
# df['gender'][df['gender']=='M']=1.0
# df['gender'][df['gender']=='F']=0.0
df = df.replace('Placed', 1)
df = df.replace('Not Placed', -1)
df = df.replace('M', 1.0)
df = df.replace('F', -1.0)
df = pd.get_dummies(df)
df.fillna(0.0, inplace=True)
df.head()
type(df['salary'][3])
df.isnull().sum() 
import matplotlib.pyplot as plt
import seaborn as sns
s = df.corr()
g = s['status'].sort_values(ascending=False)
col = g.index[(abs(g)>0.1) & (abs(g)<1)]
g = g[(abs(g)>0.1) & (abs(g)<1)]
plt.figure(figsize=(20,5))
plt.scatter(col,g)
plt.title('Correlation with status')
plt.figure(figsize=(20,5))
plt.bar(col,g,orientation ='vertical')
plt.title('Correlation with status')
g
import sklearn 
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(-1.0,1.0), copy=True)
y = df['status'].values
x_nomi = list(df.columns)
x_nomi.remove('status')
x = df[x_nomi].values
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import random, torch
from torch.autograd import Variable
from sklearn import svm

classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.LinearRegression()]

random.seed(1)
numero_split = 10
kf = KFold(n_splits=numero_split,random_state=1,shuffle=True)
mae = []
for c in classifiers:
    model1 = c
    contatore = 0
    for train_index, test_index in kf.split(x):
        contatore += 1
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        Scaler.fit(X_train)
        X_train, X_test = Scaler.transform(X_train),Scaler.transform(X_test)

        x_train = Variable(torch.Tensor(X_train))
        y_train = Variable(torch.Tensor(Y_train))
        x_test = Variable(torch.Tensor(X_test))
        y_test = Variable(torch.Tensor(Y_test))

        model1.fit(x_train, y_train)
        mae.append(accuracy_score(y_test, np.sign(model1.predict(x_test))))
        if contatore == numero_split:
            print('Classifiers {} has average test accuracy {}'.format(str(c),np.mean(mae)))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn 
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(-1.0,1.0), copy=True)
y = df['status'].values
x_nomi = list(df.columns)
x_nomi.remove('status')
x = df[x_nomi].values


def prev(classif):
    if classif == 'RF':
        model = RandomForestClassifier(random_state=0)
    elif classif == 'GB':
        model = GradientBoostingClassifier(n_estimators=100,max_depth=9,random_state=0)
    elif classif == 'TREE':
        model = DecisionTreeClassifier(random_state=0)
    nsplit = 10
    score = []
    conta = 0
    for train_index, test_index in kf.split(x):
        conta += 1
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        Scaler.fit(X_train)
        X_train, X_test = Scaler.transform(X_train),Scaler.transform(X_test)

        x_train = Variable(torch.Tensor(X_train))
        y_train = [i for i in Y_train]
        x_test = Variable(torch.Tensor(X_test))
        y_test = [i for i in Y_test]

        model.fit(x_train, y_train)
        score.append(model.score(x_test,y_test))
        if conta == nsplit:
            print('Average test Accuracy of {} is {}'.format(classif, np.mean(score)))
                  
prev('TREE')