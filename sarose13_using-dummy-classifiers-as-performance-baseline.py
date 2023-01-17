# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\

import seaborn as sns

import matplotlib.pyplot as graph

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from IPython.display import display



graph.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HR_comma_sep.csv')

data.rename(columns={'sales': 'sales_string'}, inplace=True)

display(data.head())



# Quick Visualisation (Expense)

#sns.pairplot(data, hue='left')

#graph.show()

#

#sns.pairplot(data, hue='sales_string')

#graph.show()

#

#sns.pairplot(data, hue='salary')

#graph.show()



# Base Rate

print('Base Rate:', (data['left'] == 1).mean())



# Encode Text Features

salaries = data['salary'].unique().tolist()

salaries = dict(zip(salaries, range(len(salaries))))

data['salaries_encode'] = data['salary'].replace(salaries)



sales = data['sales_string'].unique().tolist()

sales = dict(zip(sales, range(len(sales))))

data['sales_encode'] = data['sales_string'].replace(sales)



# One Hot Class Encodings

salaries_onehot = OneHotEncoder(sparse=False).fit_transform(data['salaries_encode'].values.reshape(-1, 1))

print('Salaries Onehot', salaries_onehot.shape)

data = data.join(

    pd.DataFrame(salaries_onehot, columns=salaries.keys()),

    how='outer'

)



sales_onehot = OneHotEncoder(sparse=False).fit_transform(data['sales_encode'].values.reshape(-1, 1))

print('Sales Onehot', sales_onehot.shape)

data = data.join(

    pd.DataFrame(sales_onehot, columns=sales.keys()),

    how='outer'

)



# Drop None ML Columns

data.drop(['sales_string', 'salary'], axis='columns', inplace=True)

display(data.head(3))



# Standardise

y, x = data['left'].values, data.drop('left', axis='columns').values

x_std = StandardScaler().fit_transform(x)



# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.25)

print('Data Shape')

print(x.shape, y.shape)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# Simple Data Visualisation

data_pca = PCA(n_components=2, whiten=True).fit_transform(x_std)



# Satisfcation Level

graph.title('Left or not')

graph.scatter(data_pca[:, 0], data_pca[:, 1], c=y, cmap='inferno')

graph.show()



# Salaries

graph.title('Salaries')

graph.scatter(data_pca[:, 0], data_pca[:, 1], c=data['salaries_encode'], cmap='inferno')

graph.show()



# Sales

graph.title('Sales')

graph.scatter(data_pca[:, 0], data_pca[:, 1], c=data['sales_encode'], cmap='inferno')

graph.show()
# Grid Search CV for K-NN Classification Model

knn = GridSearchCV(

    KNeighborsClassifier(),

    scoring='roc_auc',

    param_grid={'n_neighbors': [3*(x+1) for x in range(25)]},

    cv=2,

    n_jobs=-1

)

knn.fit(x_train, y_train)



print(knn.best_score_)

print(knn.best_params_)
def judge_model(model, name, plot=False):

    print(name)

    print('-'*20)

    

    print('Training Performance')

    print('-> Acc:', accuracy_score(y_train, model.predict(x_train)) )

    print('-> AUC:', roc_auc_score(y_train, model.predict_proba(x_train)[:, 1] ))

    

    print('Testing Performance')

    print('-> Acc:', accuracy_score(y_train, model.predict(x_train)) )

    print('-> AUC:', roc_auc_score(y_test, model.predict_proba(x_test)[:, 1] ))

    print()

    

    if plot:

        fpr, tpr, thres = roc_curve(y_test, model.predict_proba(x_test)[:, 1])

        graph.figure(figsize=(4, 4))

        graph.plot(fpr, tpr, label='Test')

        graph.xlabel('FPR')

        graph.ylabel('TPR')

        graph.show()
# Baseline (AUC should be 0.5 because we're guessing even though the accuracies are different)

for strategy in ['stratified', 'most_frequent', 'prior', 'uniform']:

    dummy = DummyClassifier(strategy=strategy)

    dummy.fit(x_train, y_train)

    judge_model(dummy, 'Dummy {}'.format(strategy), plot=True)
judge_model(knn, 'K-NN', plot=True)