# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd





pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
#データのロード

Dataset = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")



#データの表示

Dataset.head(50)
#データの大きさ

print(Dataset.shape)



#columns

print(Dataset.columns)

print(Dataset.dtypes)
Dataset.describe()
#列の種類を確認

print(Dataset['category'].unique())

print(Dataset['main_category'].unique())

print(Dataset['currency'].unique())
#欠損値の確認

print(Dataset.isnull().sum(axis = 0))
#相関を確認

Dataset.corr().style.background_gradient().format('{:.2f}')
'''

#散布図

pd.plotting.scatter_matrix(Dataset[['goal', 'pledged', 'backers', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']], figsize=(10,10))

plt.show()

'''
#終了前に得られるデータを選定

Dataset_x = Dataset[['category', 'main_category', 'currency', 'country', 'deadline', 'launched', 'goal']]

Dataset_x.head(10)
#onehot encording

import copy

onehot = copy.deepcopy(Dataset_x)



onehot_currency = pd.get_dummies(onehot['currency'])

onehot_country = pd.get_dummies(onehot['country'])

onehot_category = pd.get_dummies(onehot['main_category'])



#term

import datetime

datetime = copy.deepcopy(Dataset_x)

datetime_launched = pd.to_datetime(datetime['launched'], format='%Y-%m-%d %H:%M:%S')

datetime_deadline = pd.to_datetime(datetime['deadline'], format='%Y-%m-%d %H:%M:%S')

period = datetime_deadline - datetime_launched

period_df = pd.DataFrame(period.values / np.timedelta64(1, 's'), columns=['period'])



#standization

goal = copy.deepcopy(Dataset_x)

goal_df = pd.DataFrame(goal['goal'], columns=['goal'])

data_values = pd.concat([period_df, goal_df], axis=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

data_values_std = pd.DataFrame(sc.fit_transform(data_values), columns=['period','goal'])

data_values_std.head(10)
Dataset['state'].head()
#dataset

from sklearn.model_selection import train_test_split

X_data = pd.concat([data_values_std ,onehot_category, onehot_currency, onehot_country],axis=1)

X_data2 = data_values_std 

Y_data = Dataset['state'].values

Y_data = [1 if i == 'successful' else 0 for i in Y_data]

Y_data2 = Dataset[['usd_pledged_real', 'usd_goal_real']].values#そのまま

Y_data3 = pd.get_dummies(Dataset['state'])


#Classification

X1_train1, X1_test1, Y1_train1, Y1_test1 = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg = logreg.fit(X1_train1, Y1_train1)

print("LogisticRegression(Train) = ", round(logreg.score(X1_train1, Y1_train1) * 100, 2))

print("LogisticRegression(Test) = ", round(logreg.score(X1_test1, Y1_test1) * 100, 2))



X1_train2, X1_test2, Y1_train2, Y1_test2 = train_test_split(X_data2, Y_data, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X1_train2, Y1_train2)

print("Decision_tree(Train) = ", round(decision_tree.score(X1_train2, Y1_train2) * 100, 2))

print("Decision_tree(Test) = ", round(decision_tree.score(X1_test2, Y1_test2) * 100, 2))

'''



X1_train3, X1_test3, Y1_train3, Y1_test3 = train_test_split(X_data2, Y_data, test_size=0.2, random_state=0)

from sklearn.svm import SVC

svc = SVC()

svc.fit(X1_train3, Y1_train3)

print("svc(Train) = ", round(svc.score(X1_train3, Y1_train3) * 100, 2))

print("svc(Test) = ", round(svc.score(X1_test3, Y1_test3) * 100, 2))

'''
#Regression

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_data, Y_data2, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

Linreg = LinearRegression()

Linreg = Linreg.fit(X2_train, Y2_train[:, 0])

print("LinearRegression(Train) = ", round(Linreg.score(X2_train, Y2_train[:, 0]) * 100, 2))

print("LinearRegression(Test) = ", round(Linreg.score(X2_test, Y2_test[:, 0]) * 100, 2))
from sklearn.metrics import confusion_matrix

Y1_pre = logreg.predict(X1_test1)#Logistic Regression



#1=positive

#0=negative

C_mat = pd.DataFrame(confusion_matrix(Y1_test1, Y1_pre), 

                        index=['1', '0'], 

                        columns=['1', '0'])

print(C_mat)

#正解率

accuracy = (C_mat['1'][0]+C_mat['0'][1])/(C_mat['0'][0]+C_mat['0'][1]+C_mat['1'][0]+C_mat['1'][1])

print('Accuracy = ', accuracy)



#適合率

precision = C_mat['1'][0] / (C_mat['1'][0] + C_mat['1'][1]) 

print('Precision  = ', precision)



#再現率

recall = C_mat['1'][1]/(C_mat['1'][0]+C_mat['0'][0])

print('Recall = ', recall)
from sklearn.metrics import mean_squared_error, mean_absolute_error

y2_pre = Linreg.predict(X2_test)

MSE = mean_squared_error(Y2_test[:, 0], y2_pre)

print('MSE = ', MSE)

RMSE = np.sqrt(MSE)

print('RMSE = ', RMSE)

MAE = mean_absolute_error(Y2_test[:, 0], y2_pre)

print('MAE = ', MAE)
print("y2_pre = ", y2_pre)
print("Y2_test = ", Y2_test[:, 1])
expectation = y2_pre / Y2_test[:, 1] 

expectation = pd.DataFrame(expectation, columns=['expectaion'])

expectation.head(100)

print("expectation = ", expectation)

print("expectation.mean = ", np.mean(expectation))