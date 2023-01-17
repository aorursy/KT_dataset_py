# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
filename_train = '../input/train.csv'
filename_test = '../input/test.csv'
file_train = pd.read_csv(filename_train)
file_test = pd.read_csv(filename_test)
file_train.describe()
file_test.describe()
file_train.head()
file_test.head()
file_train = file_train.dropna(axis=1)
file_train.describe()
file_train.columns
train_target = file_train.Cover_Type
train_feature = file_train.drop(['Cover_Type','Id'],axis=1)
train_feature.describe()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

train_X, validate_X, train_Y, validate_Y = train_test_split(train_feature,train_target, random_state = 0)

#定义函数评价MAE
#def randomforest_mae_split(train_X,validate_X,train_Y,validate_Y):
start = time.clock()
pp_randomforest = RandomForestClassifier()
pp_randomforest.fit(train_X,train_Y)
prediction_pp_rf = pp_randomforest.predict(validate_X)
pred = pd.DataFrame({'Cover_Type':prediction_pp_rf})
pred.head()
mae_rf = mean_absolute_error(validate_Y,prediction_pp_rf)
print('最大绝对误差为')
print(mae_rf)
print('预测分数(train):')
print(pp_randomforest.score(train_X,train_Y))
print('预测分数(validate):')
print(pp_randomforest.score(validate_X,validate_Y))
elapsed_rf = (time.clock() - start)
print("Time used:",elapsed_rf)
col_num = file_train.shape[1]
row_train_num = file_train.shape[0]
print(col_num)
print(row_train_num)
test_feature = file_test.drop('Id',axis=1)

start = time.clock()
predict_test = pp_randomforest.predict(test_feature)
elapsed_knn = (time.clock() - start)
print("Time used:",elapsed_knn)

start = time.clock()
from xgboost import XGBClassifier
pp_xgboost = XGBClassifier()
pp_xgboost.fit(train_X,train_Y)
prediction_pp_xg = pp_xgboost.predict(validate_X)
pred = pd.DataFrame({'Cover_Type':prediction_pp_xg})
pred.head()
mae_xg = mean_absolute_error(validate_Y,prediction_pp_xg)
print('最大绝对误差为')
print(mae_xg)
print('预测分数:')
print(pp_xgboost.score(validate_X,validate_Y))
elapsed_xg = (time.clock() - start)
print("Time used:",elapsed_xg)

learning_rate = np.linspace(0.05,0.1,num=10)
score_kn_train = []
score_kn_test = []
mae_kn = []
for LR in learning_rate:
    pp_xgboost = XGBClassifier(learning_rate=LR)
    pp_xgboost.fit(train_X,train_Y)
    prediction_pp_xg = pp_xgboost.predict(validate_X)
    mae_kn.append(mean_absolute_error(validate_Y,prediction_pp_xg))
    score_kn_train.append(pp_xgboost.score(train_X,train_Y))
    score_kn_test.append(pp_xgboost.score(validate_X,validate_Y))

pp_xgboost = XGBClassifier(learning_rate=0.1)
pp_xgboost.fit(train_X,train_Y)
prediction_pp_xg = pp_xgboost.predict(validate_X)
print('最大绝对误差为')
print(mean_absolute_error(validate_Y,prediction_pp_xg))
print('预测分数(train):')
print(pp_xgboost.score(train_X,train_Y))
print('预测分数(validate):')
print(pp_xgboost.score(validate_X,validate_Y))

start = time.clock()
predict_xgb_out = pp_xgboost.predict(test_feature)
elapsed_knn = (time.clock() - start)
print("Time used:",elapsed_knn)

from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score

def naivebayes_scores(clas_naivebayes,train_X,train_Y,validate_X,validate_Y,start):
    clas_naivebayes.fit(train_X,train_Y)
    prediction_nb = clas_naivebayes.predict(validate_X)
    mae_nb = mean_absolute_error(validate_Y,prediction_nb)
    print('最大绝对误差为 %.2f' %mae_nb)
    print('训练得分为: %.2f'%clas_naivebayes.score(train_X,train_Y))
    print('测试得分为: %.2f'%clas_naivebayes.score(validate_X,validate_Y))
    elapsed_xg2 = (time.clock() - start)
    print("Time used:",elapsed_xg2)
    pred_nb = pd.DataFrame({'Cover_Type':prediction_nb})
    pred_nb.head()
start = time.clock()
clas_naivebayes = naive_bayes.GaussianNB()
naivebayes_scores(clas_naivebayes,train_X,train_Y,validate_X,validate_Y,start)
start = time.clock()

clas_naivebayes.fit(train_X,train_Y)
pred_nb = clas_naivebayes.predict(validate_X)
print('最大绝对误差为')
print(mean_absolute_error(validate_Y,pred_nb))
print('预测分数(train):')
print(clas_naivebayes.score(train_X,train_Y))
print('预测分数(validate):')
print(clas_naivebayes.score(validate_X,validate_Y))


start = time.clock()
prediction_nb = clas_naivebayes.predict(test_feature)
elapsed_knn = (time.clock() - start)
print("Time used:",elapsed_knn)



from sklearn import neighbors
import matplotlib.pyplot as plt
start = time.clock()
clas_KNN = neighbors.KNeighborsClassifier()
naivebayes_scores(clas_KNN,train_X,train_Y,validate_X,validate_Y,start)
start = time.clock()
clas_KNN = neighbors.KNeighborsClassifier(n_neighbors=1,weights='distance')
clas_KNN.fit(train_X,train_Y)

prediction_knn = clas_KNN.predict(validate_X)
print('最大绝对误差为')
print(mean_absolute_error(validate_Y,prediction_knn))
print('预测分数(train):')
print(clas_KNN.score(train_X,train_Y))
print('预测分数(validate):')
print(clas_KNN.score(validate_X,validate_Y))

elapsed_knn = (time.clock() - start)
print("Time used:",elapsed_knn)
start = time.clock()
prediction_KNN = clas_KNN.predict(test_feature)
elapsed_knn = (time.clock() - start)
print("Time used:",elapsed_knn)
my_submission = pd.DataFrame({'Id': file_test.Id, 'Cover_Type': predict_test})
my_submission_xg = pd.DataFrame({'Id': file_test.Id, 'Cover_Type': predict_xgb_out })
my_submission_knn = pd.DataFrame({'Id': file_test.Id, 'Cover_Type': prediction_KNN  })
my_submission_nb = pd.DataFrame({'Id': file_test.Id, 'Cover_Type': prediction_nb  })
my_submission.to_csv('submission.csv', index=False)
my_submission.head()