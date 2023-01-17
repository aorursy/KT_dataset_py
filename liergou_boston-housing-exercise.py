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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
def notEmpty(s):
    return s != ''
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "../input/boston_housing.data"
## 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再按照每行数据进行处理
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)
x, y = np.split(data, (13,), axis=1)
y = y.ravel()
print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])
#数据的分割，
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.8, random_state=13)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))
#标准化
ss = MinMaxScaler()

x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

print ("原始数据各个特征属性的调整最小值:",ss.min_)
print ("原始数据各个特征属性的缩放数据值:",ss.scale_)

#构建模型（回归）
model = DecisionTreeRegressor(criterion='mse', max_depth=7)
#模型训练
model.fit(x_train, y_train)
#模型预测
y_test_hat = model.predict(x_test)

#评估模型
score = model.score(x_test, y_test)
print ("回归树Score：", score)

#构建线性回归
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_y_test_hat = lr.predict(x_test)
lr_score = lr.score(x_test, y_test)
print ("普通最小二乘线性回归lr-Score:", lr_score)
#构建lasso
lasso = LassoCV(alphas=np.logspace(-3,1,20))
lasso.fit(x_train, y_train)
lasso_y_test_hat = lasso.predict(x_test)
lasso_score = lasso.score(x_test, y_test)
print ("lasso:", lasso_score)
#构建岭回归
ridge = RidgeCV(alphas=np.logspace(-3,1,20))
ridge.fit(x_train, y_train)
ridge_y_test_hat = ridge.predict(x_test)
ridge_score = ridge.score(x_test, y_test)
print ("ridge:", ridge_score)

## 7. 画图
plt.figure(figsize=(12, 6), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'really price')
plt.plot(ln_x_test, lr_y_test_hat, 'b-', lw=2, label=u'LinearRegression，$R^2$=%.3f' % lr_score)
plt.plot(ln_x_test, lasso_y_test_hat, 'y-', lw=2, label=u'LassoRegression，$R^2$=%.3f' % lasso_score)
plt.plot(ln_x_test, ridge_y_test_hat, 'c-', lw=2, label=u'RidgeRegression，$R^2$=%.3f' % ridge_score)
plt.plot(ln_x_test, y_test_hat, 'g-', lw=4, label=u'Regression Decision，$R^2$=%.3f' % score)
plt.xlabel(u'code', fontsize=16)
plt.ylabel(u'price', fontsize=16)
plt.legend(loc = 'upper center')
plt.grid(True)
plt.title(u'boston housing price', fontsize=18)
plt.show()
# 参数优化
pipes = [
    Pipeline([
        ('mms', MinMaxScaler()),  ## 归一化操作
        ('pca', PCA()),  ## 降纬
        ('decision', DecisionTreeRegressor(criterion='mse'))
    ]),
    Pipeline([
        ('mms', MinMaxScaler()),
        ('decision', DecisionTreeRegressor(criterion='mse'))
    ]),
    Pipeline([
        ('decision', DecisionTreeRegressor(criterion='mse'))
    ])
]

# 参数
parameters = [
    {
        "pca__n_components": [0.25, 0.5, 0.75, 1],
        "decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
    },
    {
        "decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
    },
    {
        "decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
    }
]

# 获取数据
x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1

for t in range(3):
    pipe = pipes[t]

    gscv = GridSearchCV(pipe, param_grid=parameters[t])

    gscv.fit(x_train2, y_train2)

    print(t, "score值:", gscv.best_score_, "最优参数列表:", gscv.best_params_)
#使用最优参数看看正确率
mms_best = MinMaxScaler()
decision3 = DecisionTreeRegressor(criterion='mse', max_depth=5)
x_train3, x_test3, y_train3, y_test3 = x_train1, x_test1, y_train1, y_test1
x_train3 = mms_best.fit_transform(x_train3, y_train3)
x_test3 = mms_best.transform(x_test3)
decision3.fit(x_train3, y_train3)

print ("深度为5---最优参数看正确率:", decision3.score(x_test3, y_test3))
# 查看各个不同深度的错误率
x_train4, x_test4, y_train4, y_test4 = x_train1, x_test1, y_train1, y_test1

depths = np.arange(1, 20)
err_list = []
for d in depths:
    clf = DecisionTreeRegressor(criterion='mse', max_depth=d)
    clf.fit(x_train4, y_train4)

    score1 = clf.score(x_test4, y_test4)
    err = 1 - score1
    err_list.append(err)
    print("%d深度，正确率%.5f" % (d, score1))

## 画图
plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'decision deep', fontsize=16)
plt.ylabel(u'error rare', fontsize=16)
plt.grid(True)
plt.title(u'deep vs fit', fontsize=18)
plt.show()
#构建模型（回归）
model = DecisionTreeRegressor(criterion='mse',max_depth=5, random_state=8)
#模型训练
model.fit(x_train, y_train)
#模型预测
y_test_hat = model.predict(x_test)

#评估模型
score = model.score(x_test, y_test)
print ("Score：", score)














