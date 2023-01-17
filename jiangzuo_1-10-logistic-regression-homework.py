import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库
data.left = data.left.astype(int)
观察员工满意度的分布图(histogram)
X = X.rename(columns = {

    'C(sales)[T.RandD]': 'Department: Random',

    'C(sales)[T.accounting]': 'Department: Accounting',

    'C(sales)[T.hr]': 'Department: HR',

    'C(sales)[T.management]': 'Department: Management',

    'C(sales)[T.marketing]': 'Department: Marketing',

    'C(sales)[T.product_mng]': 'Department: Product_Management',

    'C(sales)[T.sales]': 'Department: Sales',

    'C(sales)[T.support]': 'Department: Support',

    'C(sales)[T.technical]': 'Department: Technical',

    'C(salary)[T.low]': 'Salary: Low',

    'C(salary)[T.medium]': 'Salary: Medium'}) 

y = np.ravel(y) # 将y变成np的一维数组
print(model.score(X,y))

model.coef_
model.predict_proba(X)

pred = model.predict(X)

(abs(pred-y)).sum() / len(y)
model2 = LogisticRegression(C=10000)

model2.fit(Xtrain, ytrain)

pred = model2.predict(Xtest)

metrics.accuracy_score(ytest, pred)

print(metrics.classification_report(ytest, pred))

print(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10))