import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv("../input/HR_comma_sep.csv")

data.head()
data.dtypes
# 觀察薪資比例

cross_salary_by_left = pd.crosstab(data.left, data.salary)

cross_salary_by_left.plot(kind = 'bar', stacked = True)

plt.show()

sum_of_left = cross_salary_by_left.sum(1)

percentage = cross_salary_by_left.div(sum_of_left, axis = 0)

percentage.plot(kind = 'bar', stacked = True)

plt.show()

# 觀察離職比例

cross_left_by_salary = pd.crosstab(data.salary, data.left)

cross_left_by_salary.plot(kind = 'bar', stacked = True)

plt.show()

sum_of_each_salary = cross_left_by_salary.sum(1)

percentage = cross_left_by_salary.div(sum_of_each_salary, axis = 0)

percentage.plot(kind = 'bar', stacked = True)

plt.show()
data['satisfaction_level'].hist()

plt.show()

data[data['left'] == 1].satisfaction_level.hist()

plt.show()

data[data['left'] == 0].satisfaction_level.hist()

plt.show()
data[data.left==1].satisfaction_level.hist()

plt.show()
# Collect features as dmatrice parameter

def feature_string(columns, label, categorical_vars):

    features = []

    for feature in columns:

        if feature == label:

            continue

        if feature in categorical_vars:

            features.append("C(" + feature + ")")

        else:

            features.append(feature)

            

    return 'left~' + "+".join(features)





label_features = feature_string(data.columns, 'left', ['sales', 'salary'])

y, X = dmatrices(label_features, data, return_type='dataframe')

X.head()
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
model = LogisticRegression(solver = 'liblinear')

model.fit(X, y)

pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

print("Score:", model.score(X,y))

pred = model.predict(X)

print("Error:", (abs(pred-y)).sum() / len(y))

#一个高工资HR，对公司满意度0.5, 上次评审0.7分，做过4个项目，每月平均工作160小时，在公司呆了3年，过去5年没有被晋升，没有工伤

# model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])
Xtrain,Xtest,ytrain,ytest=train_test_split(X, y, test_size=0.3, random_state=0)

model2 = LogisticRegression(solver = 'liblinear')

model2.fit(Xtrain, ytrain)

pred = model2.predict(Xtest)

metrics.accuracy_score(ytest, pred)
# Add Regulation C

model2 = LogisticRegression(C=10000, solver = 'liblinear')

model2.fit(Xtrain, ytrain)

pred = model2.predict(Xtest)

metrics.accuracy_score(ytest, pred)
metrics.confusion_matrix(ytest, pred)
print(metrics.classification_report(ytest, pred))

print(cross_val_score(LogisticRegression(solver = 'liblinear'), X, y, scoring='accuracy', cv=10))