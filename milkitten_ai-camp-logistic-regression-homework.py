import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve
data = pd.read_csv('../input/HR_comma_sep.csv')

print(data.dtypes)

print(data.shape)

print(data.columns)

data.head()

pd.crosstab(data.salary, data.left).plot(kind='bar')

plt.show()
cross = pd.crosstab(data.salary, data.left)

cross.div(cross.sum(axis=1), axis = 0).plot.bar(stacked='True')

plt.ylabel('%')

plt.xlabel('salary')

plt.show()
data.satisfaction_level[data.left==0].plot.hist(alpha = 0.5, label = 'stay')

data.satisfaction_level[data.left==1].plot.hist(alpha = 0.5, label = 'left')

plt.legend()

plt.show()
model = LogisticRegression()

y,X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type = 'dataframe')

print(y.head())

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
model.fit(X,y)

print(model.coef_.shape)

pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

# pd.DataFrame({'columns':X.columns, 'coef': np.ravel(model.coef_)})
accu = model.score(X,y)

print(accu)
model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])
pred_prob = model.predict_proba(X)

pred = model.predict(X)

miss = (abs(pred-y)).sum() / len(y)

print(miss)

accu + miss
# threshold = np.linspace(0,1,11)

# print(threshold)

# for i in threshold:

#     preds = np.where(model.predict_proba(X)[:,1] > i, 1, 0) 

    

# pcs = [0.24, 0.34, 0.44, 0.53, 0.59, 0.61, 0.6, 0.62, 0.16,0]

# pcs = np.array(pcs)

# rc = [1, 0.96, 0.81, 0.69, 0.53, 0.36, 0.21, 0.12, 0.01,0] 

# rc = np.array(rc)

# plt.scatter(pcs, rc)

# plt.show()

# f1 = [0.38, 0.51, 0.57, 0.6, 0.56, 0.45, 0.31, 0.2, 0.01, 0]

precision, recall, thresholds = precision_recall_curve(y,pred_prob[:,1])

plt.scatter(precision, recall, s = 1)

plt.show()

print(thresholds)

accu_vec = []

f1 = 2/(1/precision + 1/recall)

plt.plot(thresholds, f1[1:])

for i in thresholds:

    preds = np.where(model.predict_proba(X)[:,1] > i, 1, 0) 

    accu_vec.append(accuracy_score(y,preds))

plt.plot(thresholds, f1[1:], label = 'f1')

plt.plot(thresholds, accu_vec, label = 'accuracy')

plt.legend() 

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0, test_size=0.3)
model2 = LogisticRegression(C=10000)

model2.fit(Xtrain, ytrain)

pred = model2.predict(Xtest)

metrics.accuracy_score(ytest, pred)


print(confusion_matrix(ytest, pred))

print(classification_report(ytest, pred))
print(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10))