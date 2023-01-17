import numpy as np 
import pandas as pd 
from patsy import dmatrices   # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("../input/HR_comma_sep.csv")
data
data.dtypes
pd.crosstab(data.salary, data.left).plot(kind='bar')
plt.show()
q = pd.crosstab(data.salary, data.left)
print(q)
print(q.sum(1))
q.div(q.sum(1), axis = 0).plot(kind='bar', stacked = True)
plt.show()
data[data.left==0].satisfaction_level.hist()
plt.show()
data[data.left==1].satisfaction_level.hist()
plt.show()
model = LogisticRegression()
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
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
model.fit(X, y)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

print(model.score(X,y))
#预测这样一个人的离职概率： 一个高工资HR， 对公司满意度0.5, 上次评审0.7分， 做过4个项目， 每月平均工作160小时， 在公司呆了3年， 过去5年没有被晋升， 没有工伤
model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])
pred = model.predict(X)
(abs(pred-y)).sum() / len(y)
Xtrain,Xtest,ytrain,ytest=train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(Xtrain, ytrain)
model2 = LogisticRegression(C=10000)
model2.fit(Xtrain, ytrain)
pred = model2.predict(Xtest)
metrics.accuracy_score(ytest, pred)
metrics.confusion_matrix(ytest, pred)
#sklearn.metrics.classification_report(y_true, y_pred)
print(metrics.classification_report(ytest, pred))

print(cross_val_score(LogisticRegression(C=1e5), X, y, scoring='accuracy', cv=10).mean())
# when C=1e5 R = 1/C
R = 1e-5
R
print(cross_val_score(LogisticRegression(C=1e10), X, y, scoring='accuracy', cv=10).mean())
print(cross_val_score(LogisticRegression(C=1), X, y, scoring='accuracy', cv=10).mean())
print(cross_val_score(LogisticRegression(C=0.001), X, y, scoring='accuracy', cv=10).mean())  # cv=10 分十份
cross_val_score(LogisticRegression(C=0.01), X, y, scoring='accuracy', cv=10)
