import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
data = pd.read_csv("../input/HR_comma_sep.csv")
print(data.head())
data.columns
data.dtypes
data.describe()
pd.crosstab(data.salary, data.left).plot(kind='bar')
plt.show()
q = pd.crosstab(data.salary, data.left)
print(q)
print('\n')
print(q.sum(1))
q.div(q.sum(1), axis = 0).plot(kind='bar', stacked = True)
plt.show()
p1 = data[data.left==0].satisfaction_level.hist()
p2 = data[data.left==1].satisfaction_level.hist()
plt.show()
p = pd.crosstab(data.promotion_last_5years, data.left)
p.div(q.sum(0), axis = 0).plot(kind='bar', stacked = True)
plt.show()
from patsy import dmatrices  # dummy variable

y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
# X.head()
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
y = np.ravel(y)
model = LogisticRegression()

model.fit(X, y)
print(pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))))
print(model.score(X,y))
# use trained model to predict an example
model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])
pred = model.predict(X)
(abs(pred-y)).sum() / len(y)
Xtrain,Xtest,ytrain,ytest=train_test_split(X, y, test_size=0.2, random_state=0)
model2 = LogisticRegression()
model2.fit(Xtrain, ytrain)
model2 = LogisticRegression(C=10000)
model2.fit(Xtrain, ytrain)
pred = model2.predict(Xtest)
metrics.accuracy_score(ytest, pred)
#sklearn.metrics.classification_report(y_true, y_pred)
print(metrics.classification_report(ytest, pred))
print(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10))