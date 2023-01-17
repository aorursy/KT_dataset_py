import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
data.describe()
data.dtypes
pd.crosstab(data.salary, data.left).plot(kind='bar')

plt.show()
q = pd.crosstab(data.salary, data.left)

print(q,'\n')

print(q.sum(1)) # axis=0是按照index加起来，axis=1按照column加起来
ratio = q.div(q.sum(1), axis=0)

print(ratio)

ax = ratio.plot(kind='bar',figsize=(8,4),stacked='True')

plt.legend(labels = ['left','remain'])



for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.yticks([])



# Add this loop to add the annotations

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x, y + height + 0.01))



plt.title('Ratio of left and remaining employees in different salary levels')

plt.show()
plt.hist(data[data.left==0].satisfaction_level,facecolor='blue', edgecolor='white', alpha=0.5)

plt.xlabel('satisfaction level')

plt.ylabel('num of people')

plt.title('distribution of satisfaction level of left employees')

plt.show()
plt.hist(data[data.left==1].satisfaction_level,facecolor='blue', edgecolor='white', alpha=0.5)

plt.xlabel('satisfaction level')

plt.ylabel('num of people')

plt.title('distribution of satisfaction level of remaining employees')

plt.show()
promote = pd.crosstab(data.promotion_last_5years, data.left)

promote.index = ['not promoted', 'promoted']

print(promote)

promote.plot(kind='bar')

plt.legend()

plt.show()
plt.hist(data[data.left==0].average_montly_hours,facecolor='red', edgecolor='white', alpha=0.5, label='remain')

plt.hist(data[data.left==1].average_montly_hours,facecolor='blue', edgecolor='white', alpha=0.5, label='left')

plt.xlabel('average_montly_hours')

plt.ylabel('num of people')

plt.title('distribution of average_montly_hours of employees')

plt.legend()

plt.show()
plt.hist(data[data.left==0].number_project,facecolor='red', edgecolor='white', width=0.4, alpha=0.5, label='remain')

plt.hist(data[data.left==1].number_project,facecolor='blue', edgecolor='white', width=0.4, alpha=0.5, label='left')

plt.xlabel('number_project')

plt.ylabel('num of people')

plt.title('distribution of number_project of employees')

plt.legend()

plt.show()
plt.hist(data[data.left==0].time_spend_company,facecolor='red', edgecolor='white', alpha=0.5, label='remain')

plt.hist(data[data.left==1].time_spend_company,facecolor='blue', edgecolor='white', width=0.8, alpha=0.5, label='left')

plt.xlabel('time_spend_company')

plt.ylabel('num of people')

plt.title('distribution of time_spend_company of employees')

plt.legend()

plt.show()
data.average_montly_hours.corr(data.time_spend_company)
model = LogisticRegression()



y,X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
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

pd.DataFrame(list(zip(X.columns,np.transpose(model.coef_))))
print(model.score(X,y))

model.coef_
X.columns
model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0,0.5,0.7,4,160,3,0,0]])
model.predict_proba(X)

pred = model.predict(X)

(abs(pred-y)).sum() / len(y)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

model2 = LogisticRegression()

model2.fit(Xtrain,ytrain)
model2 = LogisticRegression(C=10000)

model2.fit(Xtrain, ytrain)

pred = model2.predict(Xtest)

metrics.accuracy_score(ytest, pred)
metrics.confusion_matrix(ytest,pred)
print(metrics.classification_report(ytest, pred))
print(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10))