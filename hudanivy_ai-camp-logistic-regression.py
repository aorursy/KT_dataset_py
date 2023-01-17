import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import math 
from patsy import dmatrices 

data = pd.read_csv("../input/HR_comma_sep.csv")
data.head()
data.corr()['left']
tmp=pd.crosstab(data.time_spend_company,data.left)
tmp.div(tmp.sum(1),axis=0).plot(kind='bar',stacked=True)

tmp=pd.crosstab(data.number_project,data.left)
tmp.div(tmp.sum(1),axis=0).plot(kind='bar',stacked=True)
#把average monthly hours变成离散的bin, 以便分析monthly hours跟left的关系
data['avg_hours_level'] = pd.qcut( data['average_montly_hours'], 15 , labels=range(15))
tmp=pd.crosstab(data.avg_hours_level,data.left)
tmp.div(tmp.sum(1),axis=0).plot(kind='bar',stacked=True)
#把average monthly hours变成离散的bin, 以便分析monthly hours跟left的关系
data['satisfaction_level_discrete'] = pd.qcut( data['satisfaction_level'], 15 , labels=range(15))
tmp=pd.crosstab(data.avg_hours_level,data.left)
tmp.div(tmp.sum(1),axis=0).plot(kind='bar',stacked=True)
data['last_evaluation_level'] = pd.qcut( data['last_evaluation'], 15 , labels=range(15))
tmp=pd.crosstab(data.last_evaluation_level,data.left)
tmp.div(tmp.sum(1),axis=0).plot(kind='bar',stacked=True)
tmp=pd.crosstab(data.time_spend_company,data.left)
tmp.div(tmp.sum(1),axis=0).plot(kind='bar',stacked=True)
Y, X = dmatrices('left~C(satisfaction_level_discrete)+C(number_project)+C(avg_hours_level)+C(time_spend_company)+C(last_evaluation_level)+Work_accident+promotion_last_5years+C(salary)+C(sales)', data, return_type='dataframe')
Y=np.ravel(Y)
X=np.asmatrix(X)
xtrain, xvali, ytrain, yvali = train_test_split(X, Y, test_size=0.3, random_state=0)
#xtrain = np.asmatrix(xtrain)
#xvali = np.asmatrix(xvali)
#ytrain = np.ravel(ytrain)
#yvali = np.ravel(yvali)

model = LogisticRegression()
model.fit(xtrain, ytrain)

pred = model.predict(xtrain)
print("model.score on train set: ",model.score(xtrain,ytrain))


pred_vali=model.predict(xvali)
print(metrics.accuracy_score(yvali, pred_vali))
print(metrics.confusion_matrix(yvali, pred_vali))
print(metrics.classification_report(yvali, pred_vali))
print(cross_val_score(LogisticRegression(), X, Y, scoring='accuracy', cv=10))
#结论：有很多列是连续变量，也跟跳槽结果有关系，但是不是简单的线性关系，
#所以把它们变成离散的bin,然后做成dummy variable来处理。最后的结果有很大的提高。