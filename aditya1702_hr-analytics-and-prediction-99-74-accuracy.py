%matplotlib inline

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import mode,skew,skewtest



from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold,train_test_split
data=pd.read_csv('../input/HR_comma_sep.csv')
data.head(5)
data.isnull().sum()
data.shape
data.dtypes
data.describe()
#We first split the data into training and testing sets



train,test=train_test_split(data,train_size=0.95,test_size=0.05)

train.size,test.size
#We see that the avg monthly hours has a higher data amount and thus we will have to scale the data afterwards

#We run a basic random forest to determine the feature importances



forest=RandomForestClassifier(n_estimators=100,n_jobs=-1)

forest.fit(train.drop(['left','salary','sales'],1),train['left'])
importances=forest.feature_importances_

indices=np.argsort(importances)[::-1]

feature=train.drop(['left','salary','sales'],1).columns

plt.yticks(range(len(indices)),feature[indices],fontsize=10)

plt.barh(range(len(indices)),importances[indices])

#plt.tick_params(axis='x',)

#plt.tight_layout()
#As we see the above features have the respective importances.Satisfaction level has the highest importance as is clear

#logically.Lets examine each feature 



#satisfaction_level - we set a threshold of 0.5 for satisfaction level



def set_threshold(x):

    if x['satisfaction_level']>0.5:

        x['satisfy_new']=1.0

    else:

        x['satisfy_new']=0.0

    return x

train=train.apply(lambda x:set_threshold(x),1)
#We plot the variation of satisfaction level with the leaving of the employee



pd.crosstab(train['left'],train['satisfy_new']).plot(kind='bar',stacked=True)

plt.legend(['< 0.5','>=0.5'])
#Thus we see that of the people who have stayed here most have a higher satisfaction and hence it is important to our

#prediction



#number of projects



train.drop('satisfy_new',1,inplace=True)

train['number_project'].value_counts()
#For number of projects we check how the number of projects have had an impact on whether they have left or not

pd.crosstab(train['number_project'],train['left'],).plot(kind='bar',stacked=True)
#We see that those who did 3 projects have stayed and the percentage of people who are leaving increases with increasing

#projects.We create a new feature which calculates the number of project per time for which the person works



train['project/time']=train['number_project']/train['time_spend_company']

test['project/time']=test['number_project']/test['time_spend_company']

pd.crosstab(train['project/time'],train['left']).plot(kind='bar',stacked=True)
#We see that there are only binary values





train['Work_accident'].value_counts()
pd.crosstab(train['Work_accident'],train['left'],).plot(kind='bar',stacked=True)
#We see that in those who have had work accidents and those who havent there is no logical relationship with Left.



#Promotion last 5 years



pd.crosstab(train['promotion_last_5years'],train['left']).plot(kind='bar',stacked=True)
#We see that those who had a promotion in the last 5 years all stayed but a significant number of those didnt have a 

#promotion also stayed.Lets look at last evaluation



def set_threshold_for_evaluation(x):

    if x['last_evaluation']>0.5:

        x['eval_new']=1.0

    else:

        x['eval_new']=0.0

    return x

train=train.apply(lambda x:set_threshold_for_evaluation(x),1)
pd.crosstab(train['eval_new'],train['left']).plot(kind='bar',stacked=True)
pd.crosstab(train['left'],train['eval_new'])
#percentage of people who left and had evaluations greater than 0.5

percent1=9719.0/(2707+9719)



#percentage of people who left and had evaluations less 0.5

percent2=1138.0/(1138.0+685.0)



percent1,percent2
#We see that the evaluations did not have any significant effect on whether the people left or not



train.drop('eval_new',1,inplace=True)
#Sales



train['sales'].value_counts()
train.head(1)
sns.countplot('left',data=train,hue='sales')
pd.crosstab(train['sales'],train['left']).plot(kind='bar',stacked=True)
#We observe that people in sales department have higher percentage of those who have left, followed by support and tech

# -cal 



#Salary



train['salary'].value_counts()
pd.crosstab(train['left'],train['salary']).plot(kind='bar',stacked=True)
pd.crosstab(train['salary'],train['left']).plot(kind='bar',stacked=True)
#Time spent at the company



train['time_spend_company'].value_counts()
pd.crosstab(train['left'],train['time_spend_company']).plot(kind='bar',stacked=True)
pd.crosstab(train['time_spend_company'],train['left']).plot(kind='bar',stacked=True)
#We see that majority of people who have left are from those who spent little time at the company. Experienced and olde

#employees have all stayed. New ones have also stayed.



#Average Monthly Hours



train['average_montly_hours'].value_counts()
sns.distplot(train['average_montly_hours'])
#Assuming the time_spent_at company to be years we can calculate the number of months the person worked at the company.



train['num_months']=train['time_spend_company']*12

test['num_months']=test['time_spend_company']*12
lb=LabelEncoder()

train['sales_new']=lb.fit_transform(train['sales'])

train['salary_new']=lb.fit_transform(train['salary'])

train.drop(['sales','salary'],1,inplace=True)



test['sales_new']=lb.fit_transform(test['sales'])

test['salary_new']=lb.fit_transform(test['salary'])

test.drop(['sales','salary'],1,inplace=True)
train.head(1)
X_TRAIN,Y_TRAIN=train.drop('left',1),train['left']

x_test,y_test=test.drop('left',1),test['left']



train_,val=train_test_split(train,train_size=0.8)

x_train,y_train=train_.drop('left',1),train_['left']

x_val,y_val=val.drop('left',1),val['left']
#Logistic Regression



from sklearn import linear_model,metrics

logreg=linear_model.LogisticRegression(C=100.0,max_iter=500)

logreg.fit(x_train,y_train)

y_pred_val1=logreg.predict(x_val)

val_accuracy1=metrics.accuracy_score(y_pred_val1,y_val)



y_pred_test1=logreg.predict(x_test)

test_accuracy1=metrics.accuracy_score(y_pred_test1,y_test)

'validation accuracy= '+str(val_accuracy1)+'   '+'final accuracy= '+str(test_accuracy1)
#Random Forest Classifier



from sklearn import ensemble

forest=ensemble.RandomForestClassifier(n_estimators=100)

forest.fit(x_train,y_train)

y_pred_val2=forest.predict(x_val)

val_accuracy2=metrics.accuracy_score(y_pred_val2,y_val)



y_pred_test2=forest.predict(x_test)

test_accuracy2=metrics.accuracy_score(y_pred_test2,y_test)

'validation accuracy= '+str(val_accuracy2)+'   '+'final accuracy= '+str(test_accuracy2)
#Support Vector Machine



from sklearn import svm



sv=svm.SVC()

sv.fit(x_train,y_train)

y_pred_val3=sv.predict(x_val)

val_accuracy3=metrics.accuracy_score(y_pred_val3,y_val)



y_pred_test3=forest.predict(x_test)

test_accuracy3=metrics.accuracy_score(y_pred_test3,y_test)

'validation accuracy= '+str(val_accuracy3)+'   '+'final accuracy= '+str(test_accuracy3)
#Now we remove the 2 least important features



train2=train.drop(['promotion_last_5years','Work_accident'],1)

test2=test.drop(['promotion_last_5years','Work_accident'],1)
X_TRAIN2,Y_TRAIN2=train2.drop('left',1),train2['left']

x_test2,y_test2=test2.drop('left',1),test2['left']



train2_,val2=train_test_split(train2,train_size=0.8)

x_train2,y_train2=train2_.drop('left',1),train2_['left']

x_val2,y_val2=val2.drop('left',1),val2['left']
#Random Forest Classifier



from sklearn import ensemble

forest=ensemble.RandomForestClassifier(n_estimators=100)

forest.fit(x_train2,y_train2)

y_pred_val2=forest.predict(x_val2)

val_accuracy2=metrics.accuracy_score(y_pred_val2,y_val2)



y_pred_test2=forest.predict(x_test2)

test_accuracy2=metrics.accuracy_score(y_pred_test2,y_test2)

'validation accuracy= '+str(val_accuracy2)+'   '+'final accuracy= '+str(test_accuracy2)