import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
print(os.listdir("../input"))
data=pd.read_csv('../input/HR_comma_sep.csv')
data.head(5)
data.tail(5)
data.info()
data.shape
data.sample(10)
# now we have to change the sales to the department and the salary to the numerical values for the better results and better understanding.
data.rename(columns={'sales':'department'},inplace=True)

data['salary']=data['salary'].map({'low':1,'medium':2,'high':3})
data.head()
data.describe()
#How many employees works in each department?
print(data['department'].value_counts())
#How many employees per salary range?
print(data['salary'].value_counts())
#4.3 How many employees per salary range and department?
table=data.pivot_table(values='satisfaction_level',index='department',columns='salary',aggfunc=np.count_nonzero)
table
f, axes=plt.subplots(3,3, figsize=(10,10) , sharex=True) 

plt.subplots_adjust(wspace=0.5)
                     
sns.despine(left=True)
                    
sns.boxplot( x= 'satisfaction_level', data=data, orient='v',ax=axes[0,0])
sns.boxplot( x ='last_evaluation' , data=data, orient='v' , ax =axes[0,1])
sns.boxplot(x='number_project',data=data, orient='v' , ax =axes[1,0])
sns.boxplot(x='salary',data=data, orient='v' , ax =axes[1,1])
              

plt.figure(figsize=(4,5))
sns.boxplot( x= 'time_spend_company',  data=data, orient='v');
corr=data.corr()
corr
sns.set(style='white')

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Inserir a figura
f, ax = plt.subplots(figsize=(13,8))

cmap = sns.diverging_palette(10,220, as_cmap=True)

#Desenhar o heatmap com a máscara
ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax= .5, annot=True, annot_kws= {'size':11}, square=True, xticklabels=True, yticklabels=True, linewidths=.5, 
           cbar_kws={'shrink': .5}, ax=ax)
ax.set_title('Correlation between variables', fontsize=20);
#hypothesis


# How many employees left the company?

print(data['left'].value_counts(),)
print(data['left'].value_counts()[1],"employees left the company")
# the plot show the amount of employees that stayed and left the company.
plt.figure(figsize=(4,5))
ax=sns.countplot(data.left)
total=float(len(data))

for p in ax.patches:
    height= p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center"
           )
plt.title('Stayed or Left', fontsize=16)

#First Hypothesis The first hypothesis is that salary is the reason why the employees left the company. Let's see if is this correct.
j = sns.factorplot(x='salary', y='left', kind='bar', data=data)
plt.title('Employees that left by salary level', fontsize=14)
j.set_xticklabels(['High', 'Medium', 'Low']);
#In the graphic Salaries by department is possible to see the distribuition of the salaries by department.

h = sns.factorplot(x = 'salary', hue='department', kind ='count', size = 5,aspect=1.5, data=data, palette='Set1' )
plt.title("Salaries by department", fontsize=14)
h.set_xticklabels(['High', 'Medium', 'Low']);
#The first hypothesis looks very weak to be the main reason why the employees left the company.
sns.set()
plt.figure(figsize=(12,6))
sns.barplot(x='department',y='salary',hue='left', data=data)
plt.title('Salary Comaprison', fontsize=14);
#second hypothesis 
#The second hypothesis is: employees leave the company because work is not safe.

sns.factorplot(x='Work_accident',y='left',kind='bar', data=data)
plt.title('Employess that had work accident',fontsize=16);

print(data.Work_accident.sum())
print(data.Work_accident.mean())
print((data[data['left']==1]['Work_accident']).sum())      
#third hypothesis
#Is this company a good place to grow professionally?
sns.factorplot(x='promotion_last_5years', y='left', kind='bar', data=data)
plt.title('Employees who have been promoted in the last 5 years', fontsize=16);

print(data.promotion_last_5years.sum())
print(data.promotion_last_5years.mean())


plt.figure()
bins=np.linspace(1.0 , 11, 10)
plt.hist(data[data['left']==1]['time_spend_company'], bins=bins, alpha=1, label='Employees Left'
)
plt.hist(data[data['left']==0]['time_spend_company'], bins=bins, alpha = 0.5, label = 'Employee Stayed')

plt.grid(axis='x')
plt.xticks(np.arange(2,11))
plt.xlabel('time_spend_company')
plt.title('Years in the Company',fontsize=16)
plt.legend(loc='best');
plt.figure(figsize =(7,7))
bins = np.linspace(0.305, 1.0001, 14)
plt.hist(data[data['left']==1]['last_evaluation'], bins=bins, alpha=1, label='Employees Left')
plt.hist(data[data['left']==0]['last_evaluation'], bins=bins, alpha = 0.5, label = 'Employee Stayed')
plt.title('Employees Performance', fontsize=14)
plt.xlabel('last_evaluation')
plt.legend(loc='best');
poor_performance_left = data[(data.last_evaluation <= 0.62) & (data.number_project == 2) & (data.left == 1)]
print('poor_performance_left:',len(poor_performance_left))

poor_performance_stayed = data[(data.last_evaluation > 0.62) & (data.number_project == 2) & (data.left == 1)]
print('poor_performance_stayed:',len(poor_performance_stayed))

print('\n')

high_performance_left= data[(data.last_evaluation <= 0.62) & (data.number_project >=5) & (data.left == 1)]
high_performance_stayed= data[(data.last_evaluation > 0.8) & (data.number_project >=5) & (data.left == 0)]
print('high_performance_left:',len(high_performance_left))
print('high_performance_stayed', len(high_performance_stayed))

plt.figure(figsize =(7,5))
bins = np.linspace(1.5,7.5, 7)
plt.hist(data[data['left']==1]['number_project'], bins=bins, alpha=1, label='Employees Left')
plt.hist(data[data['left']==0]['number_project'], bins=bins, alpha = 0.5, label = 'Employee Stayed')
plt.title('Number of projects', fontsize=14)
plt.xlabel('number_ projects')
plt.legend(loc='best');
plt.figure(figsize =(7,5))
bins = np.linspace(80,315, 15)
plt.hist(data[data['left']==1]['average_montly_hours'], bins=bins, alpha=1, label='Employees Left')
plt.hist(data[data['left']==0]['average_montly_hours'], bins=bins, alpha = 0.5, label = 'Employee Stayed')
plt.title('Working Hours', fontsize=14)
plt.xlabel('average_montly_hours')
plt.xlim((70,365))
plt.legend(loc='best');
groupby_number_projects = data.groupby('number_project').mean()
groupby_number_projects = groupby_number_projects['average_montly_hours']
print(groupby_number_projects)
plt.figure(figsize=(7,5))
groupby_number_projects.plot();


work_less_hours_left = data[(data.average_montly_hours < 200) & (data.number_project == 2) & (data.left == 1)]
print('work_less_hours_left:',len(work_less_hours_left))

work_more_hours_left = data[(data.average_montly_hours > 240) & (data.number_project >=5 ) & (data.left == 1)]
print('work_more_hours_left:',len(work_more_hours_left))
plt.figure(figsize =(7,5))
bins = np.linspace(0.006,1.000, 15)
plt.hist(data[data['left']==1]['satisfaction_level'], bins=bins, alpha=1, label='Employees Left')
plt.hist(data[data['left']==0]['satisfaction_level'], bins=bins, alpha = 0.5, label = 'Employee Stayed')
plt.title('Employees Satisfaction', fontsize=14)
plt.xlabel('satisfaction_level')
plt.xlim((0,1.05))
plt.legend(loc='best');
groupby_time_spend = data.groupby('time_spend_company').mean()
groupby_time_spend['satisfaction_level']


sns.set()
sns.set_context("talk")
ax = sns.factorplot(x="number_project", y="satisfaction_level", col="time_spend_company",col_wrap=4, size=3, color='blue',sharex=False, data=data)
ax.set_xlabels('Number of Projects');
func_living = data[(data.last_evaluation >= 0.70) | (data.time_spend_company >=4) | (data.number_project >= 5)]

corr2 = func_living.corr()

sns.set(style='white')

mask = np.zeros_like(corr2, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Insert the graphic
f, ax = plt.subplots(figsize=(13,8))

cmap = sns.diverging_palette(10,220, as_cmap=True)

#Draw heat map mask
ax = sns.heatmap(corr2, mask=mask, cmap=cmap, vmax= .5, annot=True, annot_kws= {'size':11}, square=True, xticklabels=True, yticklabels=True, linewidths=.5, 
           cbar_kws={'shrink': .5}, ax=ax)
ax.set_title('Correlation: Why Valuable Employees Tend to Leave', fontsize=20);
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
index = data.index
columns = data.columns
values = data.values
print(columns)

y=data['left']

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc

print('Shape of x:',x.shape,' Shape of y:', y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)
print('Shape of x:',x_train.shape,' Shape of y:', y_train.shape)
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('report',classification_report(y_test,y_pred))
print('Accuracy',accuracy_score(y_test, y_pred))

fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ')
plt.show()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('report',classification_report(y_test,y_pred))
print('Accuracy',accuracy_score(y_test, y_pred))

fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ')
plt.show()
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('report',classification_report(y_test,y_pred))
print('Accuracy',accuracy_score(y_test, y_pred))

fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('report',classification_report(y_test,y_pred))
print('Accuracy',accuracy_score(y_test, y_pred))

fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ')
plt.show()
#SVM CLASSIFIER

from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train,y_train )
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False)
y_pred=clf.predict(x_test)
print(y_pred)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('report',classification_report(y_test,y_pred))
print('Accuracy',accuracy_score(y_test, y_pred))

fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ')
plt.show()
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=1000, learning_rate=0.2)
ada.fit(x_train, y_train)
y_pred=ada.predict(x_test)
print(y_pred)
results = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print('report',classification_report(y_test,y_pred))
print('Accuracy',accuracy_score(y_test, y_pred))

fpr,tpr,threshold=roc_curve(y_test,y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ')
plt.show()
