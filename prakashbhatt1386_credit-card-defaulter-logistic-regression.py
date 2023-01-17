import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns',None)
import seaborn as sns
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
df_credit_card=pd.read_csv("../input/credit-card-defaulter-data/BankCreditCard (2).csv")
df_credit_card.head()
df_credit_card.columns
df_credit_card.info()
df_credit_card.drop('Customer ID',axis=1,inplace=True)
#EDA

#count each category in default payment variable

df_credit_card['Default_Payment'].value_counts()
import seaborn as sns
sns.countplot(x='Default_Payment',data=df_credit_card)
plt.show()
df_credit_card['Gender'].value_counts()
sns.countplot(x='Gender',data=df_credit_card)
plt.show()
df_credit_card['Academic_Qualification'].value_counts()
sns.countplot(x='Academic_Qualification',data=df_credit_card,saturation=0.50) # shift+tab to make changes in graph
plt.show()
df_credit_card['Marital'].value_counts()
sns.countplot(x='Marital',data=df_credit_card,saturation=0.50) # shift+tab to make changes in graph
plt.show()
# compare academic quali. with default payment

%matplotlib inline
pd.crosstab(df_credit_card.Academic_Qualification,df_credit_card.Default_Payment).plot(kind='bar')
plt.title('Default_Payment for Job Title')
plt.xlabel('Academic_Qualification')
plt.ylabel('Default_Payment')
#compare  academic quali. and marital

table=pd.crosstab(df_credit_card.Academic_Qualification,df_credit_card.Marital)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('stacked Bar chart of marital status vs Academic Qualification')
plt.xlabel('Academic Qualification')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack')

table.sum(0)
table.sum(1)
# correlation matrix
#calculating correlation among numeric variable

corr_matrix=df_credit_card.corr()

#plot correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,cmap='coolwarm',annot=True)
# spliting

x=df_credit_card.drop('Default_Payment',axis=1)
y=df_credit_card.loc[:,'Default_Payment']
# spliting input data into training and test

#import train and test module from sklearn
from sklearn.model_selection import train_test_split
#split train and test

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
#create the logistic regression model with SGD
from sklearn.linear_model import SGDClassifier
logreg_SGD=SGDClassifier(loss='log',max_iter=1000,early_stopping=True)
# training the model

logreg_SGD.fit(X_train,Y_train)
#prediciting the test set result and calculating the accuracy
pred_test=logreg_SGD.predict(X_test)

from sklearn.metrics import classification_report 
print(classification_report(Y_test,pred_test))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(Y_test,pred_test)
accuracy_score(Y_test,pred_test)
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg_SGD.score(X_test,Y_test)))
# cross validation
from sklearn import model_selection
#import cross validation score model from sklearn
from sklearn.model_selection import cross_val_score
#create model selection objext with number of splits
kfold=model_selection.KFold(n_splits=10,random_state=0)
#create a logistic regression model with SGD
modelCV=SGDClassifier(loss='log',tol=0.01,eta0=1.0,learning_rate='adaptive',max_iter=1000,early_stopping=True)
#call cross_val_score
result=model_selection.cross_val_score(modelCV,x,y,cv=10,scoring='accuracy')

print('10-fold cross validation average accuracy:%.3f'%(result.mean()))
print(result)
# confussion matrix
# import confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
# create  confusion matrix table
confusion_matrix=confusion_matrix(Y_test,pred_test)
print(confusion_matrix)
# import classification result from sklearn
from sklearn.metrics import classification_report
print(classification_report(Y_test,pred_test))
from sklearn.linear_model import SGDClassifier
from time import time
from sklearn.model_selection import GridSearchCV
logreg_SGD=SGDClassifier(loss='log')
param_grid
param_grid= { "n_iter_no_change":[1,5,10],
      'alpha':[0.0001,0.001,.01,.1,1,10,100],
      'tol':[0.0001,0.001,.01,.1,1],
      'eta0':[0.2,0.5,1.0,1.5,2.0,2.5,3.0],
     'learning_rate':['adaptive']}
#create grid search
grid_search=GridSearchCV(logreg_SGD,param_grid=param_grid)
grid_search.fit(X_train,Y_train)
#view the accuracy score
print('Best score for data1:',grid_search.best_score_)
#view the best parameter for the model found using grid search
print('Best C:',grid_search.best_estimator_.C)
print('Best alpha:',grid_search.best_estimator_.alpha)
print('Best n_iter:',grid_search.best_estimator_.n_iter_no_change)
print('Best tol:',grid_search.best_estimator_.tol)
print('Best eta:',grid_search.best_estimator_.eta0)
print('Best learning rate',grid_search.best_estimator_.learning_rate)
grid_search_train=grid_search.predict(X_test)
print(classification_report(grid_search_train,Y_test))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(grid_search_train,Y_test)
accuracy_score(grid_search_train,Y_test)
