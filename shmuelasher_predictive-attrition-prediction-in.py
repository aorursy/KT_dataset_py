    #impoting relevant packages 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#impotring dataset
hr = pd.read_csv(r'../input/HR - Cleaned V2.csv')
#exploring the dataset
hr.info()
hr.head()
#Visualising null cells
sns.heatmap(hr.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#EDA
sns.set_style('whitegrid')
sns.countplot(x='Gender',data=hr)
sns.set_style('whitegrid')
sns.countplot(x='WorkingStatus',data=hr,palette='RdBu_r')
sns.distplot(hr['Tenure'],bins=20,hist=True,kde=True,color='green')
sns.set_style('whitegrid')
sns.countplot(x='WorkingStatus', hue='Gender',data=hr,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(y='RaceDesc',hue='Gender',data=hr,palette='rainbow')
sns.set_style('whitegrid')
sns.countplot(y='EmploymentStatus',hue='Gender',data=hr,palette='rainbow')
#Interactive visualisation
import cufflinks as cf
cf.go_offline()
hr['MaritalDesc'].iplot(kind='hist',bins=30,color='blue',title='Marital Description Distribution',width=2)
#Training a Linear Regression Mode

#Converting categorical features MaritalDesc & PerformanceScore into dummy variables

MaritalDescDum = pd.get_dummies(hr['MaritalDesc'],drop_first=True)
hr = hr.join(MaritalDescDum)

PerformanceScoreDum = pd.get_dummies(hr['PerformanceScore'],drop_first=True)
hr = hr.join(PerformanceScoreDum)
#X and y arrays
hr.columns
X=hr[['Age','GenderID','Tenure','EngagementSurvey','EmpSatisfaction','Married', 'Separated',
       'Single', 'Widowed', 'Fully Meets', 'Needs Improvement', 'PIP']]
y=hr[['WorkingStatus']]
#Building a Logistic Regression model
from sklearn.model_selection import train_test_split
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
#Training and Predicting
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
#Model evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
cf_matrix = confusion_matrix(y_test,predictions)
cf_matrix
#Visualising confusion matrix
names = ['True Positive','False Positive','False Negative','True Negative']
counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Oranges',annot_kws={"size": 12})