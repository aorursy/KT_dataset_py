# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import statsmodels.api as sm
from scipy import stats
stats.chisqprob=lambda chisq,df:stats.chi2.sf(chisq,df)
raw_data=pd.read_csv('../input/titanic/train.csv')
raw_test_data=pd.read_csv('../input/titanic/test.csv')
raw_data.tail()
raw_data.isnull().sum()
raw_test_data.isnull().sum()
raw_data2=raw_data.drop(['Cabin'],axis=1)
raw_test_data2=raw_test_data.drop(['Cabin'],axis=1)
raw_data2.tail()
raw_data2['Survived'].value_counts()
sns.countplot(raw_data2['Survived'],palette='RdBu_r')
plt.title('Survived and Deceased',fontsize=15)
table=pd.crosstab(raw_data2.Sex,raw_data2.Survived)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
sns.countplot(x='Survived',hue='Pclass',data=raw_data2,palette='winter')
sns.jointplot(x=raw_data2['Survived'],y=raw_data2['Age'],kind='kde')
plt.figure(figsize=(15,4))
raw_data2.Age.hist(bins=20,color='darkred',alpha=0.4)
plt.figure(figsize=(15,8))
sns.boxplot(x=raw_data2['Pclass'],y=raw_data2['Age'],hue=raw_data2['Survived'],palette='mako_r')
    
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
        

raw_data2['Age'] = raw_data2[['Age','Pclass']].apply(impute_age,axis=1)
raw_test_data2['Age'] = raw_test_data2[['Age','Pclass']].apply(impute_age,axis=1)
raw_data2.isnull().sum()
raw_test_wn=raw_test_data2.fillna(raw_test_data2['Fare'].median())
raw_test_wn=raw_test_data2.dropna(how='any')
raw_test_wn.isnull().sum()
raw_data2['Embarked']=raw_data2['Embarked'].fillna('S')
raw_data2['Sex'] = raw_data2['Sex'].map({'male': 0,'female': 1})
raw_test_wn['Sex'] = raw_test_wn['Sex'].map({'male': 0,'female': 1})
raw_test_wn.head()
raw_data3=raw_data2.drop(['Name','Ticket'],axis=1)
raw_test_3=raw_test_wn.drop(['Name','Ticket'],axis=1)
data_final=pd.get_dummies(raw_data3)
data_final
TEST=pd.get_dummies(raw_test_3)
x=data_final.drop(['Survived'],axis=1)
y=data_final['Survived']
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logreg=LogisticRegression()
rfe=RFE(logreg,20)
rfe=rfe.fit(x,y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
results=sm.Logit(y,x).fit()
results.summary()
final=x.drop(['PassengerId','Parch','Fare'],axis=1)
X=final
X
Final_test=TEST.drop(['PassengerId','Parch','Fare'],axis=1)
Final_test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_predict=logreg.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm=confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True)
accuracy_score(y_test,y_predict)*100
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
from sklearn.tree import DecisionTreeClassifier
tree_class=DecisionTreeClassifier(max_leaf_nodes=70000,random_state=0)
tree_class.fit(x_train,y_train)
Y_pred_tree=tree_class.predict(x_test)
cm_tree=confusion_matrix(Y_pred_tree,y_test)
sns.heatmap(cm_tree,annot=True)
accuracy_score(y_test,Y_pred_tree)*100
print(classification_report(y_test,Y_pred_tree))
from sklearn.ensemble import RandomForestClassifier
classifier_forest=RandomForestClassifier()
classifier_forest.fit(x_train,y_train)
pred_forest=classifier_forest.predict(x_test)
cm_forest=confusion_matrix(y_test,pred_forest)
sns.heatmap(cm_forest,annot=True)
accuracy_score(y_test,pred_forest)*100
print(classification_report(y_test,pred_forest))
from sklearn.naive_bayes import GaussianNB
classifier_nb=GaussianNB()
classifier_nb.fit(x_train,y_train)
pred_nb=classifier_nb.predict(x_test)
cm_nb=confusion_matrix(y_test,pred_nb)
sns.heatmap(cm_nb,annot=True)
accuracy_score(y_test,pred_nb)*100

from sklearn.svm import SVC
classifier_svc=SVC(kernel='linear',random_state=0)
classifier_svc.fit(x_train,y_train)
pred_svc=classifier_svc.predict(x_test)
cm_svc=confusion_matrix(y_test,pred_svc)
sns.heatmap(cm_svc,annot=True)
accuracy_score(y_test,pred_svc)*100

classifier_randomforest=RandomForestClassifier(random_state=0)
classifier_randomforest.fit(X,y)
final_result=classifier_randomforest.predict(Final_test)
Submission=pd.DataFrame()
Submission['PassengerID']=TEST['PassengerId'].values
Submission['Survived']=final_result
Submission.to_csv(r'C:\Users\user\Music\Projects\classification prob (logistic regression)\Titanic\submit.csv')
submit=pd.read_csv(r'C:\Users\user\Music\Projects\classification prob (logistic regression)\Titanic\submit.csv',index_col=0)
submit
