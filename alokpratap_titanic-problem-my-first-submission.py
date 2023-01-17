# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
train_data.head()

# Any results you write to the current directory are saved as output.
test_data.head() #for displaying first five datas
train_data.info()  # gives information about all the columns, about its data types and no of observation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
train_data.describe()# Gives the descriptive statistics of all the columns
test_data.describe()
sns.countplot(x='Survived', data=train_data);
train_data.Survived.sum()
train_data.Survived.count()
print(train_data.Survived.sum()/train_data.Survived.count())
train_data.groupby(['Survived','Sex'])['Survived'].count()
sns.catplot(x='Sex', col='Survived', kind='count', data=train_data);
print("percentage of women survived: " ,train_data[train_data.Sex == 'female'].Survived.sum()/train_data[train_data.Sex == 'female'].Survived.count())
print("percentage of men survived:   " , train_data[train_data.Sex == 'male'].Survived.sum()/train_data[train_data.Sex == 'male'].Survived.count())
pd.crosstab(train_data.Pclass, train_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')
print("% of survivals in") 
print("Pclass=1 : ", train_data.Survived[train_data.Pclass == 1].sum()/train_data[train_data.Pclass == 1].Survived.count())
print("Pclass=2 : ", train_data.Survived[train_data.Pclass == 2].sum()/train_data[train_data.Pclass == 2].Survived.count())
print("Pclass=3 : ", train_data.Survived[train_data.Pclass == 3].sum()/train_data[train_data.Pclass == 3].Survived.count())
pd.crosstab([train_data.Sex, train_data.Survived], train_data.Pclass, margins=True).style.background_gradient(cmap='autumn_r')

pd.crosstab([train_data.Survived], [train_data.Sex, train_data.Pclass, train_data.Embarked], margins=True)
for df in [train_data, test_data]:
    df['Age_bin']=np.nan
    for i in range(8,0,-1):
        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i
print(train_data[['Age' , 'Age_bin']].head(10))
test_data['Survived'] = 0
test_data.loc[ (test_data.Sex == 'female'), 'Survived'] = 1
test_data.loc[ (test_data.Sex == 'female') & (test_data.Pclass == 3) & (test_data.Embarked == 'S') , 'Survived'] = 0

sns.distplot(train_data['Fare'])
plt.show()
for df in [train_data, test_data]:
    df['Fare_bin']=np.nan
    for i in range(12,0,-1):
        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i
sns.catplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=train_data)
plt.show()
pd.crosstab([train_data.Sex, train_data.Survived], [train_data.Fare_bin, train_data.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
pd.crosstab([train_data.Sex, train_data.Survived], [train_data.Age_bin, train_data.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
pd.crosstab([train_data.Sex, train_data.Survived], [train_data.SibSp, train_data.Pclass], margins=True).style.background_gradient(cmap='autumn_r')
test_data.loc[ (test_data.Sex == 'female') & (test_data.SibSp > 7) , 'Survived'] = 0
sns.catplot('Parch','Survived', col='Pclass' , row = 'Sex', kind='point', data=train_data)
plt.show()

test_data.loc[ (test_data.Sex == 'female') & (test_data.SibSp > 7) , 'Survived'] = 0

test_data.drop(['Survived'],axis=1,inplace=True)
train1 = train_data.copy()
test1 = test_data.copy()
train1 = pd.get_dummies(train1, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

train1.head()
train1.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)
train1.head()
train1.dropna(inplace=True)

train1.info()
passenger_id = test1['PassengerId']
test1 = pd.get_dummies(test1, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
test1.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)
test1.head()
test1.info()
correlation = train1.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(correlation, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train1.drop('Survived',axis=1))
scaled_features = scaler.transform(train1.drop('Survived',axis=1))
train1.sc = pd.DataFrame(scaled_features, columns=train1.columns[:-1])
test1.info()
test1.fillna(test1.mean(), inplace=True)
scaled_features = scaler.transform(test1)
test1.sc = pd.DataFrame(scaled_features, columns=test1.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train1.drop('Survived',axis=1), train1['Survived'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train1.sc,train1['Survived'], test_size=0.30, random_state=101)  

X_train_all = train1.drop('Survived',axis=1)
y_train_all = train1['Survived']
X_test_all = test1

X_test_all.fillna(X_test_all.mean(), inplace=True)

X_test_all.head()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
pred_logreg
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))
logreg.fit(X_train_all, y_train_all) #Train for all data
pred_all_logreg = logreg.predict(X_test_all)
sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = test_data['PassengerId']
sub_logreg['Survived'] = pred_all_logreg
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)
pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))
knn.fit(X_train_all, y_train_all)
pred_all_knn = knn.predict(X_test_all)

sub_knn = pd.DataFrame()
sub_knn['PassengerId'] = test_data['PassengerId']
sub_knn['Survived'] = pred_all_knn
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6, max_features=7)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test, pred_rfc))