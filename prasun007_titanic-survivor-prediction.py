import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
test.head()
print(train.shape, test.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
sns.countplot(x='Survived', data=train).set_title('Count plot for survived.')
sns.factorplot('Sex', hue='Survived', kind='count', data=train);
plt.title('Factor plot for male and female')
print("% of women survived: " , train[train.Sex == 'female']['Survived'].sum()/train[train.Sex == 'female']['Survived'].count())
print("% of men survived:   " , train[train.Sex == 'male']['Survived'].sum()/train[train.Sex == 'male']['Survived'].count())
f,ax = plt.subplots(1,2,figsize=(20,7))
train['Survived'][train['Sex'] == 'male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
train['Survived'][train['Sex'] == 'female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)
survived=train[train['Survived']==1]['Embarked'].value_counts()
dead=train[train['Survived']==0]['Embarked'].value_counts()
df=pd.DataFrame([survived,dead])
df.index=['survived','dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
survived=train[train['Survived']==1]['Pclass'].value_counts()
dead=train[train['Survived']==0]['Pclass'].value_counts()
df=pd.DataFrame([survived,dead])
df.index=['survived','dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
sns.pairplot(train[["Fare","Age","Pclass","Survived"]],vars = ["Fare","Age","Pclass"],hue="Survived", dropna=True,markers=["o", "s"])
plt.title('Pair Plot')
corr = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,vmax=.8,linewidth=.01, square = True, annot = True,cmap='YlGnBu',linecolor ='black')
plt.title('Correlation between features')
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train['Embarked'].fillna(value='S',inplace=True)
train['family']=train['SibSp']+train['Parch']+1
test['family']=test['SibSp']+test['Parch']+1
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])
train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[1,2,3])
test['Sex'] = test['Sex'].replace(['female','male'],[0,1])
test['Embarked'] = test['Embarked'].replace(['S','Q','C'],[1,2,3])
clean_train=train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
clean_test=test.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
X_train=clean_train.drop(columns=['Survived'])
Y_train=clean_train[['Survived']]
X_test = clean_test
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 0)
logreg.fit(X_train, Y_train)
Y_pred_train = logreg.predict(X_train)
acc_logreg = round(accuracy_score(Y_train,Y_pred_train)*100,2)
print('Accuracy score for train data ', acc_logreg)
print('confusion matrix for train data \n',  confusion_matrix(Y_train, Y_pred_train))
Y_pred_test = logreg.predict(X_test) 
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_train = gaussian.predict(X_train)
acc_gaussian = round(accuracy_score(Y_train,Y_pred_train)*100,2)
print('Accuracy score for train data ',acc_gaussian)
print('confusion matrix for train data \n',  confusion_matrix(Y_train, Y_pred_train))
Y_pred_test_gaussian = gaussian.predict(X_test) 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
knn.fit(X_train, Y_train)
Y_pred_train = knn.predict(X_train)
acc_knn = round(accuracy_score(Y_train,Y_pred_train)*100,2)
print('Accuracy score for train data ',acc_knn)
print('confusion matrix for train data \n',  confusion_matrix(Y_train, Y_pred_train))
Y_pred_test_knn = knn.predict(X_test) 
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state = 0)
dt_model.fit(X_train, Y_train)
Y_pred_train = dt_model.predict(X_train)
acc_dt = round(accuracy_score(Y_train,Y_pred_train)*100,2)
print('Accuracy score for train data ',acc_dt)
print('confusion matrix for train data \n',  confusion_matrix(Y_train, Y_pred_train))
Y_pred_test_dt_model = dt_model.predict(X_test) 
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
rf_model = RandomForestClassifier(random_state = 0,
                                  
                                n_estimators = 15,
                                min_samples_split = 2,
                                min_samples_leaf = 1)

rf_model.fit(X_train, Y_train)
Y_pred_train = rf_model.predict(X_train)
acc_rf = round(accuracy_score(Y_train,Y_pred_train)*100,2)
print('Accuracy score for train data ',acc_rf)
print('confusion matrix for train data \n',  confusion_matrix(Y_train, Y_pred_train))
Y_pred_test_rf_model = rf_model.predict(X_test) 
models = pd.DataFrame({
    'Model': ['LR', 'NB', 'KNN', 'DT', 'RF'],
    'Accuracy': [acc_logreg, acc_gaussian, acc_knn, acc_dt,acc_rf]})
models.sort_values(by='Accuracy', ascending=True)
ax=sns.barplot(x='Model', y='Accuracy', data= models)
ax.set_title("Number of matches won by Each Team")
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x(), p.get_height()+2))
gender_submission['Survived'] = Y_pred_test_dt_model
gender_submission.to_csv('/kaggle/working//submission.csv', index=False)