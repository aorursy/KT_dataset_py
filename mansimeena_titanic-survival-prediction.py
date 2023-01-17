import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np 
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train_data=pd.read_csv("/kaggle/input/titanic/train.csv")
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

#This displays first 5 data from the training-dataset
train_data.head()
#observation: the dataset consists of categorical variable and a few numerical variables.
train_data.tail()
#displaying bottom 5 data from the training-dataset
#Finding dimensions
print("number of rows in traning set",train_data.shape[0])
print("number of columns in training set",train_data.shape[1])
print("number of rows in test set",test_data.shape[0])
print("number of columns in test set",test_data.shape[1])
train_data.isna().sum()
#Here Age,Cabin and Embarked have missing values
test_data.isna().sum()
#here Age,Fare,Cabin have missing values
# Data preprocessing
sns.heatmap(train_data.corr(),annot=True)
#Age is correlated to PCLASS the most with correlation coff=-0.37.
df=train_data.groupby('Pclass',as_index=False)['Age'].median()
df.head()
#Replacing missing vales in training set
#AGE
for i,row in train_data.iterrows():
    if(np.isnan(train_data['Age'][i])==True):
        if(train_data['Pclass'][i]==1):
            train_data['Age'][i]=37.0
        elif(train_data['Pclass'][i]==2):
            train_data['Age'][i]=29.0
        else:
            train_data['Age'][i]=24.0
#CABIN (It has maximum number of missing values and hence it will NOT be worth a feature to work on. Still we will replace
#the missing values with its mode)
train_data['Cabin']=train_data['Cabin'].fillna(train_data['Cabin'].mode()[0])
#EMBARKED
#For this we have two missing values each having equivalent Fare of $80 and same PClass,same ticketnumber this
#means that they had to board from the same station.And on google search, I got to know that she boarded from SouthHampton.
#This means that Both of them boarded from SouthHampton.
train_data['Embarked']=train_data['Embarked'].fillna('S')
train_data.info()
#no missing values in training set
df=test_data.groupby('Pclass',as_index=False)['Age'].median()
df.head()
#Replacing Missing Values in Test Set
#AGE
for i,row in test_data.iterrows():
    if(np.isnan(test_data['Age'][i])==True):
        if(test_data['Pclass'][i]==1):
            test_data['Age'][i]=42.0
        elif(test_data['Pclass'][i]==2):
            test_data['Age'][i]=26.0
        else:
            test_data['Age'][i]=24.0
#CABIN
test_data['Cabin']=test_data['Cabin'].fillna(test_data['Cabin'].mode()[0])
#FARE
#We can see that Fare is most correlated to Pclass and hence it should be computed by median of fare grouped by pclass.
#As the person belonged to class 3, we will take median of class 3.
df=test_data.groupby('Pclass',as_index=False)['Fare'].median()
df.head()
test_data['Fare']=test_data['Fare'].fillna(7.89)
test_data.isna().sum()
#NO MISSING VALUES
# DATA VISUALIZATION
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Survived'].value_counts().plot.pie(autopct='%0.2f%%',colors=['red','green'])
plt.subplot(122)
plt.title('Survival distribution')
train_data.Survived.value_counts().plot(kind='bar',color=['red','green'])
sns.pairplot(train_data)
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Sex'].value_counts().plot.pie(autopct='%0.2f%%',colors=['blue','pink'])
plt.subplot(122)
sns.countplot(x = 'Survived',data = train_data,hue = 'Sex',palette=['blue','pink'])
plt.xlabel('Survived')
plt.ylabel('Passenger Count')
plt.title('Survival distribution based on Sex')
plt.legend()
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Pclass'].value_counts().plot.pie(autopct='%0.2f%%',colors=['blue','orange','green'])
plt.subplot(122)
sns.countplot(x = 'Survived',data = train_data,hue = 'Pclass',palette=['orange','green','blue'])
plt.xlabel('Survived')
plt.ylabel('Passenger Count')
plt.title('Survival distribution based on Socio economic class of Passenger')
plt.legend()
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Embarked'].value_counts().plot.pie(autopct='%0.2f%%',colors=['blue','orange','green'])
plt.subplot(122)
sns.countplot(x = 'Survived',data = train_data,hue = 'Embarked',palette=['blue','orange','green'])
plt.xlabel('Survived')
plt.ylabel('Passenger Count')
plt.title('Survival distribution based on Embarkment of Passenger')
plt.legend()
plt.figure(figsize=(12,6))
sns.countplot(x = 'SibSp',data = train_data,hue = 'Survived',palette=['red','green'])
plt.xlabel('SibSp')
plt.ylabel('Passenger Count')
plt.title('Survival distribution based on Number of Sibling/spouse with Passenger')
plt.legend()

plt.figure(figsize=(12,6))
sns.countplot(x = 'Parch',data = train_data,hue = 'Survived',palette=['red','green'])
plt.xlabel('Parent/Children')
plt.ylabel('Passenger Count')
plt.title('Survival distribution based on Number of Parent/Children with Passenger')
plt.legend()
sns.countplot(x = 'Parch',data = train_data,hue = 'Survived',palette=['red','green'])
plt.xlabel('Parent/Children')
plt.ylabel('Passenger Count')
plt.title('Survival distribution based on Number of Parent/Children with Passenger')
plt.legend()
plt.figure(figsize=(16,4))
plt.title('Age vs survival')
sns.distplot(train_data['Age'][train_data['Survived']==0],bins=20,kde=True,hist=False,kde_kws={"color": "red", "label": "Not Survived"})
sns.distplot(train_data['Age'][train_data['Survived']==1],bins=20,kde=True,hist=False,kde_kws={"color": "green", "label": "Survived"})
plt.figure(figsize=(16,4))
plt.title('Fare vs survival')
sns.distplot(train_data['Fare'][train_data['Survived']==0],bins=10,kde=True,hist=False,kde_kws={"color": "red", "label": "Not Survived"})
sns.distplot(train_data['Fare'][train_data['Survived']==1],bins=10,kde=True,hist=False,kde_kws={"color": "green", "label": "Survived"})
sns.scatterplot(data=train_data,x='Age',y='Survived',hue='Survived')
traindata=train_data.drop(['Name','Ticket','Cabin','PassengerId','Fare','SibSp','Age'],axis=1)
testdata=test_data.drop(['Name','Ticket','Cabin','PassengerId','Fare','SibSp','Age'],axis=1)
traindata.head()
X_train=traindata.iloc[:,1:]
y_train=traindata.iloc[:,0]
X_train=pd.get_dummies(X_train,['Sex','Embarked'])
X_test=testdata.iloc[:,:]
X_test=pd.get_dummies(X_test,['Sex','Embarked'])
from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
classifier.fit(X_train,y_train)
y_train_predict=classifier.predict(X_train)
y_test=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report
print('Accuracy for training set is: ',accuracy_score(y_train,y_train_predict))
print('AUC SCORE for training set: ',roc_auc_score(y_train,y_train_predict))

cm=confusion_matrix(y_train,y_train_predict)
sns.heatmap(cm,annot=True)
print(classification_report(y_train,y_train_predict))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_test})
output.to_csv('svc-titanic.csv', index=False)
