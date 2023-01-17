import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
titanic = pd.read_csv('../input/titanic/train.csv')
titanic.head()
plt.figure(figsize=(16,14))
#Survived vs Unsurvived
plt.subplot(2,2,1)
sns.countplot(titanic['Survived'],palette='viridis');
plt.title('Survived vs Unsurvived',fontsize=16)
plt.xlabel('Survived',fontsize=14)
plt.ylabel('Count',fontsize=14);
#Male vs Female
plt.subplot(2,2,2)
sns.countplot(titanic['Survived'],hue=titanic['Sex'],palette='viridis');
plt.title('Male vs Female',fontsize=16)
plt.xlabel('Survived',fontsize=14)
plt.ylabel('Count',fontsize=14);
#Passesnger Travelled on Different Classes
plt.subplot(2,2,3)
sns.countplot(titanic['Pclass'],palette='viridis');
plt.title('Passenger travelled on Different Classes',fontsize=16)
plt.xlabel('Class Travelled',fontsize=14)
plt.ylabel('Count',fontsize=14);
#Passesnger Travelled on Different Classes Survived vs Unsurvived
plt.subplot(2,2,4)
sns.countplot(titanic['Pclass'],hue=titanic['Survived'],palette='viridis');
plt.title('Passenger Survived vs Unsurvived on Different Classes',fontsize=16)
plt.xlabel('Class Travelled',fontsize=14)
plt.ylabel('Count',fontsize=14);
plt.tight_layout(pad=3.0)
plt.figure(figsize=(16,14))
#Passengers Traveled with Siblings and Spouse
plt.subplot(2,2,1)
sns.countplot(titanic['SibSp'],palette='viridis');
plt.title('Passengers Traveled with Siblings and Spouse',fontsize=16)
plt.xlabel('No of Siblings and Spouse',fontsize=14)
plt.ylabel('Count',fontsize=14);
#Passengers Traveled with Parents and Children
plt.subplot(2,2,2)
sns.countplot(titanic['Parch'],palette='viridis');
plt.title('Passengers Traveled with Parents and Children',fontsize=16)
plt.xlabel('No of Parents and Children',fontsize=14)
plt.ylabel('Count',fontsize=14);
#Distrubution of Age
plt.subplot(2,2,3)
sns.distplot(titanic['Age'],kde_kws={"color": "k", "lw": 3, "label": "KDE"},hist_kws={"histtype": "step", "linewidth":3,"alpha":1,"color": "g"})
plt.title('Distrubution of Age',fontsize=16)
plt.xlabel('Age',fontsize=14)
#Distrubution of Fare
plt.subplot(2,2,4)
sns.distplot(titanic['Fare'],kde=False,hist_kws={'lw':3,"alpha":0.8,"color":"g"},bins=30);
plt.title('Distrubution of Fare',fontsize=16)
plt.xlabel('Fare',fontsize=14)
plt.tight_layout(pad=3.0)

titanic.isnull().sum()
titanic = titanic.drop('Cabin',axis=1)
titanic['Fare'] = np.log(titanic['Fare'])
def impute(col):
    age = col[0]
    clas = col[1]
    if(pd.isnull(age)):
        if(clas==1):
            return 37
        elif(clas == 2):
            return 29
        else:
            return 25
    return age    
titanic['Age'] = titanic[['Age','Pclass']].apply(impute,axis=1)
titanic.dropna(inplace=True)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x=titanic['Fare'],data=titanic)
plt.xlabel('Fare',fontsize=14)
plt.subplot(1,2,2)
sns.boxplot(x=titanic['Age'],data=titanic)
plt.xlabel('Age',fontsize=14)
plt.suptitle('Outlier',fontsize=16);
plt.tight_layout(pad=3.0)
Q1 = titanic['Age'].quantile(0.25)
Q3 = titanic['Age'].quantile(0.75)
IQR = Q3-Q1
low_wiskers = Q1-1.5*IQR
upper_wiskers = Q3+1.5*IQR
titanic= titanic[(titanic['Age']>low_wiskers)&(titanic['Age']<upper_wiskers)]
Q1 = titanic['Fare'].quantile(0.25)
Q3 = titanic['Fare'].quantile(0.75)
IQR = Q3-Q1
low_wiskers = Q1-1.5*IQR
upper_wiskers = Q3+1.5*IQR
titanic = titanic[(titanic['Fare']>low_wiskers)&(titanic['Fare']<upper_wiskers)]
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x=titanic['Fare'],data=titanic)
plt.xlabel('Fare',fontsize=14)
plt.subplot(1,2,2)
sns.boxplot(x=titanic['Age'],data=titanic)
plt.xlabel('Age',fontsize=14)
plt.suptitle('Outlier Removal',fontsize=16);
plt.tight_layout(pad=3.0)
titanic.isnull().sum()
titanic.head()
titanic.isnull().sum()
#Mapping Sex Column Male-0,Female-1
titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
#Getting Dummies for Embarked
embarked = pd.get_dummies(titanic['Embarked'])
#Dropping original column and concatinating encoded column
titanic = titanic.drop('Embarked',axis=1)
titanic = pd.concat([titanic,embarked],axis=1)
#Creating Family as Sbsp+Parch and dropping SbSp&Parch
titanic['Family'] = titanic['SibSp']+titanic['Parch']
titanic = titanic.drop(['SibSp','Parch'],axis=1)
#Dropping Columns
titanic.drop('Ticket',axis=1,inplace=True)
titanic.drop('PassengerId',axis=1,inplace=True)
titanic = titanic.drop('Name',axis=1)
test = pd.read_csv('../input/titanic/test.csv')
test.head()
#handling Missing Value
test['Age'] = test[['Age','Pclass']].apply(impute,axis=1)
test = test.drop('Cabin',axis=1)
#Mapping Sex Column Male-0,Female-1
test['Sex'] = test['Sex'].map({'male':0,'female':1})
#Getting Dummies for Embarked
embarked = pd.get_dummies(test['Embarked'])
#Dropping original column and concatinating encoded column
test = test.drop('Embarked',axis=1)
test = pd.concat([test,embarked],axis=1)
#Creating Family as Sbsp+Parch and dropping SbSp&Parch
test['Family'] = test['SibSp']+test['Parch']
test = test.drop(['SibSp','Parch'],axis=1)
#Dropping Columns
test.drop(['Ticket','PassengerId','Name'],axis=1,inplace=True)
test = test.fillna(int(test['Fare'].mean()))
X = titanic.drop('Survived',axis=1)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Kogistic Regression
model_log = LogisticRegression()
model_log.fit(X_train,y_train)

model_log.score(X_train,y_train)
#Support Vector Classifier
model_svm = SVC(kernel='rbf')
model_svm.fit(X_train,y_train)
model_svm.score(X_train,y_train)
model_svm.score(X_test,y_test)
#Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)
model_rf.score(X_train,y_train)
model_rf.score(X_test,y_test)
#Scaling Entire Dataset
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
#Performing GridSearchCV
CV_rfc = GridSearchCV(estimator=model_rf, param_grid=param_grid,cv= 5)
CV_rfc.fit(X_scaled,y)
#Getting Best Parameters
CV_rfc.best_params_
#Retraining with best Parameters
model_cvrf = RandomForestClassifier(criterion='gini',max_depth=7,max_features='log2',n_estimators=200)
model_cvrf.fit(X_scaled,y)
model_cvrf.score(X_scaled,y)
pred = model_cvrf.predict(test_scaled)
#Submission
test_df = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':pred})
submission.to_csv("/kaggle/working/gender_submission.csv",index=False)