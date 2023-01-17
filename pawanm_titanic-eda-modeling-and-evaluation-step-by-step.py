# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the Dataset (This steps comes under data gathering for analysis)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train.dtypes.value_counts()
train.describe()
f,ax = plt.subplots(2,3,figsize=(20,10))
sns.countplot('Pclass',data=train,ax=ax[0,0])
sns.countplot('Sex',data=train,ax=ax[0,1])
sns.countplot("SibSp",data=train,ax=ax[0,2])
sns.countplot("Survived",data=train,ax=ax[1,0])
sns.countplot("Parch",data=train,ax=ax[1,1])
sns.countplot("Embarked",data=train,ax=ax[1,2])
plt.show()
# First Lets start with feature Sex 
train.groupby(['Sex','Survived']).Survived.count()
f,ax=plt.subplots(1,2,figsize=(15,5))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex:Survived or dead')
plt.show()
# Now Lets discussed with feature Pclass
train.groupby(["Pclass"]).Survived.count()
f,ax=plt.subplots(1,2,figsize=(15,5))
train['Pclass'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title("No of Pclass")
ax[0].set_ylabel('Count')
sns.countplot("Pclass",hue="Survived",data=train,ax=ax[1])
ax[1].set_title("Pclass Vs Survived or Dead")
# Analysis of SibSP features
f,ax = plt.subplots(1,2,figsize=(15,5))
train.SibSp.value_counts().plot.bar(ax=ax[0])
ax[0].set_title("No of SibSp")
ax[0].set_xlabel("SibSp")
ax[0].set_ylabel("Count of SibSp")
sns.countplot("SibSp",hue="Survived",data=train,ax=ax[1])
sns.factorplot('SibSp','Survived',data=train,color="red")
# Analysis of Parch features
f,ax = plt.subplots(1,3,figsize=(20,5))
train.Parch.value_counts().plot.bar(ax=ax[0])
train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[2])
ax[0].set_title("No of Parch")
ax[0].set_xlabel("Parch")
ax[0].set_ylabel("Count of Parch")
sns.countplot("Parch",hue="Survived",data=train,ax=ax[1])
sns.factorplot('Parch','Survived',data=train,color="red")
# Analysis of Embarked features
f,ax = plt.subplots(2,2,figsize=(20,10))
train.Embarked.value_counts().plot.bar(ax=ax[0,0])
ax[0,0].set_title("Count")
ax[0,0].set_xlabel("Embarked")
ax[0,0].set_ylabel("Count")
sns.countplot("Embarked",hue="Survived",data=train,ax=ax[0,1])
train[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax[1,0])
sns.countplot("Embarked",hue="Pclass",data=train,ax=ax[1,1])

# Checking the missing values in Embarcked 
train.Embarked.isnull().sum()
train.Embarked.fillna("S",inplace=True)
train.Embarked.isnull().sum()
sns.factorplot('Embarked','Survived',data=train)
# Analysis of AGE

print("Oldest Age:",train.Age.max(),"Years")
print("Youngest Age:",train.Age.min(),"Years")
print("Average Age:",train.Age.mean(),"Years")
f,ax = plt.subplots(1,3,figsize=(20,5))
sns.violinplot(train.Age,ax=ax[0])
ax[0].set_title("Age Distribution")
sns.violinplot("Pclass","Age",hue="Survived",data=train,split=True,ax=ax[1])
ax[1].set_title("Pclass and Age Vs Survived")
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[2])
ax[2].set_title('Sex and Age vs Survived')
train['Salutations']=0
for i in train:
#lets extract the Salutations for strings which lie between A-Z or a-z and followed by a .(dot).
    train['Salutations']=train.Name.str.extract('([A-Za-z]+)\.') 
train.Salutations.value_counts()
pd.crosstab(train.Salutations,train.Sex).T
train['Salutations'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

pd.crosstab(train.Salutations,train.Sex).T
#lets check the averageof age by Initials
train.groupby('Salutations')['Age'].mean() 
train.loc[(train.Age.isnull())&(train.Salutations=='Mr'),'Age'] = 32
train.loc[(train.Age.isnull())&(train.Salutations=="Miss"),'Age'] = 22
train.loc[(train.Age.isnull())&(train.Salutations=='Mrs'),'Age'] = 36
train.loc[(train.Age.isnull())&(train.Salutations=="Master"),'Age'] = 4
train.loc[(train.Age.isnull())&(train.Salutations=="Other"),'Age'] = 46
train.Age.isnull().any().sum()
f,ax = plt.subplots(1,2,figsize=(20,5))
train[train['Survived']==0].Age.plot.hist(bins=20,color='red',ax=ax[0])
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['Survived']==1].Age.plot.hist(bins=20,color='green',ax=ax[1])
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
# Fare features

train.Fare.describe()
f,ax= plt.subplots(2,2,figsize=(20,5))
train.Fare.plot.hist(bins=20,color='green',ax=ax[0,0])
train[train['Pclass']==1].Fare.plot.hist(bins=20,color='green',ax=ax[0,1])
train[train['Pclass']==2].Fare.plot.hist(bins=20,color='green',ax=ax[1,0])
train[train['Pclass']==3].Fare.plot.hist(bins=20,color='green',ax=ax[1,1])
# Outlier detection 
from collections import Counter
#Once initialized, counters are accessed just like dictionaries.
#Also, it does not raise the KeyValue error (if key is not present) instead the value’s count is shown as 0.
def detect_outliers(df,n,features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col],25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index       
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers   
# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop] # Show the outliers rows
# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
sns.heatmap(train.corr(),annot=True)
train['Age_band']=0
train.loc[train['Age']<=16,'Age_band']=0
train.loc[(train['Age']>16)&(train['Age']<=32),'Age_band']=1
train.loc[(train['Age']>32)&(train['Age']<=48),'Age_band']=2
train.loc[(train['Age']>48)&(train['Age']<=64),'Age_band']=3
train.loc[train['Age']>64,'Age_band']=4
train.sample(5)
train['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
sns.factorplot('Age_band','Survived',data=train,col='Pclass')
plt.show()
train['Family_Size']=0
train['Family_Size']=train['Parch']+train['SibSp']#family size
train['Alone']=0
train.loc[train.Family_Size==0,'Alone']=1#Alone

f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=train,ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=train,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()

sns.factorplot('Alone','Survived',data=train,hue='Sex',col='Pclass')
plt.show()
train['Fare_Range']=pd.qcut(train['Fare'],4)
train.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
train['Fare_cat'] = 0
train.loc[train.Fare<=7.91,'Fare_cat'] = 0
train.loc[(train.Fare>7.91) & (train.Fare<=14.454),'Fare_cat'] =1
train.loc[(train.Fare>14.454) & (train.Fare<=31.0),'Fare_cat'] =2
train.loc[(train.Fare>31.0) & (train.Fare<=512.329),'Fare_cat'] =3
sns.factorplot('Fare_cat','Survived',data=train,hue='Sex')
plt.show()
train.sample(5)
train["Sex"]=train["Sex"].astype(str)
train["Embarked"]=train["Embarked"].astype(str)
train["Salutations"]=train["Salutations"].astype(str)
train.info()
train['Sex'].replace(['male','female'],[0,1],inplace=True)
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train['Salutations'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
train["Sex"]=train["Sex"].astype(int)
train["Embarked"]=train["Embarked"].astype(int)
train["Salutations"]=train["Salutations"].astype(int)
#Dropping unneeded features
train.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
train.head()
sns.heatmap(train.corr(),annot=True)
fig=plt.gcf()
fig.set_size_inches(10,8)
sns.pairplot(train,hue="Survived",size=1.2)
from sklearn.linear_model import LogisticRegression # For Logistic Regression 
from sklearn import svm # For SVM 
from sklearn.ensemble import RandomForestClassifier # For Random Forest
from sklearn.neighbors import KNeighborsClassifier # for KNN 
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier # For Decesion Tree
from sklearn.cross_validation import train_test_split # for Splittng the data into Train and Test 
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
CopyOfTrainDataSet = train.copy() # Working on cpoy of data set and keeping original data set as safe
CopyOfTrainDataSet.head()
X = CopyOfTrainDataSet.drop(["Survived"],axis=1)
Y = CopyOfTrainDataSet["Survived"]

X.head()
Y.head()
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = 0,test_size = 0.3)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
Logist = LogisticRegression(solver="liblinear")
Logist.fit(X_train,y_train)
Logistic_Predict = Logist.predict(X_test)
print('The accuracy of the Logistic Model is',metrics.accuracy_score(Logistic_Predict,y_test))
for n in range(1,20,2):
    
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train,y_train)
    Knn_Predict = knn.predict(X_test)
    print('The accuracy of the KNN Model is',n,metrics.accuracy_score(Knn_Predict,y_test))
DecesionTree = DecisionTreeClassifier()
DecesionTree.fit(X_train,y_train)
DecesionTree_Predict = DecesionTree.predict(X_test)
print('The accuracy of the Decesion Tree Model is',metrics.accuracy_score(DecesionTree_Predict,y_test))

RandomForest = RandomForestClassifier(n_estimators=100)
RandomForest.fit(X_train,y_train)
RandomForest_Predict = RandomForest.predict(X_test)
print('The accuracy of the Random Forest Model is',metrics.accuracy_score(RandomForest_Predict,y_test))

NaiveBais = GaussianNB()
NaiveBais.fit(X_train,y_train)
NaiveBais_Predict = NaiveBais.predict(X_test)
print('The accuracy of the Naive Bais Model is',metrics.accuracy_score(NaiveBais_Predict,y_test))
SVM = svm.SVC(kernel='linear',C=1,gamma=0.1)
SVM.fit(X_train,y_train)
SVM_Predict = SVM.predict(X_test)
print('The accuracy of the Linear SVM Model is',metrics.accuracy_score(NaiveBais_Predict,y_test))

SVM = svm.SVC(kernel='rbf',C=1,gamma=0.1)
SVM.fit(X_train,y_train)
SVM_Predict = SVM.predict(X_test)
print('The accuracy of the Radial SVM Model is',metrics.accuracy_score(NaiveBais_Predict,y_test))
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict

Kfold = KFold(n_splits=10, random_state=22)

xyz=[]
accuracy=[]
std=[]
classifiers=['Radial SVM','Linear Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='rbf',C=1,gamma=0.1),svm.SVC(kernel='linear'),LogisticRegression(),KNeighborsClassifier(n_neighbors=7),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = Kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


plt.subplots(figsize=(12,6))
sns.boxplot(classifiers,accuracy)
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf',C=1,gamma=0.1),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()

from sklearn.model_selection import GridSearchCV
C=[0.5,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)









