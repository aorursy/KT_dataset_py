import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

train_df=pd.read_csv('../input/titanic/train.csv')
test_df=pd.read_csv('../input/titanic/test.csv')
train_df.head()
test_df.head()
print('shape of training data',train_df.shape)
print('shape of test data',test_df.shape)
train_df.columns
train_df.describe()
test_df.describe()
train_df.info() #ticket
train_df_without_survival=train_df.drop('Survived',axis=1)
df=pd.concat([train_df_without_survival,test_df],ignore_index=True)
df.shape
df.head()
#missing values
print(train_df.isnull().sum())
print("-"*20)
print(test_df.isnull().sum())
df.drop('Cabin',axis=1,inplace=True)
df.shape[1]
df['Fare'].fillna(df['Fare'].mean(),inplace=True)
df.isnull().sum()
df['Embarked'].value_counts()
df['Embarked'].fillna('S',inplace=True)
df.isnull().sum()
df[['Age','Sex','Pclass','Embarked']].groupby(['Sex','Pclass','Embarked']).agg(['std','median'])
#to calculate the missing value of age
df[['Age','Sex','Pclass']].groupby(['Sex','Pclass']).agg(['std','median'])
grouped=df[['Age','Sex','Pclass']].groupby(['Sex','Pclass']).median()


for i in df[df['Age'].isnull()].index:
    loop_sex=df.loc[i,'Sex']
    loop_pclass=df.loc[i,'Pclass']
    corr_of_pclass=grouped.loc[str(loop_sex),:]
    approx_age=corr_of_pclass.loc[int(loop_pclass),:]
    df.loc[i,'Age']=float(approx_age)
df['Age'].isnull().sum()
#checking the freatures of the dataset by corelating with target
cor_pclass=train_df[['Pclass','Survived']].groupby('Pclass').mean()
cor_pclass
cor_pclass.plot(kind='barh',color='gold')
corr_sex=train_df[['Sex','Survived']].groupby('Sex').mean()
corr_sex
corr_sex.plot(kind='barh',color='lightgreen')
corr_embarked=train_df[['Embarked','Survived']].groupby('Embarked').mean()
corr_embarked
corr_embarked.plot(kind='barh',color='orange')
 #to find the relation between several continuous features and target variable
cont_columns=['Age','SibSp','Parch','Fare']
corelation_df=train_df[cont_columns+['Survived']]
corelation_df.corr()['Survived']
#none of the features provide a significant corelation 


#turning out age to a categorical variable
train_df['Age_Categ']=pd.cut(train_df['Age'],8)
train_df[['Age_Categ','Survived']].groupby('Age_Categ').agg(['sum','count','mean'])
df['Age_Categ']=pd.cut(df['Age'],8)
df1=df['Age_Categ'].astype('str')
df['Age_Categ']=df1.map({'(0.0902, 10.149]': 1,
        '(10.149, 20.128]': 2,
        '(20.128, 30.106]': 2,
        '(30.106, 40.085]': 3,
        '(40.085, 50.064]': 3,
        '(50.064, 60.043]': 3,
        '(60.043, 70.021]': 4,
        '(70.021, 80.0]': 4}).astype('category')
train_df['age_categ']=df.loc[:891,'Age_Categ']
df['Age_Categ']
corr_age=train_df[['age_categ','Survived']].groupby('age_categ').mean()
corr_age
corr_age.plot(kind='bar',color='r')
#getting title from names
df['Name'].head()
title=df[['Name','Ticket']].set_index('Name')
for i in title.index:
    rev=str(i)[::-1]
    tit=rev[rev.index('.')+1:rev[rev.index('.'):].index(' ')+rev.index('.')]
    tit=tit[::-1]
    title.loc[i,'Ticket']=tit
title.reset_index(inplace=True)
df['Title']=title['Ticket']
df['Title'].value_counts()
df['Title'].replace(['Dr','Rev','Col','Major','Ms','Mlle','Lady','Sir','Mme','Countess','L',
                     'Jonkheer','Don','Dona','Capt'],'other',inplace=True)
train_df['Title']=df.loc[:891,'Title']
train_df[['Title','Survived']].groupby('Title').agg(['mean','count'])
corr_title=train_df[['Title','Survived']].groupby('Title').mean()
corr_title.plot(kind='bar',color='green')
df['Title']=df['Title'].map({'Master':1,
                'Miss':2,
                'Mr':3,
                'Mrs':4,
                'other':5})
#using parch and sibsp to find whether the person is single or not
df['Family_Size']=df[['Parch','SibSp']].sum(axis=1)
train_df['Family_Size']=df.loc[:891,'Family_Size']
train_df[['Family_Size','Survived']].groupby('Family_Size').agg(['count','mean'])
def singleornot(famsize):
    if famsize==0:
        return 1
    else:
        return 0
df['Single']=df['Family_Size'].apply(singleornot)
train_df['Single']=df.loc[:891,'Single']
train_df[['Single','Survived']].groupby('Single').agg(['mean','count'])
corr_single=train_df[['Single','Survived']].groupby('Single').mean()
corr_single.plot(kind='bar',color='blue')
df['Fare_Categ']=pd.cut(df['Fare'],20)
train_df['Fare_Categ']=df.loc[:891,'Fare_Categ']
train_df[['Fare_Categ','Survived']].groupby('Fare_Categ').agg(['mean','count'])
df['Fare_Categ']=df['Fare_Categ'].astype(str).map({'(-0.512, 25.616]':1,
                                 '(25.616, 51.233]':2,
                                 '(51.233, 76.849]':2,
                                 '(76.849, 102.466]':3,
                                 '(102.466, 128.082]':3,
                                 '(128.082, 153.699]':3,
                                 '(153.699, 179.315]':3,
                                 '(179.315, 204.932]':3,
                                 '(204.932, 230.548]':3,
                                 '(230.548, 256.165]':3,
                                 '(256.165, 281.781]':3,
                                 '(281.781, 307.398]':3,
                                 '(307.398, 333.014]':3,
                                 '(333.014, 358.63]':3,
                                 '(358.63, 384.247]':3,
                                 '(384.247, 409.863]':3,
                                 '(409.863, 435.48]':3,
                                 '(435.48, 461.096]':3,
                                 '(461.096, 486.713]':3,
                                 '(486.713, 512.329]':3})

train_df['Fare_Categ']=df.loc[:891,'Fare_Categ']
train_df[['Fare_Categ','Survived']].groupby('Fare_Categ').agg(['count','mean'])
corr_fare=train_df[['Fare_Categ','Survived']].groupby('Fare_Categ').mean()
corr_fare.plot(kind='bar',color='black')
#ticket
df.drop('Ticket',axis=1,inplace=True)
#changing the categoricals sex and embarked to numericals 
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].map({'C':1,'Q':2,'S':3})
#droping the unwanted features

df.drop(['PassengerId','Name','Age','SibSp','Parch','Fare','Family_Size'],axis=1,inplace=True)
final_testing_df=df[891:]
final_training_x=df[:891]
final_training_y=train_df.loc[:891,'Survived']
final_training_df=pd.concat([df[:891],train_df['Survived']],axis=1)
final_training_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error
x_train,x_test,y_train,y_test=train_test_split(final_training_x,final_training_y,test_size=0.3)

#logistic regression
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(x_train,y_train)
log_yhat=log_model.predict(x_test)
log_acc=accuracy_score(log_yhat,y_test)
print('accuracy score',log_acc)
from sklearn.tree import DecisionTreeClassifier
tree_model=DecisionTreeClassifier(max_depth=5)
tree_model.fit(x_train,y_train)
tree_yhat=tree_model.predict(x_test)
tree_acc=accuracy_score(tree_yhat,y_test)
print('accuracy score',tree_acc)
from sklearn.ensemble import RandomForestClassifier
forest_model=RandomForestClassifier(n_estimators=100,max_depth=5)
forest_model.fit(x_train,y_train)
forest_yhat=forest_model.predict(x_test)
forest_acc=accuracy_score(forest_yhat,y_test)
print('accuracy score',forest_acc)



from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors=8)
knn_model.fit(x_train,y_train)
knn_yhat=knn_model.predict(x_test)
knn_acc=accuracy_score(knn_yhat,y_test)
print('accuracy score',knn_acc)
from sklearn.naive_bayes import GaussianNB
naive_model=GaussianNB()
naive_model.fit(x_train,y_train)
naive_yhat=naive_model.predict(x_test)
naive_acc=accuracy_score(naive_yhat,y_test)
print('accuracy score',naive_acc)
from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(x_train,y_train)
svc_yhat=svc_model.predict(x_test)
svc_acc=accuracy_score(svc_yhat,y_test)
print('accuracy score',svc_acc)
#scochastic gradient descent
from sklearn.linear_model import SGDClassifier
sgd_model=SGDClassifier()
sgd_model.fit(x_train,y_train)
sgd_yhat=sgd_model.predict(x_test)
sgd_acc=accuracy_score(sgd_yhat,y_test)
print('accuracy score',sgd_acc)
best_algorithm=pd.DataFrame({'algorithm':['LogisticRegression','DecisionTree','RandomForest','KNeighbors',
                                          'GaussianNB','svm','SGD'],
                             'accuracy':[log_acc,tree_acc,forest_acc,knn_acc,naive_acc,svc_acc,sgd_acc]})
best_algorithm
#decision tree is best to classify the dataset
model=RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(final_training_x,final_training_y)
prediction=model.predict(final_testing_df)
prediction
final_df=pd.DataFrame({'PassengerId':final_testing_df.index +1,'Survived':prediction},index=None)

final_df.set_index('PassengerId',inplace=True)
final_df.to_csv('./titanic_submission.csv')