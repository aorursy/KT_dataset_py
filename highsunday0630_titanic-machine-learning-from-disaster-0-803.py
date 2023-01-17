import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 
%matplotlib inline 
import warnings

warnings.filterwarnings('ignore')
Data_train=pd.read_csv('../input/titanic/train.csv')

Data_test=pd.read_csv('../input/titanic/test.csv')

Data_train.head()
Train_Survived=Data_train['Survived']

Data_train=Data_train.drop(['Survived'],axis=1)
AllData=pd.concat([Data_train,Data_test])

AllData.head()
AllData.describe()
AllData.info()
AllData['Fare'].fillna(value=AllData['Fare'].mean(), inplace=True)
AllData['Cabin'].fillna("N",inplace=True)
AllData['Cabin']=AllData['Cabin'].apply(lambda x: x[0])
AllData.groupby('Cabin')['Fare'].mean().sort_values()
def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a

DataCabinEqual_N=AllData[AllData['Cabin']=='N']
DataCabinNotEqual_N=AllData[AllData['Cabin']!='N']
DataCabinEqual_N['Cabin'] = DataCabinEqual_N['Fare'].apply(lambda x: cabin_estimator(x))
AllData=pd.concat([DataCabinEqual_N, DataCabinNotEqual_N],ignore_index=True)
AllData['Age']=AllData.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
AllData['Embarked'].fillna('S',inplace=True)
AllData.sort_values(by=['PassengerId'],inplace=True)
AllData.index=range(0,len(AllData))
Data_train=AllData[:891]
Data_train['Survived']=Train_Survived
Data_test=AllData[891:]
Data_test.index=range(len(Data_test))
Data_train
Data_test
sns.catplot( x="Pclass", kind="count",hue='Survived', data=Data_train)
sns.catplot( x="Sex", kind="count",hue='Survived', data=Data_train)
sns.catplot(y="Survived", x="SibSp",data=Data_train,kind="bar")
sns.catplot("Survived", col="SibSp",data=Data_train,kind="count",aspect=0.4,height=3)
sns.catplot(y="Survived", x="Parch",data=Data_train,kind="bar")
sns.catplot("Survived", col="Parch",data=Data_train,kind="count",aspect=0.4,height=3)
#Parch 和 Sibsp 有類似的分布
sns.catplot(y="Survived", x="Cabin",data=Data_train,kind="bar")
sns.catplot("Survived", col="Cabin",data=Data_train,kind="count",aspect=0.4,height=3)
sns.catplot( x="Embarked", kind="count",hue='Survived', data=Data_train)
fig, ax = plt.subplots()
ax.set_xlim(0,80)
distplot = sns.distplot(Data_train[Data_train['Survived']==1]['Age'], label="Survived")
distplot = sns.distplot(Data_train[Data_train['Survived']==0]['Age'], label="Un-Survived")
plt.legend()
fig, ax = plt.subplots()
ax.set_xlim(1,100)
distplot = sns.distplot(Data_train[Data_train['Survived']==1]['Fare'], label="Survived")
distplot = sns.distplot(Data_train[Data_train['Survived']==0]['Fare'], label="Un-Survived")
plt.legend()
bins=[0,15, 30,40,100]
Data_train['Age_bin']=pd.cut(Data_train['Age'], bins)
sns.catplot( x="Age_bin", kind="count",hue='Survived', data=Data_train)
fig, ax = plt.subplots()
ax.set_xlim(1,100)
distplot = sns.distplot(Data_train[Data_train['Survived']==1]['Fare'], label="Survived")
distplot = sns.distplot(Data_train[Data_train['Survived']==0]['Fare'], label="Un-Survived")
plt.legend()
Data_train['Fare_bin']=pd.qcut(Data_train['Fare'], 5)
sns.catplot( x="Fare_bin", kind="count",hue='Survived', data=Data_train)
Data_train['Name'][4].split( )[1]  
def NameToTitle(x):
    return x.split()[1][:-1]
Data_train['Name'].apply(NameToTitle)
Data_train['Title']=Data_train['Name'].apply(NameToTitle)
Data_train.groupby('Title')['Title'].count().sort_values()
titleMappingList=['Mr','Miss','Mrs','Master']
def titleMapping(x):
    if x not in titleMappingList:
        return 'Rare'
    else:
        if x=='Miss' or x=='Mrs':
            return "Miss/Mrs"
        else:
            return x
Data_train['Title']=Data_train['Title'].apply(titleMapping)
sns.catplot( x="Title", kind="count",hue='Survived', data=Data_train)
Data_train2=Data_train.copy()
Data_train2['count']=1
Data_train2=Data_train2.groupby('Ticket').sum()[['Survived','count']]
Data_train2=Data_train2[Data_train2['count']>1]
Data_train2['Survived']=Data_train2['Survived']/Data_train2['count']
Data_train2['Survived']=Data_train2['Survived']-0.5
familyConDict=(Data_train2.T).to_dict()
familyConDict
def familyConFun(x):
    if(x in familyConDict):
        return familyConDict[x]['Survived']*familyConDict[x]['count']
    else:
        return 0
Data_train["Family Connection"]=Data_train['Ticket'].apply(familyConFun)
sns.catplot(y="Survived", x="Family Connection",data=Data_train,kind="bar",aspect=0.8,height=8)
sns.catplot("Survived", col="Family Connection",data=Data_train[Data_train["Family Connection"]<0],kind="count",aspect=0.4,height=6)
sns.catplot("Survived", col="Family Connection",data=Data_train[Data_train["Family Connection"]>0],kind="count",aspect=0.4*7/4,height=6)
Data_train
AllData
bins=[0,15, 30,40,100]
AllData['Age_bin']=pd.cut(AllData['Age'], bins)
AllData['Fare_bin']=pd.qcut(AllData['Fare'], 5)
#對Title作處理
AllData['Name'].apply(NameToTitle)
AllData['Title']=AllData['Name'].apply(NameToTitle)
AllData['Title']=AllData['Title'].apply(titleMapping)
AllData["Family Connection"]=AllData['Ticket'].apply(familyConFun)
AllData
AllData.info()
AllData_selected=AllData[['PassengerId','Pclass','Sex','Cabin','Embarked','Age_bin','Fare_bin','Title','Family Connection']]
AllData_selected
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
AllData_selected['Sex']=labelencoder.fit_transform(AllData_selected.iloc[:, 2])
Pclass_oneHot=pd.get_dummies(AllData_selected['Pclass'],prefix='Pclass')
Cabin_oneHot=pd.get_dummies(AllData_selected['Cabin'],prefix='Cabin')
Embarked_oneHot=pd.get_dummies(AllData_selected['Embarked'],prefix='Embarked')
Age_bin_oneHot=pd.get_dummies(AllData_selected['Age_bin'],prefix='Age')
Fare_bin_oneHot=pd.get_dummies(AllData_selected['Fare_bin'],prefix='Fare')
Title_oneHot=pd.get_dummies(AllData_selected['Title'],prefix='Title')
AllData_selected=pd.concat([AllData_selected[['PassengerId','Sex','Family Connection']],Pclass_oneHot,Age_bin_oneHot,Fare_bin_oneHot,Title_oneHot],axis=1)
Data_train=AllData_selected[:891]
#Train_Survived is label of training set
#Data_train['Survived']=Train_Survived
Data_test=AllData_selected[891:]
Data_train.drop('PassengerId',axis=1,inplace=True)
Data_test.drop('PassengerId',axis=1,inplace=True)
Data_train
Data_test
X=Data_train
y=Train_Survived
scoreList=[]
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression( C=1)
scores = cross_val_score(clf, X, y, cv=5)
print('LogisticRegression 平均準確率:'+str(scores.mean()))
scoreList.append(scores.mean())
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print('SVM(linear) 平均準確率:'+str(scores.mean()))
scoreList.append(scores.mean())
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print('SVM(rbf) 平均準確率:'+str(scores.mean()))
scoreList.append(scores.mean())
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1)
#cv = StratifiedKFold(y, random_state=1)        # Setting random_state is not necessary here
scores = cross_val_score(clf, X,y,scoring='accuracy', cv=5)
print('RandomForest 平均準確率:'+str(scores.mean()))
scoreList.append(scores.mean())
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
#cv = StratifiedKFold(y, random_state=1)        # Setting random_state is not necessary here
scores = cross_val_score(clf, X,y,scoring='accuracy', cv=5)
print('GradientBoost 平均準確率:'+str(scores.mean()))
scoreList.append(scores.mean())
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
#cv = StratifiedKFold(y, random_state=1)        # Setting random_state is not necessary here
scores = cross_val_score(clf, X,y,scoring='accuracy', cv=5)
print('AdaBoost 平均準確率:'+str(scores.mean()))
scoreList.append(scores.mean())
print("Overall average : "+str(sum(scoreList)/len(scoreList)))
model1 = GradientBoostingClassifier()
model1.fit(X, y)
print('in data 準確率:',model1.score(X,y))
res=model1.predict(Data_test)
df1=AllData['PassengerId'][891:].copy()
df1.reset_index(drop=True, inplace=True)
df2=pd.DataFrame(res)
df2.reset_index(drop=True, inplace=True)
model1_Res = pd.concat( [df1, df2], axis=1) 
model1_Res.columns=['PassengerId','Survived']
model1_Res
model1_Res.to_csv('GradBoost_Res4_2.csv',index=False)
model2 = svm.SVC(kernel='rbf', C=1)
model2.fit(X, y)
print('in data 準確率:',model2.score(X,y))
model2_res = model2.predict(Data_test)
df1=AllData['PassengerId'][891:].copy()
df1.reset_index(drop=True, inplace=True)
df2=pd.DataFrame(model2_res)
df2.reset_index(drop=True, inplace=True)
model2_Res = pd.concat( [df1, df2], axis=1) 
model2_Res.columns=['PassengerId','Survived']
model2_Res.to_csv('SVM_rbf4_2.csv',index=False)
model3 = LogisticRegression( C=1)
model3.fit(X, y)
print('in data 準確率:',model3.score(X,y))
model3_res = model3.predict(Data_test)
df1=AllData['PassengerId'][891:].copy()
df1.reset_index(drop=True, inplace=True)
df2=pd.DataFrame(model3_res)
df2.reset_index(drop=True, inplace=True)
model3_Res = pd.concat( [df1, df2], axis=1) 
model3_Res.columns=['PassengerId','Survived']
model3_Res.to_csv('Linear_Regression4_1.csv',index=False)
model4 = RandomForestClassifier(random_state=1,n_estimators=1000)
model4.fit(X, y)
print('in data 準確率:',model4.score(X,y))
model4_res = model4.predict(Data_test)
df1=AllData['PassengerId'][891:].copy()
df1.reset_index(drop=True, inplace=True)
df2=pd.DataFrame(model4_res)
df2.reset_index(drop=True, inplace=True)
model4_Res = pd.concat( [df1, df2], axis=1) 
model4_Res.columns=['PassengerId','Survived']
model4_Res.to_csv('RandomForest4_1.csv',index=False)
model5 = AdaBoostClassifier()
model5.fit(X, y)
print('in data 準確率:',model5.score(X,y))
model5_res = model5.predict(Data_test)
df1=AllData['PassengerId'][891:].copy()
df1.reset_index(drop=True, inplace=True)
df2=pd.DataFrame(model5_res)
df2.reset_index(drop=True, inplace=True)
model5_Res = pd.concat( [df1, df2], axis=1) 
model5_Res.columns=['PassengerId','Survived']
model5_Res.to_csv('AdaBoost4_1.csv',index=False)
model1_Res.columns=['PassengerId', 'Survived_1']
model2_Res.columns=['PassengerId', 'Survived_2']
model3_Res.columns=['PassengerId', 'Survived_3']
model4_Res.columns=['PassengerId', 'Survived_4']
model5_Res.columns=['PassengerId', 'Survived_5']

temp=model1_Res.merge(model2_Res, on='PassengerId')

temp=model3_Res.merge(temp, on='PassengerId')

temp=model4_Res.merge(temp, on='PassengerId')

AllModel=model5_Res.merge(temp, on='PassengerId')
AllModel['sum']=AllModel['Survived_1']+AllModel['Survived_2']+AllModel['Survived_3']+AllModel['Survived_4']++AllModel['Survived_5']
AllModel['sum']=AllModel['sum']/5
AllModel
sns.catplot(x='sum',kind='count',data=AllModel)
def AllModel1(x):
    if(x<0.5):
        return 0
    elif(x>0.5):
        return 1
AllModel['Survived']=AllModel['sum'].apply(AllModel1)

AllModel=AllModel[['PassengerId','Survived']]

AllModel['Survived'] = AllModel['Survived'].astype(np.int64)

AllModel.to_csv('5_Model_4_1.csv',index=False)
