import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_raw=pd.read_csv('../input/train.csv')
test_raw=pd.read_csv('../input/test.csv')
train_raw['Title']=train_raw['Name'].apply(lambda x:x.split(',')[1].split()[0].strip('.'))
test_raw['Title']=test_raw['Name'].apply(lambda x:x.split(',')[1].split()[0].strip('.'))
train_raw.info()
test_raw.info()
train_raw.describe()
sns.heatmap(train_raw.corr(),cmap='viridis',cbar=False)
train=train_raw.copy()
test=test_raw.copy()
train.head()
train.Title.unique()
train.Title.replace(['Mlle','Ms'],'Miss',inplace=True)
train.Title.replace('Mme','Mrs',inplace=True)
train.Title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'the','Jonkheer', 'Dona'],'Higher',inplace=True)

test.Title.replace(['Mlle','Ms'],'Miss',inplace=True)
test.Title.replace('Mme','Mrs',inplace=True)
test.Title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'the','Jonkheer', 'Dona'],'Higher',inplace=True)
train.Title.unique()
test.Title.unique()
train.drop(columns=['Name','Cabin','Ticket','PassengerId'],inplace=True)
test.drop(columns=['Name','Cabin','Ticket'],inplace=True)
train=pd.get_dummies(data=train,columns=['Sex','Embarked'],drop_first=True)
test=pd.get_dummies(data=test,columns=['Sex','Embarked'],drop_first=True)
train=pd.get_dummies(data=train,columns=['Title'],drop_first=True)
test=pd.get_dummies(data=test,columns=['Title'],drop_first=True)
train.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
xage=train.dropna().drop(['Survived','Age','Embarked_Q','Embarked_S'],axis=1); yage=train.dropna()['Age']
xage_train,xage_test,yage_train,yage_test=train_test_split(xage,yage,test_size=0.2,random_state=5)
lmage=LinearRegression().fit(xage_train,yage_train)
agepred=np.abs(lmage.predict(xage_test))
train_raw.columns
def imputeage(cols):
    pclass=cols[0]
    sex=cols[1]
    age=cols[2]
    sibsp=cols[3]
    title=cols[4]
    return train_raw[(train_raw['Pclass']==pclass) & (train_raw['Sex']==sex) & (train_raw['SibSp']==sibsp)].drop('Cabin',axis=1).dropna()['Age'].mean()
agedf2=train_raw[['Pclass', 'Sex', 'Age', 'SibSp','Title']].copy().dropna()
agedf2['NewAge']=agedf2[['Pclass','Sex','Age','SibSp','Title']].apply(imputeage,axis=1)
agedf2.info()
xage2_train,xage2_test,yage2_train,yage2_test=train_test_split(agedf2.dropna(subset=['Age'],axis=0)['Age'],agedf2.dropna(subset=['Age'],axis=0)['NewAge'],test_size=0.2,random_state=5)
fillmethodcompare=pd.DataFrame()
fillmethodcompare['m1_age']=yage_test
fillmethodcompare['m1_predage']=agepred
fillmethodcompare['m2_age']=xage2_test
fillmethodcompare['m2_newage']=yage2_test
fillmethodcompare.sample(10,random_state=101)
from IPython.display import display, HTML
agedf=pd.DataFrame()
agedf['Real']=yage_test
agedf['Predicted']=agepred
agedf['Delta']=agedf['Real']-agedf['Predicted']
agedf['Mean_Delta']=np.abs(agedf['Real']-agedf['Predicted'])
agedf['Mean_Delta_Normalised']=(agedf['Mean_Delta']/agedf['Real'])*100

#display(HTML(agedf.to_html()))

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(yage_test, agepred))
print('Mean Square Error:', metrics.mean_squared_error(yage_test, agepred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yage_test, agepred)))
print('Mean Error by Percentage of value: ',agedf['Mean_Delta_Normalised'].mean(),'%')
print('Standard deviation of Error Percentage: ',np.std(agedf['Mean_Delta_Normalised']),'%')
sns.distplot(agedf['Delta'],bins=15,kde=False); plt.tight_layout()
age2_errorpercent=np.abs((yage2_test-xage2_test)*100/xage2_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(xage2_test,yage2_test))
print('Mean Square Error:', metrics.mean_squared_error(xage2_test,yage2_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(xage2_test,yage2_test)))
print('Mean Error by Percentage of value: ',age2_errorpercent.mean(),'%')
print('Standard deviation of Error Percentage: ',np.std(age2_errorpercent),'%')
sns.distplot(yage2_test-xage2_test,bins=15,kde=False); plt.tight_layout()
train['NewAge']=lmage.predict(train.drop(['Survived','Age','Embarked_Q','Embarked_S'],axis=1))
def imputeagefill(ages):
    if pd.isnull(ages[0]):
        return np.abs(ages[1])
    else:
        return np.abs(ages[0])
train['Age']=train[['Age','NewAge']].apply(imputeagefill,axis=1)
train.drop('NewAge',axis=1,inplace=True)
train.sample(10)
test['Fare']=test['Fare'].fillna(test['Fare'].mean())
test['NewAge']=lmage.predict(test.drop(['Age','Embarked_Q','Embarked_S','PassengerId'],axis=1))
test['Age']=test[['Age','NewAge']].apply(imputeagefill,axis=1)
test.drop('NewAge',axis=1,inplace=True)
test.sample(10)
train.info()
test.info()
plt.figure(figsize=(12,8))
sns.heatmap(train.corr(),cmap='coolwarm',annot=True,cbar=False); plt.show()
sns.countplot(x='Sex',data=train_raw,hue='Survived')
sns.countplot(x='SibSp',data=train_raw,hue='Survived')
sns.countplot(x='Pclass',data=train_raw,hue='Survived')
sns.countplot(x='Embarked',data=train_raw,hue='Survived')
sns.kdeplot(train[train['Survived']==1]['Age'].dropna(),shade=True,label='Survived=1')
sns.kdeplot(train[train['Survived']==0]['Age'].dropna(),shade=True,label='Survived=0')
plt.legend(); plt.xlabel('Age'); plt.ylabel('KDE'); plt.show()
plt.figure(figsize=(8,5))
sns.kdeplot(train[train['Survived']==0]['Fare'],shade=True,label='Survived=0')
sns.kdeplot(train[train['Survived']==1]['Fare'],shade=True,label='Survived=1')
plt.xlabel('Fare'); plt.ylabel('KDE'); plt.xlim(-40,300); plt.tight_layout()
train.info()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
X=train.drop('Survived',axis=1); y=train['Survived']
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=5)
Algorithms = [LogisticRegression(), RandomForestClassifier(), SVC(), XGBClassifier(), KNeighborsClassifier()]

algoNames=[]; algoScores=[]
for algo in Algorithms:
    algo.fit(X_train,y_train)
    algoName=algo.__class__.__name__
    algoScore=algo.score(X_train,y_train)
    algoNames.append(algoName)
    algoScores.append(algoScore)
algocompare=pd.DataFrame({'Algorithm Name':algoNames,'Score':algoScores}).sort_values(by='Score',ascending=False)
sns.barplot(x='Score',y='Algorithm Name',data=algocompare); plt.show()
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators':[10,50,100,200],'criterion':['gini','entropy'],'bootstrap':[True,False],
           'random_state':[5],'max_features':['auto','log2',None]}
RFC_grid_search=GridSearchCV(RandomForestClassifier(),param_grid)
RFC_grid_search.fit(X_train,y_train)
RFC_grid_search.best_params_
RFC_grid_search.best_score_
from sklearn.metrics import accuracy_score

rfc_pred=RFC_grid_search.predict(X_test)
print(accuracy_score(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
RFC=RandomForestClassifier(n_estimators=100).fit(X,y)
RFC_test_predictions=RFC.predict(test.drop('PassengerId',axis=1))
test['Survived']=RFC_test_predictions
submit=test[['PassengerId','Survived']]
submit.to_csv('submit.csv',index=False)
submit.head()