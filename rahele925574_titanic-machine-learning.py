import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('../input/titanic/train.csv')
df.info()
import pandas_profiling
pandas_profiling.ProfileReport(df)
df.head(5)
dftest=pd.read_csv('../input/titanic/test.csv')
dftest.info()
fig,axs = plt.subplots(nrows=2, figsize=(20,20))

sns.heatmap(df.corr(), ax=axs[0], annot=True,square=True, cmap='Greens', annot_kws={'size': 14},vmin=-1,vmax=+1)
sns.heatmap(dftest.corr(), ax=axs[1], annot=True, square=True, cmap='Greens', annot_kws={'size': 14},vmin=-1,vmax=+1)

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    
axs[0].set_title('Training Set Correlations', size=16)
axs[1].set_title('Test Set Correlations', size=16)

plt.show()
#Age
df['Age']=df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
dftest['Age']=dftest.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
#Embarked
mode=df['Embarked'].mode()
mode
#Filling missing values of Embarked with most frequent value 'S'
df=df.fillna({'Embarked':'S'})
df.info()
#fare 
dftest['Fare'].fillna(dftest['Fare'].median(), inplace=True)
dftest.info()
def f(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
def g(tt):
    if tt in ['Mr']:
        return 1
    elif tt in ['Master']:
        return 3
    elif tt in ['Ms', 'Mlle', 'Miss']:
        return 4
    elif tt in ['Mrs','Mme']:
        return 5
    else:
        return 2
df['title'] = df['Name'].apply(f).apply(g)
dftest['title'] = dftest['Name'].apply(f).apply(g)
#drop unnecessary columns
df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
dftest=dftest.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
t = pd.crosstab(df['title'],df['Survived'])
t
t_pct = t.div(t.sum(1).astype(float), axis=0)
t_pct
t_pct.plot(kind='bar',title='Survival Rate by title')

plt.xlabel('title')

plt.ylabel('Survival Rate')
plt.show()
sns.countplot(x=df['Survived'],hue=df['Sex'],palette='Accent_r')
plt.show()
survived=pd.crosstab([df.Pclass,df.Sex],df.Survived)
survived
ax=survived.div(survived.sum(1),axis=0).plot.barh(stacked='True',color=['indianred','limegreen'])

ax.set_ylim(-1.75,6)
ax.legend(loc=(0.15,0.05),ncol=2)
plt.show()
sns.countplot(x=df['SibSp'],hue=df['Survived'])
plt.show()
sns.countplot(x=df['Parch'],hue=df['Survived'])
plt.show()
fig= plt.figure()
ax=fig.add_subplot()
ax.hist(x=[df[df['Survived']==1]['Age'],df[df['Survived']==0]['Age']], bins = 10, range = (df['Age'].min(),df['Age'].max()),label=['Survived','Dead'],stacked=True,color=['green','red'])
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(x=[df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']], bins = 10, range = (df['Fare'].min(),df['Fare'].max()),stacked=True,color=['green','red'],label=['Survived','Dead'])

plt.title('Fare distribution')
plt.xlabel('Fare')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()
sns.jointplot(x=df['Age'],y=df['Fare'],height=8,color='mediumvioletred')

plt.show()
sns.boxplot(x=df['Age'],y=df['Sex'],data=df, hue='Survived',palette='Accent')
plt.show()
df.boxplot(column='Fare',figsize=(10,8))
plt.show()
df.boxplot(column='Fare',by='Pclass',figsize=(10,8))
plt.show()
sns.countplot(x=df['Survived'],hue=df['Embarked'],palette='Accent')
plt.show()
g = sns.catplot(x="SibSp",y="Survived",data=df,kind="bar",  
palette = "Accent")
g.despine(left=True)
g = g.set_ylabels("survival probability")
d=[df,dftest]
sex_mapping={"female":1,"male":0}
for dataset in d:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)

Embarked_mapping={'S':0,'C':1,'Q':2}
for dataset in d:
    dataset['Embarked']= dataset['Embarked'].map(Embarked_mapping)

from sklearn.preprocessing import StandardScaler

cal=['Age','Fare']
transf=df[cal]
sc=StandardScaler().fit(transf)
df[cal]=sc.transform(transf)

transf=dftest[cal]
sc=StandardScaler().fit(transf)
dftest[cal]=sc.transform(transf)
dftest.head(5)
df.head(5)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import  GridSearchCV,StratifiedKFold

cv=StratifiedKFold(n_splits=5)
x_train=df.drop('Survived',axis=1)
y_train=df['Survived']
x_test=dftest
test=pd.read_csv('../input/titanic/test.csv')
clf1=[KNeighborsClassifier(n_neighbors = 13),DecisionTreeClassifier(),LogisticRegression(solver='lbfgs'),
       RandomForestClassifier(n_estimators=13),SVC(gamma='scale'),AdaBoostClassifier(),
      GradientBoostingClassifier(n_estimators=10, learning_rate=0.5,max_features=3, max_depth =3, random_state = 10),
     ExtraTreesClassifier()]
def model_fit():
    scoring = 'accuracy'
    for i in range(len(clf1)):
        score = cross_val_score(clf1[i], x_train, y_train, cv=cv, n_jobs=1, scoring=scoring)
        print("Score of Model",i,":",round(np.mean(score)*100,2))
#     round(np.mean(score)*100,2)
#     print("Score of :\n",score)
model_fit()
svm_params=[{'C': [0.1, 1, 10], 'kernel': ['linear']},
               {'C': [0.1, 1, 10], 'kernel': ['rbf'],
                'gamma': [0.1, 0.2, 0.3, 0.4]},
               ]
grid_svm=GridSearchCV(estimator=SVC(random_state=0), param_grid = svm_params, cv = cv, 
                   n_jobs = -1, scoring = "accuracy",iid=True)
grid_svm.fit(x_train,y_train)
grid_svm.best_score_
grid_svm.best_params_
svm=SVC(C=10, gamma=0.1, kernel='rbf')
svm.fit(x_train,y_train)
su=svm.predict(x_test)
passengerid=np.array(test['PassengerId'])
submit=pd.DataFrame({'PassengerId':passengerid,'Survived':su})
submit.to_csv("d://کلاس آنلاین/rr.csv") 
rf_params = {"max_depth": [4,5,6],
              "max_features": ["auto","log2"],
              "min_samples_split": [5,10,15,20],
              "min_samples_leaf": [1,5,10],
              "n_estimators" :[10,15,20],
            "criterion": ["gini","entropy"]}

grid_rf=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=rf_params,cv=cv, n_jobs = -1, scoring = "accuracy",iid=True)
grid_rf.fit(x_train,y_train)
grid_rf.best_score_
grid_rf.best_params_
sub3=grid_rf.predict(x_test)
passengerid=np.array(test['PassengerId'])
submit=pd.DataFrame({'PassengerId':passengerid,'Survived':sub3})
submit.to_csv("d://کلاس آنلاین/sub33.csv") 
rfclf=RandomForestClassifier(criterion='entropy',
 max_depth= 5,
max_features='auto',
min_samples_leaf=1,
min_samples_split= 15,
n_estimators=10)
rfclf.fit(x_train,y_train)
rfclf.feature_importances_
rfclf.base_estimator

df_importance =pd.DataFrame(rfclf.feature_importances_, columns=['Feature_Importance'],
                              index=x_train.columns)
df_importance.sort_values(by='Feature_Importance',ascending=False,inplace=True)
df_importance
sns.barplot(y=df_importance.index,x=df_importance['Feature_Importance'])
plt.show()
clfex=ExtraTreesClassifier()
clfex.fit(x_train,y_train)

dfimportance =pd.DataFrame(clfex.feature_importances_, columns=['Feature_Importance'],
                              index=x_train.columns)
dfimportance.sort_values(by='Feature_Importance',ascending=False,inplace=True)
dfimportance
sns.barplot(y=dfimportance.index,x=dfimportance['Feature_Importance'])
plt.show()
xtrain=df.drop(['Survived','Parch','Embarked'],axis=1)

rf_params = {"max_depth": [4,5,6],
              "max_features": ["auto","log2"],
              "min_samples_split": [5,10,15,20],
              "min_samples_leaf": [1,5,10],
              "n_estimators" :[10,15,20],
            "criterion": ["gini","entropy"]}

grid_rf=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=rf_params,cv=cv, n_jobs = -1, scoring = "accuracy",iid=True)
grid_rf.fit(xtrain,y_train)
grid_rf.best_params_
grid_rf.best_score_
#svc with gridsearchcv
svm_params = [{
                "C" : [0.1, 1, 10], 
                "kernel" : ["poly"],
                "degree" : [2],
                "gamma" : [0.1, 0.2, 0.3,0.4,0.5]},
             {"C" : [0.1, 1, 10], 
                "kernel" : ["rbf"],
                "gamma" : [0.1, 0.2, 0.3,0.4,0.5]},
             
                {"C" : [0.1, 1, 1.0], 
                "kernel" : ["linear"]}
              ]

grid_rf=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=rf_params,cv=cv, n_jobs = -1, scoring = "accuracy",iid=True)
grid_rf.fit(xtrain,y_train)
grid_svm.best_params_
grid_svm.best_score_
sub7=grid_svm.predict(x_test)
passengerid=np.array(test['PassengerId'])
submit=pd.DataFrame({'PassengerId':passengerid,'Survived':sub7})
submit.to_csv("d://کلاس آنلاین/sub7.csv") 
from sklearn import tree
fn=df.columns[1:]
cn=df.columns[0]
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(20,20))
tree.plot_tree(rfclf.estimators_[0],
               max_depth=2,
              feature_names=fn,
              class_names=['s','uns'],
              filled=True)
fig.savefig('rf_individualtree.png')