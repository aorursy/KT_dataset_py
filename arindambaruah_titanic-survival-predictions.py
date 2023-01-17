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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
train_df=pd.read_csv('../input/titanic/train.csv')
train_df['Target']=train_df['Survived'].astype(int)
train_df['Survived']=train_df['Survived'].map({1:'Yes',0:'No'})
train_df.head()
train_df.info()
sns.kdeplot(train_df['Age'],shade=True)
sns.boxplot('Age',data=train_df)
train_df['Age'].describe()
train_df['Age'].median()
train_df['Age'].fillna(train_df['Age'].median(),inplace=True)
train_df['Age'].isna().any()
bins=np.arange(0,90,10) - 0.5
plt.figure(figsize=(10,8))
plt.hist(train_df['Age'],bins=bins)
plt.xticks(rotation=90)
def child(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex
train_df['Person']=train_df[['Age','Sex']].apply(child,axis=1)
train_df['Person'].value_counts()
fig1=sns.FacetGrid(train_df,hue='Sex',aspect=2,height=6)
fig1.map(sns.kdeplot,'Age',shade=True)
fig1.add_legend()


fig2=sns.FacetGrid(train_df,hue='Person',aspect=2,height=8)
fig2.map(sns.kdeplot,'Age',shade=True)
fig2.add_legend()


sns.catplot('Person',data=train_df,kind='count',hue='Survived',aspect=2,height=6)
sns.lmplot('Age','Target',data=train_df,aspect=2,height=6)
train_df['Embarked'].unique()
train_df['Embarked'].value_counts()
train_df['Embarked'].isnull().value_counts()
train_df['Embarked'].replace(np.nan,'S',inplace=True)
train_df['Embarked'].isnull().any()
sns.catplot('Embarked',data=train_df,kind='count',aspect=2,height=6)
ax=sns.catplot('Embarked',data=train_df,kind='count',hue='Survived',aspect=2,height=6)
ax.set_xticklabels(['Southampton','Charlton','Queenstown'])
deck=train_df['Cabin']
deck.isna().value_counts()
deck=deck.dropna()
levels=[]
for level in deck:
    levels.append(level[0])
    
cabin_df=pd.DataFrame(levels,columns=['Cabin level'])
cabin_df.head()
cabin_df['Cabin level'].value_counts().sort_values(ascending=False)
sns.catplot('Cabin level',data=cabin_df,kind='count',aspect=2,height=6,palette='summer_d')
train_temp=train_df.copy()
train_temp.isna().any()
train_temp=train_temp.dropna(axis=0)
train_temp.reset_index(inplace=True,drop=True)
train_temp['Level']=cabin_df['Cabin level']
sns.catplot('Level',kind='count',hue='Person',aspect=2,height=6,data=train_temp)
sns.catplot('Level',kind='count',hue='Survived',aspect=2,height=6,data=train_temp)
sns.factorplot('Survived',
               data=train_temp,
               kind='count',col='Level',
               col_wrap=4,height=4,aspect=1)
train_df['Pclass'].value_counts()
sns.catplot('Pclass',data=train_df, kind='count')
sns.catplot('Pclass',data=train_df,kind='count',hue='Survived',aspect=2,height=6,palette='winter')
sns.lmplot('Age','Target',data=train_df,aspect=2,height=6,hue='Pclass')

sns.catplot('Pclass','Target',data=train_df,hue='Person',kind='point',aspect=2,height=6)

train_df['SibSp'].value_counts()
sns.catplot('SibSp',data=train_df,kind='count')
sns.catplot('SibSp',data=train_df,kind='count',hue='Target',aspect=2,height=6)
sns.lmplot('SibSp','Target',data=train_df)
train_df['Sex'].value_counts()
sns.catplot('Sex',data=train_df,kind='count',hue='Survived')
sns.lmplot('Age','Target',data=train_df,hue='Sex',aspect=2,height=6)
train_df['Parch'].unique()
train_df['Parch'].value_counts()
sns.catplot('Parch',data=train_df,kind='count')
sns.catplot('Parch',data=train_df,kind='count',hue='Survived')
train_df['Total relatives']=train_df['Parch']+train_df['SibSp']
ax=sns.catplot('Total relatives','Target',kind='point',data=train_df,aspect=2,height=6)
ax.set_ylabels('Survival probability')
correlations=train_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlations,annot=True,cmap='summer')
train_mod=train_df.copy()
train_mod.columns.isna()
train_mod.drop(['PassengerId','Survived','Sex','Name','Ticket','Cabin','Parch','SibSp'],axis=1,inplace=True)
train_mod.head()
train_mod['Person']=train_mod['Person'].map({'male':1,'female':2,'child':3})
train_mod.head()
temp=pd.get_dummies(train_mod['Embarked'])
train_mod=train_mod.merge(temp,on=train_mod.index)
train_mod.head()
train_mod.drop(['key_0','Embarked'],axis=1,inplace=True)
train_mod.head()
train_mod.loc[train_mod['Age']<=16,'Age band']=0
train_mod.loc[(train_mod['Age']>16) & (train_mod['Age']<33),'Age band']=1
train_mod.loc[(train_mod['Age']>32) & (train_mod['Age']<49),'Age band']=2
train_mod.loc[(train_mod['Age']>48) & (train_mod['Age']<65),'Age band']=3
train_mod.loc[train_mod['Age']>64,'Age band']=4
train_mod.head()
plt.figure(figsize=(10,8))
plt.boxplot(train_df['Fare'])
plt.ylabel('Fare value')


plt.figure(figsize=(10,7))
sns.kdeplot(train_mod['Fare'],shade=True)
train_mod.loc[(train_mod['Fare']<51),'Fare band']=1
train_mod.loc[(train_mod['Fare']>50)&(train_mod['Fare']<101),'Fare band']=2
train_mod.loc[(train_mod['Fare']>100)&(train_mod['Fare']<201),'Fare band']=3
train_mod.loc[(train_mod['Fare']>200),'Fare band']=4
train_mod.head()
train_mod.drop('Fare',axis=1,inplace=True)
target_df=pd.DataFrame(columns=['Target'])
target_df['Target']=train_mod['Target']
target_df['Target'].value_counts()
train_mod.drop('Target',axis=1,inplace=True)
train_mod.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_mod,target_df,test_size=0.2,shuffle=True,random_state=365)


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)
print('Score with KNN on test dataset:{}'.format(np.round(knn.score(X_test,y_test) *100,2)))
print('Score with KNN on train dataset:{}'.format(np.round(knn.score(X_train,y_train) *100,2)))
from sklearn.metrics import confusion_matrix
cnf_knn=confusion_matrix(y_knn_pred,y_test)
sns.heatmap(cnf_knn,annot=True,cmap='winter')
from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()
reg_log.fit(X_train,y_train)
y_log_pred=reg_log.predict(X_test)
print('Score with Logistic regression on test dataset:{}'.format(np.round(reg_log.score(X_test,y_test) *100,2)))
print('Score with Logistic regression on train dataset:{}'.format(np.round(reg_log.score(X_train,y_train) *100,2)))
cnf_reg=confusion_matrix(y_test,y_log_pred)
sns.heatmap(cnf_reg,annot=True,cmap='gnuplot')
y_lr=reg_log.fit(X_train,y_train).decision_function(X_test)
from sklearn.metrics import roc_curve,auc,precision_recall_curve

fpr,tpr,_=roc_curve(y_test,y_lr)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],linestyle='--')
auc_reg=auc(fpr,tpr).round(2)
plt.title('ROC curve with AUC={}'.format(auc_reg))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
precision,recall,threshold=precision_recall_curve(y_test,y_lr)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with Logistic Regression')
plt.xlabel('Precision')
plt.ylabel('Recall')
from sklearn.svm import SVC
svc=SVC(gamma=1e-07,C=1e9)
svc.fit(X_train,y_train)
y_svc_pred=svc.predict(X_test)
print('Score with SVC on test dataset:{}'.format(np.round(svc.score(X_test,y_test) *100,2)))
print('Score with SVC on train dataset:{}'.format(np.round(svc.score(X_train,y_train) *100,2)))
cnf_reg=confusion_matrix(y_test,y_svc_pred)
sns.heatmap(cnf_reg,annot=True,cmap='summer',fmt='g')
y_svc=svc.fit(X_train,y_train).decision_function(X_test)
fpr,tpr,_=roc_curve(y_test,y_svc)
plt.plot(fpr,tpr,color='indianred')
plt.plot([0,1],[0,1],linestyle='--')
auc_reg=auc(fpr,tpr).round(2)
plt.title('ROC curve with AUC={}'.format(auc_reg))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


precision,recall,threshold=precision_recall_curve(y_test,y_svc)
closest_zero=np.argmin(np.abs(threshold))
closest_zero_p=precision[closest_zero]
closest_zero_r = recall[closest_zero]
plt.plot(precision,recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.title('Precision-Recall curve with SVC')
plt.xlabel('Precision')
plt.ylabel('Recall')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc=RandomForestClassifier()
param_grid={'n_estimators':[5,7,9,10], 'max_depth':[5,7,9,10]}
grid_search=GridSearchCV(rfc,param_grid,scoring='roc_auc')
X_train.drop('Age',axis=1,inplace=True)
X_train.head()
grid_result=grid_search.fit(X_train,y_train)
grid_result.best_params_
grid_result.best_score_
X_test.drop('Age',axis=1,inplace=True)

y_rfc_pred=grid_result.predict(X_test)
print('Score with RFC on test dataset:{}'.format(np.round(grid_result.score(X_test,y_test) *100,2)))
print('Score with RFC on train dataset:{}'.format(np.round(grid_result.score(X_train,y_train) *100,2)))
cnf_rfc=confusion_matrix(y_test,y_rfc_pred)
sns.heatmap(cnf_rfc,annot=True,fmt='g')
from sklearn.model_selection import cross_val_score
rfc_opt=RandomForestClassifier(max_depth=5,n_estimators=9)
score_cv=cross_val_score(rfc_opt,X_train,y_train,cv=5,scoring='accuracy')
cv_df=pd.DataFrame(columns=['Cross validated score'])
cv_scores=np.round(score_cv*100,2)
cv_df['Cross validated score']=cv_scores
cv_df.index=cv_df.index + 1
cv_df
print('Cross validated mean score: {}'.format(cv_scores.mean()))
print('Cross validated score standard deviation: {}'.format(np.round(cv_scores.std(),2)))
test_df=pd.read_csv('../input/titanic/test.csv')
test_df.head()

test_df.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)
train_df.head()
test_df['Total relatives']=test_df['SibSp']+test_df['Parch']
test_df.drop(['SibSp','Parch'],axis=1,inplace=True)
embarks=pd.get_dummies(test_df['Embarked'])
test_df=test_df.merge(embarks,on=test_df.index)
test_df.drop(['key_0','Embarked'],axis=1,inplace=True)
test_df.head()
test_df['Person']=test_df[['Age','Sex']].apply(child,axis=1)
test_df.head()
test_df['Person']=test_df['Person'].map({'male':1,'female':2,'child':3})
test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())
test_df.loc[test_df['Age']<=16,'Age band']=0
test_df.loc[(test_df['Age']>16) & (test_df['Age']<33),'Age band']=1
test_df.loc[(test_df['Age']>32) & (test_df['Age']<49),'Age band']=2
test_df.loc[(test_df['Age']>48) & (test_df['Age']<65),'Age band']=3
test_df.loc[test_df['Age']>64,'Age band']=4
test_df.head()
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())
test_df.isna().any()
test_df.loc[(test_df['Fare']<51),'Fare band']=1
test_df.loc[(test_df['Fare']>50)&(test_df['Fare']<101),'Fare band']=2
test_df.loc[(test_df['Fare']>100)&(test_df['Fare']<201),'Fare band']=3
test_df.loc[(test_df['Fare']>200),'Fare band']=4
test_df.head()
test_df.drop(['Sex','Age','Fare'],axis=1,inplace=True)
test_df.head()
train_mod.head()

test_df[train_mod.columns].head()
rfc_opt.fit(X_train,y_train)
y_final_predictions=rfc_opt.predict(test_df[train_mod.columns])
final_predictions_df=pd.DataFrame(columns=['PassengerId','Survived'])
final_predictions_df['PassengerId']=test_df['PassengerId']
final_predictions_df['Survived']=y_final_predictions
final_predictions_df.isna().any()
final_predictions_df
