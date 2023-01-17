import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train_data=pd.read_csv('../input/titanic/train.csv')
test_data=pd.read_csv('../input/titanic/test.csv')
train_data.head(10)
train_data.describe()
train_data.isna().any()
train_data.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
train_data['Embarked'].isna().value_counts()
train_data.isna().any()
imp=SimpleImputer(strategy='most_frequent')
train_data[['Embarked']]=imp.fit_transform(train_data[['Embarked']])
mean=train_data['Age'].mean()
train_data['Age'].fillna(mean,inplace=True)
train_data['Age']=np.ceil(train_data['Age'])
def get_age(val):
    age_classification={'Child':range(0,13),
                        'Teen':range(13,21),
                        'Young':range(21,31),
                        '30_adult':range(31,41),
                        '40_adult':range(41,51),
                        'Elderly':range(51,90)}
    for key,value in age_classification.items():
        if val in value:
            return key
for i in range(len(train_data)):
    train_data['Age'][i]=('{}'.format(get_age(train_data['Age'][i])))    
train_data['Name'].unique()
def get_title(val):
    words=val.split()
    title={'Officer':['Capt.','Col.','Major.','Dr.','Rev.'],
          'Royalty':['Jonkheer.','Don.','Sir.','the Countess.','Lady.'],
          'Mrs':['Mme.','Ms.','Mrs.'],
          'Mr':['Mr.'],
          'Miss':['Mlle.','Miss.'],
          'Master':['Master.']}
    for key,value in title.items():
        for word in words:
            if word in value:
                return str(key)
train_data['Title']=np.NAN
for i in range(len(train_data)):
    train_data['Title'][i]=get_title(train_data['Name'][i])
train_data['Title'].isna().value_counts()
train_data['Fam_mem']=train_data['SibSp']+train_data['Parch']+1
def fam_size(val):
    fam={'Single':[1],
        'Small_family':[2,3,4],
        'large_family':[5,6,7,8,9,10,11]}
    for key,value in fam.items():
        if val in value:
            return key
for i in range(len(train_data)):
    train_data['Fam_mem'][i]=fam_size(train_data['Fam_mem'][i])
train_data.head(10)
train_data.isna().sum()
def most_common(lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]
frequent=most_common(train_data['Title'])
train_data['Title'].fillna(frequent,inplace=True)
train_data['Title'].unique()
train_data.head(10)
oe=OrdinalEncoder()
train_data[['Embarked','Sex','Age','Title','Fam_mem']]=oe.fit_transform(train_data[['Embarked','Sex','Age','Title','Fam_mem']])
train_data.head(10)
train_data.hist(bins=10,figsize=(20,15))
corr_mat=train_data.corr()
corr_mat['Survived'].sort_values(ascending=False)
attributes=['Survived','Fare','Embarked','Pclass','Sex']
scatter_matrix(train_data[attributes],figsize=(15,10),alpha=0.1)
plt.scatter(train_data.iloc[:,1],train_data.iloc[:,3],c=train_data.iloc[:,0],s=50,cmap='RdBu')
men_survived_truth=(((train_data['Sex']==1)&(train_data['Survived']==1)))
men_death_truth=(((train_data['Sex']==1)&(train_data['Survived']==0)))
women_survived_truth=(((train_data['Sex']==0)&(train_data['Survived']==1)))
women_death_truth=(((train_data['Sex']==0)&(train_data['Survived']==0)))
men_survived=men_survived_truth.value_counts()
men_death=men_death_truth.value_counts()
women_survived=women_survived_truth.value_counts()
women_death=women_death_truth.value_counts()
men=[men_survived[1],men_death[1]]
women=[women_survived[1],women_death[1]]
men_ratio=[(men[0]/(men[0]+men[1]))*100,(men[1]/(men[0]+men[1]))*100]
print(men_ratio)
women_ratio=[(women[0]/(women[0]+women[1]))*100,(women[1]/(women[0]+women[1]))*100]
print(women_ratio)
fig,ax=plt.subplots(1,2,figsize=(15,10))
explode=(0.1,0)
ax[0].pie(men_ratio,explode=explode,labels=['Survivors','Deaths'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[0].set_title('Men Ratio')
ax[1].pie(women_ratio,explode=explode,labels=['Survivors','Deaths'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[1].set_title('Women Ratio')
men_survivors=[]
men_death=[]
for i in range(1,4):
    Pclassmen_survived_truth=(((train_data['Sex']==1)&(train_data['Survived']==1)&(train_data['Pclass']==i)))
    Pclassmen_death_truth=(((train_data['Sex']==1)&(train_data['Survived']==0)&(train_data['Pclass']==i)))
    pclassmen_survivors=Pclassmen_survived_truth.value_counts()
    pclassmen_deaths=Pclassmen_death_truth.value_counts()
    men_survivors.append(pclassmen_survivors[1])
    men_death.append(pclassmen_deaths[1])
men=[men_survivors,men_death]
women_survivors=[]
women_death=[]
for i in range(1,4):
    Pclasswomen_survived_truth=(((train_data['Sex']==0)&(train_data['Survived']==1)&(train_data['Pclass']==i)))
    Pclasswomen_death_truth=(((train_data['Sex']==0)&(train_data['Survived']==0)&(train_data['Pclass']==i)))
    pclasswomen_survivors=Pclasswomen_survived_truth.value_counts()
    pclasswomen_deaths=Pclasswomen_death_truth.value_counts()
    women_survivors.append(pclasswomen_survivors[1])
    women_death.append(pclasswomen_deaths[1])
women=[women_survivors,women_death]
print(men)
print(women)
fig,ax=plt.subplots(1,2,figsize=(15,10))
sur_death=['Survived','Death']
width=0.5
for i,axi in enumerate(ax.flat):
    N=3
    ind=[x for x in np.arange(1,N+1)]
    axi.bar(ind,women[i],width,label='Women',bottom=men[i],color='Pink')
    axi.bar(ind,men[i],width,label='Men',color='Blue')
    axi.set_xticklabels(['0','Pclass 1','','Pclass 2','','Pclass 3'])
    axi.set_title(sur_death[i])
    axi.legend()
sns.pairplot(train_data)
survivors=[]
death=[]
for i in range(0,6):
    title_survived_truth=(((train_data['Survived']==1)&(train_data['Title']==i)))
    title_death_truth=(((train_data['Survived']==0)&(train_data['Title']==i)))
    title_survived=title_survived_truth.value_counts()
    title_death=title_death_truth.value_counts()
    survivors.append(title_survived[1])
    death.append(title_death[1])
title_sur_death=[survivors,death]
title_sur_death
oe.categories_
fig,ax=plt.subplots(1,2,figsize=(15,10))
to_plot=[survivors,death]
name=['Survivors','Deaths']
color=['green','red']
for i,axi in enumerate(ax.flat):
    N=6
    ind=[x for x in np.arange(1,N+1)]
    axi.bar(ind,to_plot[i],width,label=('{}'.format(name[i])),color=color[i])
    axi.set_xticklabels(['0','Master','Miss','Mr','Mrs','Officer','Royalty'])
    axi.set_title('{} on basis of Title'.format(name[i]))
    axi.legend()
survivors=[]
death=[]
for i in range(0,6):
    age_survived_truth=(((train_data['Survived']==1)&(train_data['Age']==i)))
    age_death_truth=(((train_data['Survived']==0)&(train_data['Age']==i)))
    age_survived=age_survived_truth.value_counts()
    age_death=age_death_truth.value_counts()
    survivors.append(age_survived[1])
    death.append(age_death[1])
age_sur_death=[survivors,death]
age_sur_death
fig,ax=plt.subplots(1,2,figsize=(15,10))
to_plot=[survivors,death]
name=['Survivors','Deaths']
color=['green','red']
for i,axi in enumerate(ax.flat):
    N=6
    ind=[x for x in np.arange(1,N+1)]
    axi.bar(ind,to_plot[i],width,label=('{}'.format(name[i])),color=color[i])
    axi.set_xticklabels(['0','30_adult','40_adult','Child','Elderly','Teen','Young'])
    axi.set_title('{} on basis of Age'.format(name[i]))
    axi.legend()
embarked_survivor=[]
embarked_death=[]
for i in range(3):
    embarked_survived_truth=(((train_data['Survived']==1)&(train_data['Embarked']==i)))
    embarked_death_truth=(((train_data['Survived']==0)&(train_data['Embarked']==i)))
    embarked_survivors=embarked_survived_truth.value_counts()
    embarked_deaths=embarked_death_truth.value_counts()
    embarked_survivor.append(embarked_survivors[1])
    embarked_death.append(embarked_deaths[1])
embarked=[embarked_survivor,embarked_death]
embarked
fig=plt.figure(figsize=(15,10))
N=3
ind=[x for x in np.arange(1,N+1)]
plt.bar(ind,embarked_survivor,width,label='Survived',bottom=embarked_death,color='Orange')
plt.bar(ind,embarked_death,width,label='Death',color='cyan')
plt.xticks(ind,['C','Q','S'])
plt.title('Survive/death on basis of Embarked')
plt.legend()
survivors=[]
death=[]
for i in range(0,3):
    fam_survived_truth=(((train_data['Survived']==1)&(train_data['Fam_mem']==i)))
    fam_death_truth=(((train_data['Survived']==0)&(train_data['Fam_mem']==i)))
    fam_survived=fam_survived_truth.value_counts()
    fam_death=fam_death_truth.value_counts()
    survivors.append(fam_survived[1])
    death.append(fam_death[1])
fam_sur_death=[survivors,death]
fam_sur_death
sur_ratio=[(survivors[0]/(survivors[0]+survivors[1]+survivors[2]))*100,(survivors[1]/(survivors[0]+survivors[1]+survivors[2]))*100,
          (survivors[2]/(survivors[0]+survivors[1]+survivors[2]))*100]
death_ratio=[(death[0]/(death[0]+death[1]+death[2]))*100,(death[1]/(death[0]+death[1]+death[2]))*100,
            (death[2]/(death[0]+death[1]+death[2]))*100]
print(sur_ratio)
print(death_ratio)
fig,ax=plt.subplots(1,2,figsize=(15,10))
explode=(0.1,0.1,0.1)
ax[0].pie(sur_ratio,explode=explode,labels=['Single','Small_family','Large_family'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[0].set_title('Survive Ratio')
ax[1].pie(death_ratio,explode=explode,labels=['Single','Small_family','Large_family'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[1].set_title('Death Ratio')
train_data.head(10)
preprocess_data=train_data.copy()
preprocess_data=pd.get_dummies(preprocess_data,columns=['Pclass','Sex','Age','Embarked','Title','Fam_mem'])
preprocess_data.head(10)
corr_mat=preprocess_data.corr()
corr_mat['Survived'].sort_values(ascending=False)
preprocess_data.drop(['Age_0.0','Title_5.0','Embarked_1.0','Age_1.0','Age_4.0','Age_3.0','Title_4.0','SibSp','Age_5.0',
                     'Parch','Title_0.0','Pclass_2','Name'],axis=1,inplace=True)
preprocess_data.shape
scale=StandardScaler()
preprocess_data[['Fare']]=scale.fit_transform(preprocess_data[['Fare']])
preprocess_data.head()
preprocess_feature=preprocess_data.drop('Survived',axis=1)
preprocess_label=preprocess_data['Survived']
preprocess_feature
preprocess_label
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(preprocess_data,preprocess_data['Survived']):
    strat_preprocess_data=preprocess_data.loc[train_index]
    strat_preprocess_label=preprocess_data.loc[test_index]
X_train=strat_preprocess_data.drop('Survived',axis=1)
y_train=strat_preprocess_data['Survived']
X_test=strat_preprocess_label.drop('Survived',axis=1)
y_test=strat_preprocess_label['Survived']
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
tree=DecisionTreeClassifier(random_state=42,criterion='entropy',splitter='best')
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)
accuracy_score(y_test,y_pred)
param_grid={'max_depth':[2,3,4,5,6]}
grid_tree=GridSearchCV(tree,param_grid,cv=5)
grid_tree.fit(X_train,y_train)
y_pred=grid_tree.best_estimator_.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_tree.best_params_)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
accuracy_score(y_test,y_pred)
param_grid={'C':[0.0001,0.001,0.01,0.1,0.5],'penalty':['l1','l2']}
grid_log=GridSearchCV(log_reg,param_grid,cv=5)
grid_log.fit(X_train,y_train)
y_pred=grid_log.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_log.best_params_)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
rfc=RandomForestClassifier(criterion='entropy',random_state=42)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
accuracy_score(y_test,y_pred)
param_grid={'max_depth':[2,3,4,5,6],'n_estimators':[100,200,300,400,500]}
grid_forest=GridSearchCV(rfc,param_grid,cv=5)
grid_forest.fit(X_train,y_train)
y_pred=grid_forest.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_forest.best_params_)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False,fmt='d')
plt.xlabel('true label')
plt.ylabel('predicted label')
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy_score(y_test,y_pred)
param_grid={'penalty':['l1','l2'],'alpha':[0.0001,0.001,0.01,0.1],'max_iter':[1000,1500,2000,2500]}
grid_sgd=GridSearchCV(sgd_clf,param_grid,cv=5)
grid_sgd.fit(X_train,y_train)
y_pred=grid_sgd.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_sgd.best_params_)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
svc_clf=SVC(kernel='rbf',random_state=42)
svc_clf.fit(X_train,y_train)
y_pred=svc_clf.predict(X_test)
accuracy_score(y_test,y_pred)
param_grid={'C':[1,2,3,4,5],'degree':[2,3,4,5,6],'gamma':['scale','auto']}
grid_svc=GridSearchCV(svc_clf,param_grid,cv=5)
grid_svc.fit(X_train,y_train)
y_pred=grid_svc.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_svc.best_params_)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
accuracy_score(y_test,y_pred)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)
param_grid={'n_neighbors':[4,5,6,7,8],'algorithm':['ball_tree','kd_tree','brute'],'p':[1,2]}
grid_knn=GridSearchCV(knn,param_grid,cv=5)
grid_knn.fit(X_train,y_train)
y_pred=grid_knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_knn.best_params_)
mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
def display_scores(scores):
    print('Scores:',scores)
    print('Mean:',scores.mean())    
    print('Standard Deviation:',scores.std()) 
svc_clf=SVC(kernel='rbf',random_state=42,C=2,degree=2,gamma='auto')
scores=cross_val_score(svc_clf,X_train,y_train,cv=3,scoring='accuracy')
display_scores(scores)
y_scores=cross_val_predict(svc_clf,X_train,y_train,method='decision_function')
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
precisions,recalls,thresholds=precision_recall_curve(y_train,y_scores)
threshold_90_precision=thresholds[np.argmax(precisions>=0.90)]
y_train_pred_90=(y_scores>=threshold_90_precision)
precision_score(y_train,y_train_pred_90)
recall_score(y_train,y_train_pred_90)
threshold_90_precision
svc_clf=SVC(kernel='rbf',random_state=42,C=2,degree=2,gamma='auto')
scores=cross_val_score(svc_clf,preprocess_feature,preprocess_label,cv=3,scoring='accuracy')
display_scores(scores)
test_data.head()
test_data.describe()
test_data.isna().any()
def get_age(val):
    age_classification={'Child':range(0,13),
                        'Teen':range(13,21),
                        'Young':range(21,31),
                        '30_adult':range(31,41),
                        '40_adult':range(41,51),
                        'Elderly':range(51,90)}
    for key,value in age_classification.items():
        if val in value:
            return key
def get_title(val):
    words=val.split()
    title={'Officer':['Capt.','Col.','Major.','Dr.','Rev.'],
          'Royalty':['Jonkheer.','Don.','Sir.','the Countess.','Lady.'],
          'Mrs':['Mme.','Ms.','Mrs.'],
          'Mr':['Mr.'],
          'Miss':['Mlle.','Miss.'],
          'Master':['Master.']}
    for key,value in title.items():
        for word in words:
            if word in value:
                return str(key)
def fam_size(val):
    fam={'Single':[1],
        'Small_family':[2,3,4],
        'large_family':[5,6,7,8,9,10,11]}
    for key,value in fam.items():
        if val in value:
            return key
def most_common(lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]
from sklearn.base import BaseEstimator,TransformerMixin
class CombinedWorks(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.preprocessing import StandardScaler
        oe=OrdinalEncoder()
        scale=StandardScaler()
        mean_age=X['Age'].mean()
        mean_fare=X['Fare'].mean()
        X['Age'].fillna(mean_age,inplace=True)
        X['Fare'].fillna(mean_fare,inplace=True)
        PassengerId=X['PassengerId']
        X.drop(['PassengerId','Cabin','Ticket'],axis=1,inplace=True)
        X['Title']=np.NAN
        X['Age']=np.ceil(X['Age'])
        X['Fam_mem']=X['SibSp']+X['Parch']+1
        for i in range(len(X)):
            X['Age'][i]=('{}'.format(get_age(X['Age'][i])))  
            X['Title'][i]=get_title(X['Name'][i])
            X['Fam_mem'][i]=fam_size(X['Fam_mem'][i])
        frequent=most_common(X['Title'])
        X['Title'].fillna(frequent,inplace=True)
        X[['Embarked','Sex','Age','Title','Fam_mem']]=oe.fit_transform(X[['Embarked','Sex','Age','Title','Fam_mem']])
        X=pd.get_dummies(X,columns=['Pclass','Sex','Age','Embarked','Title','Fam_mem'])
        X[['Fare']]=scale.fit_transform(X[['Fare']])
        name=X['Name']
        X.drop(['SibSp','Parch','Name'],axis=1,inplace=True)
        return X,name,oe.categories_,PassengerId
cw=CombinedWorks()
test_data,Name,catagories,passenger_id=cw.fit_transform(test_data)
test_data.head()
catagories
test_data.drop(['Age_0.0','Embarked_1.0','Age_1.0','Age_4.0','Age_3.0','Title_4.0','Age_5.0',
                'Title_0.0','Pclass_2'],axis=1,inplace=True)
test_data.shape
svc_clf=SVC(kernel='rbf',random_state=42,C=2,degree=2,gamma='auto')
svc_clf.fit(preprocess_feature,preprocess_label)
y_pred=svc_clf.predict(test_data)
y_pred
prediction=np.c_[Name,y_pred]
for i in range(len(test_data)):
    if prediction[i][1]==1:
        prediction[i][1]='Yes'
    else:
        prediction[i][1]='No'
for i in range(len(test_data)):
    print('Name:{0}\t\tSurvived:{1}'.format(prediction[i][0],prediction[i][1]))