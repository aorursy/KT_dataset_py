import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

pd.set_option('display.max_columns', 50)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combined = pd.concat([train,test],axis=0,ignore_index=True)

combined.shape
combined.sort_values(['Fare','PassengerId'], ascending=[0,1]).head()
combined.isnull().sum()
fig,ax=plt.subplots(1,2,figsize=(15,5))

Mean=sns.barplot(x='Pclass',y='Fare',data=combined,estimator=np.mean,ax=ax[0]).set(ylabel='meanFare',title='MEAN_FARE')

Median=sns.barplot(x='Pclass',y='Fare',data=combined,estimator=np.median,ax=ax[1]).set(ylabel='medianFare',title='MEDIAN_FARE')



combined['Fare'] = combined['Fare'].groupby(combined['Pclass']).apply(lambda x: x.fillna(x.median()))
combined['Fare_per_person'] = combined['Fare'].groupby(combined['Ticket']).transform(lambda x: x / x.count())

combined.sort_values(['Fare','PassengerId'], ascending=[0,1]).head()
combined['Ticket_cnt'] = combined['Ticket'].groupby(combined['Ticket']).transform(lambda x: x.count())

fig, ax = plt.subplots(figsize=(10,5))

Cabin_plot=sns.barplot(x='Ticket_cnt',y='Survived',data=combined).set(ylabel='Survival_Probability',title='TICKET_COUNT_SURVIVAL')
combined['Cabin'] = combined['Cabin'].groupby(combined['Ticket']).apply(lambda x: x.fillna(x.astype(str).str[0].mode()[0])).apply(lambda x: x[0]).apply(lambda x: 'U' if(x=='n') else x)

combined.sort_values(['Fare','PassengerId'], ascending=[0,1]).head()
fig, ax = plt.subplots(figsize=(10,5))

Cabin_plot=sns.barplot(x='Cabin',y='Survived',data=combined,order='ABCDEFGTU').set(ylabel='Survival_Probability',title='CABIN_SURVIVAL')
combined['Cabin_survived'] = combined['Cabin'].apply(lambda x: 1 if(x in ['B','C','D','E','F']) else 0)
fig,ax=plt.subplots(1,2,figsize=(15,5))

People_count=sns.countplot(x='Embarked',data=combined,ax=ax[0]).set(ylabel='People_count',title='PEOPLE_COUNT')

Embarked_survival=sns.barplot(x='Embarked',y='Survived',data=combined,ax=ax[1]).set(ylabel='Survival_Probability',title='EMBARKED_SURVIVAL')



combined['Embarked'].fillna(combined['Embarked'].mode()[0],inplace=True)
combined['Name_len'] = combined['Name'].apply(lambda x: len(x)).astype(int)

fig, ax = plt.subplots(figsize=(20,7))

Namelen_survival = sns.barplot(x='Name_len',y='Survived',data=combined).set(ylabel='Survival_Probability',title='NAME_LENGTH_SURIVAL')
combined['Title'] = combined['Name'].apply(lambda x: x.split(', ')[1].split('.')[0].strip())

fig, ax = plt.subplots(figsize=(20,7))

order_list = ['Rev','Col','Major','Don','Capt','Sir','Jonkheer','Dr','Ms','Mlle','Mme','Lady','Dona']

Title_Sex=sns.countplot(x='Title',hue='Sex',data=combined,order=order_list).set(ylabel='Person_count',title='TITLE respective to SEX (ignoring Mr,Master,Miss,Mrs)')
combined.loc[(combined['Title']=='Dr') & (combined['Sex']=='female'),'Title'] = 'Mrs'

title_mapping = {'Capt':'Mr', 'Col':'Mr','Don':'Mr','Dona':'Mrs',

                 'Dr':'Mr','Jonkheer':'Mr','Lady':'Mrs','Major':'Mr',

                 'Master':'Master','Miss':'Miss','Mlle':'Miss','Mme':'Mrs',

                 'Mr':'Mr','Mrs':'Mrs','Ms':'Miss','Rev':'Mr','Sir':'Mr',

                 'the Countess':'Mrs'}

combined['Title'] = combined['Title'].map(title_mapping)
fig, ax = plt.subplots(figsize=(15,5))

Title_Age = sns.boxplot(x='Age',y='Title',data=combined).set(title='TITLE vs AGE')
combined.loc[(combined['Title']=='Miss') & (combined['Age']<=14.0),'Title'] = 'Girl'

fig, ax = plt.subplots(figsize=(10,5))

Title_survival=sns.barplot(x='Title',y='Survived',data=combined).set(ylabel='Survival_Probability',title='TITLE_SURVIVAL')
fig,ax=plt.subplots(1,2,figsize=(15,5))

SibSp_survival=sns.barplot(x='SibSp',y='Survived',data=combined,ax=ax[0]).set(ylabel='Survival_Probability',title='SIBSP_SURVIVAL')

Parch_survival=sns.barplot(x='Parch',y='Survived',data=combined,ax=ax[1]).set(ylabel='Survival_Probability',title='PARCH_SURVIVAL')
combined['Family'] = combined['SibSp'] + combined['Parch'] + 1

fig, ax = plt.subplots(figsize=(10,5))

Family_survival=sns.barplot(x='Family',y='Survived',data=combined).set(ylabel='Survival_Probability',title='FAMILY_SURVIVAL')
combined['IsAlone'] = combined['Family'].apply(lambda x: 1 if(x==1) else 0)
fig,ax=plt.subplots(1,2,figsize=(15,5))

Title_Age=sns.barplot(x='Title',y='Age',data=combined,ax=ax[0]).set(ylabel='Age',title='AGE vs TITLE')

Pclass_Age=sns.barplot(x='Pclass',y='Age',data=combined,ax=ax[1]).set(ylabel='Age',title='AGE vs PCLASS')

fig,ax=plt.subplots(figsize=(15,5))

Both_Age=sns.barplot(x='Title',y='Age',hue='Pclass',data=combined,estimator=np.median).set(ylabel='Age',title='AGE vs (Title,Pclass)')
combined['Age'] = combined.groupby(['Title','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
sex_mapping = {'male':1,'female':0}

combined['Sex'] = combined['Sex'].map(sex_mapping)
combined['Age_grp'] = pd.cut(combined['Age'],5,labels=[1,2,3,4,5])

combined['Fare_grp'] = pd.qcut(combined['Fare_per_person'],5,labels=[1,2,3,4,5])

fig,ax=plt.subplots(1,2,figsize=(15,5))

Age_grp=sns.barplot(x='Age_grp',y='Survived',data=combined,ax=ax[0]).set(ylabel='Survival_Probability',title='AGE_GRP SURVIVAL')

Fare_grp=sns.barplot(x='Fare_grp',y='Survived',data=combined,ax=ax[1]).set(ylabel='Survival_Probability',title='FARE_GRP SURVIVAL')
dummy_features=['Age','Age_grp','Sex','Fare_grp','Fare_per_person','Ticket_cnt',

                'Cabin_survived','Survived','Embarked','Pclass','Title',

                'Parch','SibSp','Family','IsAlone']

combined_wo_dummies = combined[dummy_features]

combined = pd.get_dummies(combined[dummy_features])

train_features=combined.iloc[:891,:]

train_labels=train_features.pop('Survived').astype(int)

test_features=combined.iloc[891:,:].drop('Survived',axis=1)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
models=[KNeighborsClassifier(),LogisticRegression(),GaussianNB(),SVC(),DecisionTreeClassifier(),

        RandomForestClassifier(),GradientBoostingClassifier(),AdaBoostClassifier()]

names=['KNN','LR','NB','SVM','Tree','RF','GB','Ada']

for name,model in zip(names,models):

    score=cross_val_score(model,train_features,train_labels,cv=5)

    print('{} :: {} , {}'.format(name,score.mean(),score))
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

train_features_scaled=scaler.fit(train_features).transform(train_features)

test_features_scaled=scaler.fit(test_features).transform(test_features)

for name,model in zip(names,models):

    score=cross_val_score(model,train_features_scaled,train_labels,cv=5)

    print('{} :: {} , {}'.format(name,score.mean(),score))
GB_imp=GradientBoostingClassifier()

GB_imp.fit(train_features,train_labels)

features_imp = pd.DataFrame({'importance':GB_imp.feature_importances_},index=train_features.columns).sort_values('importance',ascending=True)

features_imp_plot = features_imp.plot(kind='barh',figsize=(15,10))
from sklearn.metrics import roc_auc_score

misclassified=[]

from sklearn.model_selection import KFold

kf = KFold(n_splits=10,random_state=42)

for i,(train_index,test_index) in enumerate(kf.split(train_features)):

    kf_x_train = train_features.loc[train_index]

    kf_y_train = train_labels[train_index]

    kf_x_test = train_features.loc[test_index]

    pred = GB_imp.fit(kf_x_train,kf_y_train).predict(kf_x_test)

    misclassified.append(train_labels[test_index][pred != train_labels[test_index]].index)

    print('roc_auc_score ',i,' : ',roc_auc_score(pred,train_labels[test_index]))

misclassified_index = np.concatenate(misclassified)

misclassified_index
len(misclassified_index)
misclassified_df = combined_wo_dummies.loc[misclassified_index]

misclassified_df.head()
misclassified_df.groupby(['Title','Pclass','Family'])['Survived'].agg(['mean','count'])
from sklearn.model_selection import GridSearchCV



train_features=combined.iloc[:891,:]

train_labels=train_features.pop('Survived').astype(int)

test_features=combined.iloc[891:,:].drop('Survived',axis=1)
HP_tuning = False
if(HP_tuning):

    parameter_grid = {'n_neighbors':[4,6,8,10,12], 'algorithm':['auto','ball_tree','kd_tree','brute']}

    KNN = GridSearchCV(KNeighborsClassifier(),parameter_grid,cv=5)

    KNN.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',KNN.best_params_,KNN.best_score_)
if(HP_tuning):

    parameter_grid = {'C':[0.01,0.1,1,10]}

    LR = GridSearchCV(LogisticRegression(),parameter_grid,cv=5)

    LR.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',LR.best_params_,LR.best_score_)
if(HP_tuning):

    parameter_grid = {'C':[0.1,1,10],'gamma':['auto',0.01,0.1,1,10]}

    SVM = GridSearchCV(SVC(),parameter_grid,cv=5)

    SVM.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',SVM.best_params_,SVM.best_score_)
if(HP_tuning):

    parameter_grid = {'min_samples_split':[2,3,4],'min_samples_leaf':[1,2,3],'max_depth':[None,2,3,4]}

    DT = GridSearchCV(DecisionTreeClassifier(),parameter_grid,cv=5)

    DT.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',DT.best_params_,DT.best_score_)
if(HP_tuning):

    parameter_grid = {'n_estimators':[100,110,120,130],'max_depth':[None,1,2,3]}

    RF = GridSearchCV(RandomForestClassifier(),parameter_grid,cv=5)

    RF.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',RF.best_params_,RF.best_score_)

    parameter_grid = {'min_samples_split':[2,3,4],'min_samples_leaf':[2,3,4]}

    RF = GridSearchCV(RandomForestClassifier(n_estimators=100,max_depth=None),parameter_grid,cv=5)

    RF.fit(train_features_scaled,train_labels)

    print('parameter_grid2 : ',RF.best_params_,RF.best_score_)
if(HP_tuning):

    parameter_grid = {'n_estimators':[110,120,130],'max_depth':[2,3,4]}

    GB = GridSearchCV(GradientBoostingClassifier(learning_rate=0.2),parameter_grid,cv=5)

    GB.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',GB.best_params_,GB.best_score_)

    parameter_grid = {'min_samples_split':[60,65,70],'min_samples_leaf':[1,2,3]}

    GB = GridSearchCV(GradientBoostingClassifier(n_estimators=120,max_depth=3,learning_rate=0.2),parameter_grid,cv=5)

    GB.fit(train_features_scaled,train_labels)

    print('parameter_grid2 : ',GB.best_params_,GB.best_score_)
if(HP_tuning):

    parameter_grid ={'n_estimators':[110,120,130],'learning_rate':[0.01,0.1,1,10],'random_state':[None,42]}

    Ada = GridSearchCV(AdaBoostClassifier(),parameter_grid,cv=5)

    Ada.fit(train_features_scaled,train_labels)

    print('parameter_grid1 : ',Ada.best_params_,Ada.best_score_)
KNN_clf = KNeighborsClassifier(algorithm='auto', n_neighbors=6)

LR_clf = LogisticRegression(C=0.1)

SVM_clf = SVC(C=1,gamma='auto',probability=True)

Tree_clf = DecisionTreeClassifier(max_depth=4,min_samples_leaf=1,min_samples_split=4)

RF_clf = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=2,min_samples_split=3)

GB_clf = GradientBoostingClassifier(n_estimators=120,max_depth=3,min_samples_leaf=3,min_samples_split=65,learning_rate=0.2)

Ada_clf = AdaBoostClassifier(learning_rate=0.1,n_estimators=130,random_state=None)



model_list = [('KNN',KNN_clf),('LR',LR_clf),('SVM',SVM_clf),('Tree',Tree_clf),('RF',RF_clf),('GB',GB_clf),('Ada',Ada_clf)]

from sklearn.ensemble import VotingClassifier

voteh_clf = VotingClassifier(estimators=model_list,voting='hard')

votehw_clf = VotingClassifier(estimators=model_list,voting='hard',weights=[1,1,2,1,2,2,1])

votes_clf = VotingClassifier(estimators=model_list,voting='soft')

votesw_clf = VotingClassifier(estimators=model_list,voting='soft',weights=[1,1,2,1,2,2,1])



names = ['KNN','LR','SVM','Tree','RF','GB','Ada','voteh','votehw','votes','votesw']

models = [KNN_clf,LR_clf,SVM_clf,Tree_clf,RF_clf,GB_clf,Ada_clf,voteh_clf,votehw_clf,votes_clf,votesw_clf]



for name,model in zip(names,models):

    score=cross_val_score(model,train_features_scaled,train_labels,cv=5)

    print('{} :: {} , {}'.format(name,score.mean(),score))
import numpy as np

n_train=train.shape[0]

n_test=test.shape[0]

NFOLDS = 10

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=NFOLDS,shuffle=True,random_state=0)
def get_oof(clf,X_train,Y_train,X_test):

    oof_train=np.zeros((n_train,))

    oof_test_mean=np.zeros((n_test,))

    oof_test_single=np.empty((NFOLDS,n_test))

    for i, (train_index,test_index) in enumerate(kf.split(X_train,Y_train)):

        kf_X_train=X_train[train_index]

        kf_Y_train=Y_train[train_index]

        kf_X_test=X_train[test_index]

        

        clf.fit(kf_X_train,kf_Y_train)

        

        oof_train[test_index]=clf.predict(kf_X_test)

        oof_test_single[i,:]=clf.predict(X_test)

    oof_test_mean=oof_test_single.mean(axis=0)

    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)
KNN_train,KNN_test= get_oof(KNN_clf, train_features_scaled,train_labels,test_features_scaled)

LR_train,LR_test= get_oof(LR_clf, train_features_scaled,train_labels,test_features_scaled)

SVM_train,SVM_test= get_oof(SVM_clf, train_features_scaled,train_labels,test_features_scaled)

Tree_train,Tree_test= get_oof(Tree_clf, train_features_scaled,train_labels,test_features_scaled)

RF_train,RF_test= get_oof(RF_clf, train_features_scaled,train_labels,test_features_scaled)

GB_train,GB_test= get_oof(GB_clf, train_features_scaled,train_labels,test_features_scaled)

Ada_train,Ada_test= get_oof(Ada_clf, train_features_scaled,train_labels,test_features_scaled)



X_train_stack = np.concatenate((KNN_train,LR_train,SVM_train,Tree_train,RF_train,Ada_train,GB_train),axis=1)         

X_test_stack = np.concatenate((KNN_test,LR_test,SVM_test,Tree_test,RF_test,Ada_test,GB_test),axis=1)
X_train_stack_df = pd.DataFrame(X_train_stack,columns=['KNN','LR','SVM','Tree','RF','GB','Ada'])

fig,ax=plt.subplots(figsize=(15,10))

Heatmap = sns.heatmap(X_train_stack_df.corr(),annot=True)
if(HP_tuning):

    from xgboost import XGBClassifier

    parameter_grid = {'n_estimators':[10,20,30],'max_depth':[1,2,3,4,5],'min_child_weight':[1,2,3]}

    XGB_clf = GridSearchCV(XGBClassifier(),parameter_grid,cv=5)

    XGB_clf.fit(X_train_stack,train_labels)

    print('parameter_grid1 : ',XGB_clf.best_params_,XGB_clf.best_score_)

    from xgboost import XGBClassifier

    parameter_grid = {'gamma':[0.01,0.1,0],'subsample':[0.1,0.5,1]}

    XGB_clf = GridSearchCV(XGBClassifier(n_estimators=20,max_depth=4,min_child_weight=2),parameter_grid,cv=5)

    XGB_clf.fit(X_train_stack,train_labels)

    print('parameter_grid2 : ',XGB_clf.best_params_,XGB_clf.best_score_)

    parameter_grid = {'learning_rate':[0.01,0.1,1,10]}

    XGB_clf = GridSearchCV(XGBClassifier(n_estimators=20,max_depth=4,min_child_weight=2,gamma=0.01,subsample=0.5),parameter_grid,cv=5)

    XGB_clf.fit(X_train_stack,train_labels)

    print('parameter_grid3 : ',XGB_clf.best_params_,XGB_clf.best_score_)
from xgboost import XGBClassifier

XGB_clf = XGBClassifier(n_estimators=20,learning_rate=0.1,max_depth=4,min_child_weight=2,gamma=0.01,subsample=0.5)

XGB_clf.fit(X_train_stack,train_labels)

pred = XGB_clf.predict(X_test_stack)

test_labels = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':list(map(int,pred))})

test_labels.to_csv('Titanic_solution.csv',index=False)
from IPython.display import HTML

import base64

def create_download_link( df, title = "Download CSV file", filename = "Titanic_solution.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(test_labels)
