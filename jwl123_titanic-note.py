import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)

import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_train=pd.read_csv('../input/titanic/train.csv')
df_test=pd.read_csv('../input/titanic/test.csv')
for col in df_train.columns:
    msg='column: {:>10}\t percent of NaN value: {:.2f}%'.format(col,100*(df_train[col].isnull().sum()/df_train[col].shape[0]))
    print(msg)
for col in df_test.columns:
    msg='column: {:>10}\t percent of NaN value: {:.2f}%'.format(col,100*(df_test[col].isnull().sum()/df_test[col].shape[0]))
    print(msg)
msno.matrix(df=df_train.iloc[:,:],figsize=(8,8),color=(0.8,0.5,0.2 ))
msno.bar(df=df_train.iloc[:,:],figsize=(8,8),color=(0.8,0.5,0.2 ))
f,ax=plt.subplots(1,2,figsize=(18,8))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Pie plot -Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1])
ax[1].set_title('count plot -survived')
plt.show()
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).count()
pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True)
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar()
y_position=1.02
f,ax=plt.subplots(1,2,figsize=(18,8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of passengers by pclass',y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Pclass"Survived vs Dead',y=y_position)
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
df_train[['Sex','Survived']].groupby(['Sex'],as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[df_train['Survived']==1]['Age'],ax=ax)
sns.kdeplot(df_train[df_train['Survived']==0]['Age'],ax=ax)
plt.legend(['Survived==1','Survived==0'])
plt.show()
plt.figure(figsize=(8,6))
df_train['Age'][df_train['Pclass']==1].plot(kind='kde')
df_train['Age'][df_train['Pclass']==2].plot(kind='kde')
df_train['Age'][df_train['Pclass']==3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class','2ndClass','3rd class'])
fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==1)]['Age'],ax=ax)
sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==1)]['Age'],ax=ax)
plt.legend(['Survived==0','Survived==1'])
plt.title('1st class')
plt.show()
fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==2)]['Age'],ax=ax)
sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==2)]['Age'],ax=ax)
plt.legend(['Survived==0','Survived==1'])
plt.title('2nd class')
plt.show()
fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==3)]['Age'],ax=ax)
sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==3)]['Age'],ax=ax)
plt.legend(['Survived==0','Survived==1'])
plt.title('3rd class')
plt.show()
change_age_range_survival_ratio=[]

for i in range(1,80):
    change_age_range_survival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum()/len(df_train[df_train['Age']<i]['Survived']))
    
plt.figure(figsize=(7,7))
plt.plot(change_age_range_survival_ratio)
plt.title('Survival rate change depending on range of Age',y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range og age')
plt.show()
                           
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=df_train,scale='count',split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot('Sex','Age',hue='Survived',data=df_train,scale='count',split=True,ax=ax[1])
ax[0].set_title('Sex and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

plt.show()
f,ax=plt.subplots(1,1,figsize=(7,7))
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=df_train,ax=ax[0,0])
ax[0,0].set_title('(1) No. of passengers boared')
sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])
ax[0,1].set_title('(2) Male-Femail split for embarked')
sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs survived')
sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
df_train['FamilySize']=df_train['SibSp']+df_train['Parch']+1
df_test['FamilySize']=df_test['SibSp']+df_test['Parch']+1
f,ax=plt.subplots(1,3,figsize=(40,10))
sns.countplot('FamilySize',data=df_train,ax=ax[0])
ax[0].set_title('(1)No. of Passenger Boarded',y=1.02)

sns.countplot('FamilySize',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('(2) survived countplot depending on FamilySize',y=1.02)

df_train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) survived rate depending on FamilySize',y=1.02)

plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
fig,ax=plt.subplots(1,1,figsize=(8,8))
g=sns.distplot(df_train['Fare'],color='b',label='Skewness:{:.2f}'.format(df_train['Fare'].skew()),ax=ax)
g=g.legend(loc='best')
df_train['Initial']=df_train['Name'].str.extract('([A-Za-z]+)\.')
df_test['Initial']=df_test['Name'].str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train['Initial'].replace(['Mile','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                           ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'],inplace=True)

df_test['Initial'].replace(['Mile','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                           ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'],inplace=True)
df_train.groupby('Initial').mean()
df_train.groupby('Initial')['Survived'].mean().plot.bar()
df_all=pd.concat([df_train,df_test])
df_all.groupby('Initial').mean()
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Mr'),'Age']=33
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Mrs'),'Age']=37
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Master'),'Age']=5
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Miss'),'Age']=22
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Other'),'Age']=45

df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Mr'),'Age']=33
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Mrs'),'Age']=37
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Master'),'Age']=5
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Miss'),'Age']=22
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Other'),'Age']=45
df_train['Embarked'].fillna('S',inplace=True)
df_test['Embarked'].fillna('S',inplace=True)
def category_age(x):
    if x<10:
        return 0
    elif x<20:
        return 1
    elif x<30:
        return 2
    elif x<40:
        return 3
    elif x<50:
        return 4
    elif x<60:
        return 5
    elif x<70:
        return 6
    else:
        return 7
df_train['Age_cat']=df_train['Age'].apply(category_age)
df_train.drop(['Age'],axis=1,inplace=True)
df_test['Age_cat']=df_test['Age'].apply(category_age)
df_test.drop(['Age'],axis=1,inplace=True)
df_train['Initial']=df_train['Initial'].map({'Master':0,'Miss':1,'Mr':2,'Mrs:':3,'Other':4})
df_test['Initial']=df_test['Initial'].map({'Master':0,'Miss':1,'Mr':2,'Mrs:':3,'Other':4})
df_train['Embarked']=df_train['Embarked'].map({'C':0,'Q':1,'S':2})
df_test['Embarked']=df_test['Embarked'].map({'C':0,'Q':1,'S':2})
df_train['Sex']=df_train['Sex'].map({'female':0,'male':1})
df_test['Sex']=df_test['Sex'].map({'female':0,'male':1})
df_train['Cabin']=df_train['Cabin'].str[:1]
df_test['Cabin']=df_test['Cabin'].str[:1]
    
pc1=df_train[df_train['Pclass']==1]['Cabin'].value_counts()
pc2=df_train[df_train['Pclass']==2]['Cabin'].value_counts()
pc3=df_train[df_train['Pclass']==3]['Cabin'].value_counts()
df=pd.DataFrame([pc1,pc2,pc3])
df.index=['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))
cabin_mapping={'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8}
df_train['Cabin']=df_train['Cabin'].map(cabin_mapping)
df_test['Cabin']=df_test['Cabin'].map(cabin_mapping)
df_train['Cabin'].fillna(df_train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
df_test['Cabin'].fillna(df_test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
heatmap_data=df_train[['Survived','Pclass','Sex','Fare','Embarked','FamilySize','Initial','Age_cat']]
colormap=plt.cm.BuGn
plt.figure(figsize=(14,12))
plt.title('pearson Correlation of Featues',y=1.05,size=15)
sns.heatmap(heatmap_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
           square=True,cmap=colormap,linecolor='white',annot=True,annot_kws={'size':16})
df_train=pd.get_dummies(df_train,columns=['Initial'],prefix='Initial')
df_test=pd.get_dummies(df_test,columns=['Initial'],prefix='Initial')
df_train=pd.get_dummies(df_train,columns=['Embarked'],prefix='Embarked')
df_test=pd.get_dummies(df_test,columns=['Embarked'],prefix='Embarked')
df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Initial_4.0','Initial_0.0','Initial_1.0','Embarked_1','Embarked_2','Embarked_0'],axis=1,inplace=True)
df_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Initial_4.0','Initial_0.0','Initial_1.0','Embarked_1','Embarked_2','Embarked_0'],axis=1,inplace=True)
df_test.head()
df_train.head()
for col in df_train.columns:
    print(col)
    print(df_train[col].skew())
for col in df_test.columns:
    print(col)
    print(df_test[col].skew())
for col in  ['Fare']:
    df_train[col]=df_train[col].map(lambda i:np.log(i) if i>0 else 0)
   
for col in  ['Fare']:
    df_test[col]=df_test[col].map(lambda i:np.log(i) if i>0 else 0)
for col in ['Fare']:
    print(col)
    print(df_train[col].skew())
for col in ['Fare']:
    print(col)
    print(df_test[col].skew())
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes
from sklearn.svm import SVC #support vector classifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb  # xgboost is not a classifier in sklean, therefore needs better attention


X_train=df_train.drop('Survived',axis=1).values
target_label=df_train['Survived'].values
X_test=df_test.values
x_tr,x_vld,y_tr,y_vld=train_test_split(X_train,target_label,test_size=0.3,random_state=2018)
# build a set of base learners
SEED=420
def base_learners():
    """Construct a list of base learners"""
    lr = LogisticRegression(random_state=SEED)
    

    
    nn = MLPClassifier((80,10),random_state=SEED)
    
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    ab = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    dtc=DecisionTreeClassifier(criterion='entropy',random_state=420)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    svc =  model=SVC(C=1,kernel='rbf',coef0=1,probability=True)
    
   
    knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
 
    
    svc_rbf=SVC(kernel='rbf',random_state=420,probability=True)
    svc_linear=SVC(kernel='linear',random_state=420,probability=True)
    
    
   
    models = {

        'SVM': svc,
        'Random Forest': rf,
        'dtc':dtc, 
        'KNN': knn,
        
        'Neural Network': nn,
        'Logistic Regression': lr,
        'Extra Trees': et,
        'AdaBoosting': ab,
        'GradientBoosting': gb,
        
        ' svc_rbf': svc_rbf,
        'svc_linear':svc_linear
    }
    
    return models
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
models = base_learners()
ensemble_voting=VotingClassifier(estimators=list(zip(models.keys(),models.values())), 
                       voting='soft')
ensemble_voting.fit(x_tr,y_tr)

# model=RandomForestClassifier()
# model=KNeighborsClassifier(n_neighbors=13)
# model=DecisionTreeClassifier()
# model=GaussianNB()
# model=LogisticRegression(solver='lbfgs')
# model=SVC(C=1,kernel='rbf',coef0=1)
# model.fit(x_tr,y_tr)
prediction=ensemble_voting.predict(x_vld)
# from sklearn.preprocessing import StandardScaler

# scaler=StandardScaler()
# x_tr=scaler.fit_transform(x_tr)
# x_vld=scaler.fit_transform(x_vld)
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# model=keras.models.Sequential([
#     keras.layers.Dense(15, activation='relu', input_shape=(13,)),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# model.summary()
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[keras.metrics.AUC()])
# history=model.fit(x_tr,y_tr,batch_size=32,epochs=1000,validation_data=(x_vld,y_vld))

# plt.figure()
# plt.plot(history.history['loss'],'y',label='train loss')
# plt.plot(history.history['val_loss'],'r', label='val loss')
# prediction=model.predict(x_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존 맞춤'.format(y_vld.shape[0],100*metrics.accuracy_score(prediction,y_vld)))
from pandas import Series
# feature_importance=model.feature_importances_
# Series_feat_imp=Series(feature_importance,index=df_test.columns)
# plt.figure(figsize=(8,8))
# Series_feat_imp.sort_values(ascending=True).plot.barh()
# plt.xlabel('Feature importance')
# plt.ylabel('Feature')
# plt.show()
submission=pd.read_csv('../input/titanic/gender_submission.csv')
prediction=ensemble_voting.predict(X_test)
submission['Survived']=prediction
submission.to_csv('./my_first_submission.csv',index=False)
# y_pred_nn=model.predict_classes(X_test)
# submission=pd.read_csv('../input/titanic/gender_submission.csv')
# survived=np.squeeze(y_pred_nn)

# PassengerId = np.arange(892,1310)

# ans = pd.DataFrame(list(zip(PassengerId,survived)),columns=['PassengerId','Survived'])
# ans.head()

# ans.to_csv("final_ans.csv",index=False)