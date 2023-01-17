import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
INPUT_TRAIN_FILE='../input/titanic/train.csv'

INPUT_TEST_FILE='../input/titanic/test.csv'
data_train=pd.read_csv(INPUT_TRAIN_FILE)

data_test=pd.read_csv(INPUT_TEST_FILE)



data_train.shape,data_test.shape
data_train=data_train.drop('PassengerId',axis=1)

data_train.head()
data_train.describe()
data_train.info()
sns.heatmap(data_train.isna(),cmap='coolwarm')

plt.show()

#Inference:

# We have missing data for Age and Cabin column
data_train.isna().sum().apply(lambda x: f'{round(x*100/data_train.shape[0],3)}%')

## For testing data

data_test.isna().sum()
plt.figure(figsize=(8,4))

survival_rate=data_train['Survived'].value_counts(normalize=True)[1]

data_train['Survived'].value_counts().plot(kind='bar')

plt.xlabel('0-Perished| 1-Survived',fontsize=14)

plt.ylabel('# Of Passengers',fontsize=14)

plt.title(f'Survival Rate is {survival_rate:.2%} (All Passenger)',fontsize=14)
sns.countplot(x='Sex',hue='Survived',data=data_train)

plt.show()

g=sns.catplot(x='Pclass',y='Survived',data=data_train,kind='bar')



g.set_xlabels('Passenger Class',fontsize=14)

g.set_ylabels('Survival Probability', fontsize=14)

plt.show()
plt.figure(figsize=(10,6))

fg=sns.FacetGrid(data=data_train,row='Pclass',col='Survived',height=3,aspect=1.5,hue='Sex',

                )

fg.map(plt.hist,'Age',bins=20,alpha=0.7)

plt.legend(fontsize=14,bbox_to_anchor=[0.5,1,1,1])

plt.figure(figsize=(10,6))

fg=sns.FacetGrid(data=data_train,row='Embarked',col='Survived',height=3,aspect=1.5,hue='Sex',

                palette='Set1')

                

fg.map(plt.hist,'Age',bins=20,alpha=0.7)

plt.legend(fontsize=14,bbox_to_anchor=[0.5,1,1,1])
### Check Age distribution



sns.boxplot(x='Age',data=data_train,orient='v')

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

sns.countplot(x='Parch',hue='Survived',data=data_train)

plt.xlabel('Parent/Child',fontsize=12)

plt.ylabel('No of Passengers',fontsize=12)

plt.legend(loc='upper right')



Parch_prob=data_train['Parch'][data_train['Survived']==1].value_counts(normalize=True)

plt.subplot(1,2,2)

Parch_prob.plot(kind='bar',cmap='summer')

plt.legend(loc='upper right')

plt.xlabel('Parent/Child',fontsize=12)

plt.ylabel('Survival Prob',fontsize=12)

plt.show()

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

sns.countplot(x='SibSp',hue='Survived',data=data_train)

plt.xlabel('Siblings/Spouse',fontsize=12)

plt.ylabel('No of Passengers',fontsize=12)

plt.legend(loc='upper right')



SbSp_prob=data_train['SibSp'][data_train['Survived']==1].value_counts(normalize=True)

plt.subplot(1,2,2)

SbSp_prob.plot(kind='bar',cmap='winter')

plt.legend(loc='upper right')

plt.xlabel('Siblings/Spouse',fontsize=12)

plt.ylabel('Survival Prob',fontsize=12)

plt.show()

g=sns.FacetGrid(col='Pclass',hue='Survived',data=data_train,height=3.5,aspect=1.5)              

g.map(plt.hist,'Fare')

plt.legend(fontsize=13)

data_train['SibSp'][(data_train['Fare'] <= 100) & (data_train['Pclass']==1) 

                    & (data_train['Survived']==1)].value_counts()

data_train['FamilySize']=data_train['SibSp'] + data_train['Parch']

data_train['isAlone']=pd.Series(map(lambda x: 1 if x else 0,(data_train['SibSp']==0) & (data_train['Parch']==0)))



### Lets do same for Test data as well



data_test['FamilySize']=data_test['SibSp'] + data_test['Parch']

data_test['isAlone']=pd.Series(map(lambda x: 1 if x else 0,(data_test['SibSp']==0) & (data_test['Parch']==0)))
def imput_age(cols):

    age=cols[1]

    pclass=int(cols[0])

    if pd.isnull(age):

        age=age_by_Pclass.loc[pclass].values[0]

    else:

        age

            

    return int(age)

            


def impute_missing_data(df,drop_cols):

    if isinstance(drop_cols,list):

        for col in drop_cols:

            if col in df.columns:

                df.drop(col,axis=1,inplace=True)

    if isinstance(drop_cols,str) and drop_cols in df.columns:

        df.drop(drop_cols,axis=1,inplace=True)

    

    df['Age']=df[['Pclass','Age']].apply(imput_age,axis=1)

    df['Embarked'].fillna(df['Embarked'].mode().values[0],inplace=True)

    df['Fare'].fillna(df['Fare'].mean(),inplace=True)



    return df

    
age_by_Pclass=data_train[['Age','Pclass']].groupby('Pclass').mean()

### Lets Remove Features like 'Cabin(Have 77% Null values)','Name' and Ticket as well

data_train=impute_missing_data(data_train,['Cabin','Name','Ticket'])



data_test=impute_missing_data(data_test,['Cabin','Name','Ticket'])
data_train.isna().sum()
### Check heat map after imputing missing data

sns.heatmap(data_train.isna(),cmap='winter')

plt.show()
data_train.head()
gender_encode={'male':1,'female':0}

Embarked_encode={'S':0,'C':1,'Q':2}



data_train['Sex']=data_train['Sex'].replace(gender_encode)

data_train['Embarked']=data_train['Embarked'].replace(Embarked_encode)

data_train.drop(['SibSp','Parch'],axis=1,inplace=True)



data_test['Sex']=data_test['Sex'].replace(gender_encode)

data_test['Embarked']=data_test['Embarked'].replace(Embarked_encode)

data_test.drop(['SibSp','Parch'],axis=1,inplace=True)

data_train.head()
X_train=data_train.drop(['Survived','Fare'],axis=1)

y_train=data_train['Survived']

X_test=data_test.drop(['PassengerId','Fare'],axis=1)
X_train.shape,X_test.shape
classifier=LogisticRegression()

classifier.fit(X_train,y_train)
print(f'Traning Accuracy of Logistic is  {classifier.score(X_train,y_train):.2}')
pred=classifier.predict(X_train)

conf_mat=confusion_matrix(y_train,pred)

class_report=classification_report(y_train,pred)

print(f'confusion Matrix is \n {conf_mat}')

print(f'classification Matrix is \n {class_report}')

y_pred_prob=classifier.predict_proba(X_train)[::,0]

fpr,tpr,thres=roc_curve(y_train,y_pred_prob)



plt.plot(tpr,fpr)

plt.plot([0, 1], ls='--')

auc_log=round(np.trapz(fpr,tpr),3)

plt.xlabel('False Positive Rate',fontsize=14)

plt.ylabel('True Positive Rate',fontsize=14)

plt.title(f'ROC with AUC(Area under Curve): {auc_log}',fontsize=14)
class_rand=RandomForestClassifier(n_estimators=100,min_samples_leaf=3,n_jobs=-1,

                                  max_features=3,min_samples_split=5,max_depth=10)

class_rand.fit(X_train,y_train)

score=class_rand.score(X_train,y_train)

print(f'Random Forest Accuracy is {score:.2}')


y_pred_prob=class_rand.predict_proba(X_train)[::,0]

fpr,tpr,thres=roc_curve(y_train,y_pred_prob)

plt.plot(tpr,fpr)

plt.plot([0, 1], ls='--')

plt.xlabel('False Positive Rate',fontsize=14)

plt.ylabel('True Positive Rate',fontsize=14)

auc_rand=round(np.trapz(fpr,tpr),3)

plt.title(f'ROC with AUC(Area under Curve): {auc_rand}',fontsize=14)
pred_rand=class_rand.predict(X_test)

submission_csv=pd.DataFrame({'PassengerId':data_test['PassengerId'],

                             'Survived':pred_rand})

#submission_csv.to_csv('.../output/gender_submission.csv',index=False)