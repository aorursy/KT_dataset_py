import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None
train=pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
test=pd.read_csv('/kaggle/input/titanic/test.csv')

pid=test.PassengerId
train.dtypes
train.info()
#Null value count

print('Train Set')

print(train[train.isnull()])

#print(train.isnull())

print('\nTest Set')

print(test.isnull().sum())
#Train dataset fillna

train['Age']=train['Age'].fillna(train['Age'].median())

train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode()[0])



#Test datset fillna

test['Age']=test['Age'].fillna(test['Age'].median())

test['Fare']=test['Fare'].fillna(test['Fare'].mean())
#Null values count

print('Train Set')

print(train.isnull().sum())

print('\nTest Set')

print(test.isnull().sum())
#Creating new columns my merging or using previous columns

for data in [train,test]:

    data['FamilySize']=data['SibSp']+data['Parch']+1

    data['IsAlone']=1

    data['IsAlone'].loc[data['FamilySize']>1]=0

    

    #Gets the title from the name (Mr, Mrs, etc)

    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train.head()
train['Title'].value_counts()
#Too many unknown titles are there

#Cleaning the 'Title column to just include [Mr, Mrs, Miss, Master, Misc]'

#Misc contains all the unknown and gender neutral titles



#Replacing known titles

train['Title']=train['Title'].replace('Ms','Miss')

train['Title']=train['Title'].replace('Mlle','Miss')

train['Title']=train['Title'].replace('the Countess','Mrs')

train['Title']=train['Title'].replace('Mme','Mrs')



test['Title']=test['Title'].replace('Dona','Mrs')

test['Title']=test['Title'].replace('Ms','Miss')



#Replacing by misc

names=(train['Title'].value_counts() < 10)

train['Title']=train['Title'].apply(lambda x: 'Misc' if names.loc[x] == True else x)

names=(test['Title'].value_counts() < 10)

test['Title']=test['Title'].apply(lambda x: 'Misc' if names.loc[x] == True else x)



print('Train Set\n',train['Title'].value_counts())

print('\nTest Set\n',test['Title'].value_counts())
#Drop columns

columns=['PassengerId','Cabin','Ticket','Name']

train=train.drop(columns,axis=1)

test=test.drop(columns,axis=1)
#Converting categorical columns into numerical columns using LabelEncoder

#LabelEncoder gives each unique str/char a numerical value starting from 0 

from sklearn.preprocessing import LabelEncoder



label=LabelEncoder()

for data in [train,test]:

    data['Sex']=label.fit_transform(data['Sex'])            # 0:Female, 1:Male

    data['Embarked']=label.fit_transform(data['Embarked'])  # 0:C , 1:Q 2:S

    data['Title']=label.fit_transform(data['Title'])        # 0:Master 1:Misc 2:Miss, 3:Mr, 4:Mrs

    

    data['Age']=data['Age'].astype('int64')

      

train.head()    
#Percentage Survived for each category

target=['Survived']

selected=['Sex','Pclass','Embarked','Title','SibSp','Parch','FamilySize','IsAlone']

for x in selected:

    print('Survival Percentage By',x)

    print(train[[x, target[0]]].groupby(x,as_index=False).mean(),'\n')

        

# Sex - 0: Female, 1: Male

# Embarked - 0: C, 1: Q, 2: S

# Title - 0: Master, 1: Misc, 2. Miss, 3: Mr, 4: Mrs

# IsAlone - 0:No, 1: Yes
import matplotlib.pyplot as plt

import seaborn as sns



#Shows the ratio of survived:dead passengers according to age

plt.figure(figsize=(10,5))

plt.hist(x=[train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Survival by Age')

plt.xlabel('Age')

plt.ylabel('# of Passengers')

plt.legend()
#Shows the percent Survival according to each Embarkment, Class and whether the passenger is alone or not

fig,ax=plt.subplots(1, 3,figsize=(15,5))

sns.barplot(x='Embarked',y='Survived',data=train,ax=ax[0])

sns.barplot(x='Pclass',y='Survived',order=[1,2,3],data=train,ax=ax[1])

sns.barplot(x='IsAlone',y='Survived',order=[1,0],data=train,ax=ax[2])
#Shows the survival of Passengers according to thier ticket price

df=train.copy()

df['Fare'] = pd.cut(df['Fare'], bins=[0, 50, 100, 150, 200, 250, 300,600])

plt.figure(figsize=(10,5))

sns.pointplot(x='Fare',y='Survived', data=df)
#Shows the survival of Passengers according to thier age

df['Age'] = pd.cut(df['Age'], bins=[0,10,20,30,40,50,60,70,80])

plt.figure(figsize=(10,5))

sns.pointplot(x='Age',y='Survived', data=df)
#Survivals according to family size

plt.figure(figsize=(10,5))

sns.pointplot(x='FamilySize', y='Survived',data=train)
#Survival of each sex on the basis of embarkment, class and whether the passenger is alone or not

fig,ax=plt.subplots(1,3,figsize=(20,7))

sns.barplot(x='Sex',y='Survived',hue='Embarked',data=train,ax=ax[0])

sns.barplot(x='Sex',y='Survived',hue ='Pclass',data=train,ax=ax[1])

sns.barplot(x='Sex',y='Survived',hue='IsAlone',data=train,ax=ax[2])
#Selecting the independent and dependent variables

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



target=train['Survived']

train.drop(['Survived'],axis=1, inplace=True)
#Training on different models

from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score

from catboost import CatBoostRegressor

from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=1, stratify=target)



print('Mean Absolute Errors:')



#RandomForestClassifier

model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=True, random_state=1, verbose=0,

                       warm_start=False)

model.fit(X_train, y_train)

predict = model.predict(X_val)

print('Random Forrest: ' + str(mean_absolute_error(predict, y_val)))



#XGBoost

model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,

                        gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror',

                        nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)

model.fit(X_train, y_train)

predict = model.predict(X_val)

print('XGBoost: ' + str(mean_absolute_error(predict, y_val)))



#LassoCV

model = LassoCV(max_iter=1e7,  random_state=14, cv=10)

model.fit(X_train, y_train)

predict = model.predict(X_val)

print('Lasso: ' + str(mean_absolute_error(predict, y_val)))



# GradientBoosting   

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

model.fit(X_train, y_train)

predict = model.predict(X_val)

print('GradientBoosting: ' + str(mean_absolute_error(predict, y_val)))
#Predicting on the best model

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_curve,auc, confusion_matrix, classification_report

import sklearn.metrics as metrics

import matplotlib.pyplot as plt



#RandomForestClassifier

model=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                           criterion='gini', max_depth=4, max_features='auto',

                           max_leaf_nodes=5, max_samples=None,

                           min_impurity_decrease=0.0, min_impurity_split=None,

                           min_samples_leaf=1, min_samples_split=15,

                           min_weight_fraction_leaf=0.0, n_estimators=350,

                           n_jobs=None, oob_score=True, random_state=1, verbose=0,

                           warm_start=False)

model.fit(X_train, y_train)

predict = model.predict(X_val)

print('Random Forest MAE: ' + str(mean_absolute_error(predict, y_val)))

print("Out of Bag Score: %.4f" % model.oob_score_)



y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_val)



# Building the ROC Curve and Confusion Matrix

print("Training accuracy: ", accuracy_score(y_train, y_pred_train))

print("Testing accuracy: ", accuracy_score(y_val, y_pred_test))

print("\nConfusion Matrix\n")

print('[[True Positive    False Positive]\n[False Negative    True Negative]]\n')

print(confusion_matrix(y_val, y_pred_test))



fpr, tpr, _ = roc_curve(y_val, y_pred_test)

roc_auc = auc(fpr, tpr)

print("\nROC AUC on evaluation set",roc_auc )



plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
#Prediction

prediction=model.predict(test)

prediction
output = pd.DataFrame({'PassengerId': pid, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Submission successfully saved!")