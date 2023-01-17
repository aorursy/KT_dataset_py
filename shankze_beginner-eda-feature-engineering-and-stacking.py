import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.tail()
train.describe()
train.info()
print(len(train))
print(len(test))
print(train.isnull().sum())
print(test.isnull().sum())
plt.figure(figsize=(12,5))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12,5))
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop(['Cabin'], axis =1, inplace=True)
test.drop(['Cabin'], axis = 1, inplace=True)

combined = pd.concat([train,test])

sns.heatmap(combined.corr(),annot=True)
combined.groupby('Pclass').mean()['Age']
def setAge(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 39
        elif pclass == 2:
            return 30
        else:
            return 25
    else:
        return age
train['Age'] = train[['Age','Pclass']].apply(setAge,axis=1)
test['Age'] = test[['Age','Pclass']].apply(setAge,axis=1)

combined['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].replace(np.NaN, 'S') 
test['Embarked'] = test['Embarked'].replace(np.NaN, 'S') 
combined.groupby('Pclass').mean()['Fare']
test[test['Fare'].isnull()]
test["Fare"].fillna(13.30, inplace=True)

print(train.isnull().sum())
print(test.isnull().sum())
sns.countplot(train['Survived'],palette= {1: "#1ab188", 0: "#c22250"})
sns.barplot(data=train,x='Sex',y='Survived',palette= {'male': "#3498db", 'female': "#ffe1ff"})
sns.countplot(data=train,x='Sex',hue='Survived',palette= {1: "#1ab188", 0: "#c22250"})
sns.barplot(data=train,x='Parch',y='Survived')
sns.barplot(data=train,x='SibSp',y='Survived')
train['FamilySize'] = train['SibSp'] + train ['Parch']
test['FamilySize'] = test['SibSp'] + test ['Parch']

sns.barplot(data=train,x='FamilySize',y='Survived')
def isAlone(cols):
    if (cols[0]==0) & (cols[1]==0):
        return 1
    else:
        return 0
train['IsAlone'] = train[['SibSp','Parch']].apply(isAlone,axis=1)
test['IsAlone'] = test[['SibSp','Parch']].apply(isAlone,axis=1)

sns.barplot(data=train,x='IsAlone',y='Survived')
sns.countplot(data=train,x='IsAlone',hue='Survived',palette= {1: "#1ab188", 0: "#c22250"})
combined = pd.concat([train,test])
sns.countplot(data=combined,x='Sex',hue='IsAlone')
f = sns.FacetGrid(combined,hue='IsAlone',size=5,aspect=4)
f.map(sns.kdeplot,'Age',shade= True)
f.add_legend()
combined[(combined['IsAlone'] == True)].sort_values(['Age']).head()
sns.distplot(train['Age'].dropna(),kde=False,bins=60)
f = sns.FacetGrid(train,hue='Survived',size=5,aspect=4)
f.map(sns.kdeplot,'Age',shade= True)
f.add_legend()
sns.barplot(data=train,x='Pclass',y='Survived',palette= {1: "#117A65", 2: "#52BE80",3: "#ABEBC6"})
f = sns.FacetGrid(combined,hue='Pclass',size=5,aspect=4)
f.map(sns.kdeplot,'Age',shade= True)
f.add_legend()
f = sns.FacetGrid(combined,hue='Pclass',size=5,aspect=4)
plt.xlim(0, 300)
f.map(sns.kdeplot,'Fare',shade= True)
f.add_legend()
f = sns.FacetGrid(combined,hue='Pclass',size=5,aspect=4)
plt.xlim(0, 50)
f.map(sns.kdeplot,'Fare',shade= True)
f.add_legend()
plt.figure(figsize=(12,5))
combined['FareBucket'] = (combined['Fare']/50).astype(int)*50
sns.countplot(data=combined,x='FareBucket',hue='Pclass',palette= {1: "#117A65", 2: "#52BE80",3: "#ABEBC6"})
combined[combined['Pclass']==1].sort_values('Fare').head()
combined[(combined['Pclass']==1)&(combined['Fare']>0)].sort_values('Fare').head()
combined[combined['Pclass']==3].sort_values('Fare',ascending=False).head(5)
combined[combined['Ticket']=='CA. 2343']
tst_sageFamily = test[test['Ticket']=='CA. 2343']
tst_sageFamily
ticCount = train.groupby('Ticket')['Sex'].count()
ticSurN = train.groupby('Ticket')['Survived'].sum()
ticCount = pd.DataFrame(ticCount)
ticSurN = pd.DataFrame(ticSurN)
ticSur = ticCount.join(ticSurN)
ticSur['TicSurvProb'] = ticSur['Survived']*(100) /(ticSur['Sex'])

ticSur.rename(index=str, columns={"Sex": "PassengerCount", "Survived": "PassengersSurvived"},inplace=True)
ticSur.reset_index(level=0, inplace=True)
ticSur.head()
ticSur = ticSur[ticSur['PassengerCount'] > 2]
train = pd.merge(train, ticSur, on=['Ticket', 'Ticket'],how='left')
train['TicSurvProb'] = train['TicSurvProb'].replace(np.NaN, 38.38)
test = pd.merge(test, ticSur, on=['Ticket', 'Ticket'],how='left')
test['TicSurvProb'] = test['TicSurvProb'].replace(np.NaN, 38.38)
train.drop(['PassengerCount','PassengersSurvived','Ticket'],axis=1,inplace=True)
test.drop(['PassengerCount','PassengersSurvived','Ticket'],axis=1,inplace=True)
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])  

train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
train['FareBucket'] = (train['Fare']/50).astype(int)
test['FareBucket'] = (test['Fare']/50).astype(int)
train['AgeBand'] = (train['Age']/5).astype(int)
test['AgeBand'] = (test['Age']/5).astype(int)
train.drop(['Age','Fare','PassengerId','Name','SibSp','Parch'], axis =1, inplace=True)
test.drop(['Age','Fare','Name','SibSp','Parch'], axis = 1, inplace=True)
p = sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'FamilySize', 'FareBucket','AgeBand', 'IsAlone']],hue='Survived', diag_kind = 'kde',palette= {1: "#1ab188", 0: "#c22250"} )
p.set(xticklabels=[])
plt.figure(figsize=(14,10))
sns.heatmap(train.corr(),annot=True)
pd.DataFrame(train.corr()['Survived']).abs().sort_values('Survived',ascending=False)
X = train.drop(['Survived'],axis=1)
y = train['Survived']
from sklearn.model_selection import KFold 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
num_of_estimators = 500
rfClass = RandomForestClassifier(n_estimators=200,max_depth=3,
                                 min_samples_leaf= 1,max_features=5,min_samples_split=3,criterion='entropy')
logClass = LogisticRegression(penalty='l1',C=21.544346900318832)
svcClass = SVC(gamma=0.001,C=10)
knnClass = KNeighborsClassifier(n_neighbors=9)
xgbClass = xgb.XGBClassifier(n_estimators=100,colsample_bytree= 0.8, gamma=1, max_depth=5, min_child_weight=1, subsample=1.0)
nbClass = MultinomialNB()
adaClass = AdaBoostClassifier(n_estimators=20,learning_rate=0.2)
extraTreesClass = ExtraTreesClassifier(n_estimators=50,bootstrap=False,criterion='entropy',max_features=3,min_samples_leaf=3,
                                        min_samples_split=10,max_depth=None)
gradientBClass = GradientBoostingClassifier(n_estimators=20,max_depth=3,max_features= 5,min_samples_leaf=3,min_samples_split=2)
res_alg = ['Random Forest','Logistic Regression','SVC','KNN','XG Boost','Naive Bayes','ADA Boost','Extra Trees','Gradient Boost']
res_acc = []
res_acc.append(cross_val_score(rfClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(logClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(svcClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(knnClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(xgbClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(nbClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(adaClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(extraTreesClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(gradientBClass,X,y,scoring='accuracy',cv=10).mean()*100)
cv_results = pd.DataFrame({'Algorithm':res_alg,'Accuracy':res_acc})
cv_results.sort_values('Accuracy',ascending=False)
plt.figure(figsize=(12,8))
cv_results = cv_results.sort_values(['Accuracy'],ascending=False).reset_index(drop=True)
sns.set(style="whitegrid")
sns.barplot(data=cv_results,x='Accuracy',y='Algorithm')
from sklearn.model_selection import train_test_split
X_s_train, X_s_test2, y_s_train, y_s_test2 = train_test_split(X, y, test_size=0.6,random_state=101)
X_s_valid, X_s_test, y_s_valid, y_s_test = train_test_split(X_s_test2, y_s_test2, test_size=0.2,random_state=101)
print(len(X_s_train))
print(len(X_s_valid)) 
print(len(X_s_test)) 
rfClass.fit(X_s_train,y_s_train)
adaClass.fit(X_s_train,y_s_train)
extraTreesClass.fit(X_s_train,y_s_train)
logClass.fit(X_s_train,y_s_train)
xgbClass.fit(X_s_train,y_s_train)
svcClass.fit(X_s_train,y_s_train)
knnClass.fit(X_s_train,y_s_train)
gradientBClass.fit(X_s_train,y_s_train)

#predict these models on validate data
vld_rfPred = rfClass.predict(X_s_valid)
vld_adaPred = adaClass.predict(X_s_valid)
vld_extPred = extraTreesClass.predict(X_s_valid)
vld_logPred = logClass.predict(X_s_valid)
vld_xgbPred = xgbClass.predict(X_s_valid)
vld_svcPred = svcClass.predict(X_s_valid)
vld_knnPred = knnClass.predict(X_s_valid)
vld_gbPred =  gradientBClass.predict(X_s_valid)
base_predictions_train = pd.DataFrame( {
    'RandomForest': vld_rfPred,
    'AdaptiveBoost': vld_adaPred,
    'ExtraTrees': vld_extPred,
    'Log': vld_logPred,
    'XGB': vld_xgbPred,  
    'SVC': vld_svcPred,
    'KNN': vld_knnPred,
    'GB' : vld_gbPred,
    'Y': y_s_valid,
    })
base_predictions_train.head()
sns.heatmap(base_predictions_train.corr(),annot=True)
#Concatenate all predictions on Validate
stacked_valid_predictions = np.column_stack((vld_rfPred, vld_adaPred, vld_extPred,vld_logPred,vld_xgbPred,
                                             vld_svcPred,vld_knnPred,vld_gbPred))
meta_model = xgb.XGBClassifier(n_estimators=90,colsample_bytree=0.8, gamma=5, max_depth=3,
                                      min_child_weight=10, subsample=0.6)
meta_model.fit(stacked_valid_predictions,y_s_valid)

feature_importances = pd.DataFrame(meta_model.feature_importances_,index = ['vld_rfPred', 'vld_adaPred', 'vld_extPred','vld_logPred','vld_xgbPred',
                                             'vld_svcPred','vld_knnPred','vld_gbPred'],columns=['importance']).sort_values('importance',   ascending=False)
feature_importances
tst_rfPred = rfClass.predict(X_s_test)
tst_adaPred = adaClass.predict(X_s_test)
tst_extPred = extraTreesClass.predict(X_s_test)
tst_logPred = logClass.predict(X_s_test)
tst_xgbPred = xgbClass.predict(X_s_test)
tst_svcPred = svcClass.predict(X_s_test)
tst_knnPred = knnClass.predict(X_s_test)
tst_gbPred = gradientBClass.predict(X_s_test)

#Concatenate base model predictions on Test
stacked_test_predictions = np.column_stack((tst_rfPred, tst_adaPred, tst_extPred,tst_logPred,tst_xgbPred,
                                           tst_svcPred,tst_knnPred,tst_gbPred))

s_test_pred = meta_model.predict(stacked_test_predictions)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_s_test,s_test_pred))
print(confusion_matrix(y_s_test,s_test_pred))
X_test = test.drop(['PassengerId'],axis=1)

#Use the base models to make predictions on test set
t_rfPred = rfClass.predict(X_test)
t_adaPred = adaClass.predict(X_test)
t_extPred = extraTreesClass.predict(X_test)
t_logPred = logClass.predict(X_test)
t_xgbPred = xgbClass.predict(X_test)
t_svcPred = svcClass.predict(X_test)
t_knnPred = knnClass.predict(X_test)
t_gbPred = gradientBClass.predict(X_test)

#Concatenate base model predictions on Test
stacked_t_predictions = np.column_stack((t_rfPred, t_adaPred, t_extPred,t_logPred,t_xgbPred,t_svcPred,t_knnPred,t_gbPred))
#Use the meta model to make predictions on test set
final_pred = meta_model.predict(stacked_t_predictions)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": final_pred
    })

tst_sageFamily.merge(submission,how='left',on='PassengerId')
submission.to_csv('titanic_output.csv', index=False)
