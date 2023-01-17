import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

sns.set(style='white', context='notebook', palette='dark')


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.dtypes)
train.describe()
print(train.isnull().sum()/train.shape[0])
print(test.isnull().sum()/test.shape[0])
def barplots(dfMean, dfCount, title1, title2):
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
    sns.barplot(x=dfMean.index, y=dfMean['Survived'], alpha=.5,ax=axis1).set_title(title1)
    sns.barplot(x=dfCount.index, y=dfCount['Survived'], alpha=.5,ax=axis2).set_title(title2)
df1=train.groupby(['Pclass'])[['Survived']].mean()
df2=train.groupby(['Pclass'])[['Survived']].count()
barplots(df1, df2, "Surival Rate", "Count")
df1=train.groupby(['SibSp'])[['Survived']].mean()
df2=train.groupby(['SibSp'])[['Survived']].count()
barplots(df1, df2, "SibSp Survival Rate", "SibSp Count")
df1=train.groupby(['Parch'])[['Survived']].mean()
df2=train.groupby(['Parch'])[['Survived']].count()
barplots(df1, df2, "Parch Survival Rate", "Parch Count")
def family_size(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] 
    return data

train = family_size(train)
test = family_size(test)

df1=train.groupby(['FamilySize'])[['Survived']].mean()
df2=train.groupby(['FamilySize'])[['Survived']].count()
barplots(df1, df2, "Surival Rate", "Count")
def Bin_family(data):
    data['FamilyBin'] = data['FamilySize'].map({0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2}).astype(int)
    return data

train=Bin_family(train)
test=Bin_family(test)

df1=train.groupby(['FamilyBin'])[['Survived']].mean()
df2=train.groupby(['FamilyBin'])[['Survived']].count()
barplots(df1, df2, "Surival Rate", "Count")
print(train.groupby(['FamilyBin', 'Pclass'])[['Survived']].mean())
print(train.groupby(['FamilyBin', 'Pclass'])[['Survived']].count())
df1=train.groupby(['Sex'])[['Survived']].mean()
df2=train.groupby(['Embarked'])[['Survived']].mean()
barplots(df1, df2, "Sex Survival Rate", "Embarked Survival Rate")
print(train.groupby(['Embarked', 'Pclass'])[['Survived']].count())
print(train.groupby(['Embarked', 'Pclass'])[['Survived']].mean())
print(train.groupby(['Embarked'])[['Pclass']].count())
def titleExtract(data):
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Mr')
    data['Title'] = data['Title'].replace(['Ms', 'Miss'], 'Miss')
    data['Title'] = data['Title'].replace(['Mlle', 'Mme','Lady', 'Countess', 'Dona'], 'Mrs')
    return data


train=titleExtract(train)
test=titleExtract(test)
df1=train.groupby(['Title'])[['Survived']].mean()
df2=train.groupby(['Title'])[['Survived']].count()
barplots(df1, df2, "Surival Rate", "Count")

print(train.groupby(['Title'])[['Age']].mean())
print(train.groupby(['Title', 'Sex'])[['Survived']].mean())
print(train.groupby(['Title', 'Sex'])[['Survived']].count())
print(train.groupby(['Title', 'Pclass'])[['Survived']].mean())
print(train.groupby(['Title', 'Pclass'])[['Survived']].count())
def mapTitle(data):
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
    data['Title'] = data['Title'].map(title_map).astype(int)
    return data
   

train=mapTitle(train)
test=mapTitle(test)
def fill_fare_nulls(data):
    data.loc[(data.Fare.isnull())&(data.Pclass==1), 'Fare']=60.3
    data.loc[(data.Fare.isnull())&(data.Pclass==2), 'Fare']=14.25
    data.loc[(data.Fare.isnull())&(data.Pclass==3), 'Fare']=8.05
    return data

test=fill_fare_nulls(test)

train.groupby(['Pclass'])[['Fare']].median()
train['FareBand'] = pd.cut(train['Fare'], (-1, 8.05, 14.25, 60.2875, 1000), labels=['0','1','2','3'])
test['FareBand'] = pd.cut(test['Fare'], (-1, 8.05, 14.25, 60.2875, 1000), labels=['0','1','2','3']) 
train['FareBand'].astype(int)
test['FareBand'].astype(int)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['FareBand'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title("Survival Rate")

d=train.groupby(['FareBand'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis2).set_title("Count")
print(train.groupby('Pclass')['Fare'].median())
print(test.groupby('Pclass')['Fare'].median())
def splitClass_Train(data):
    data.loc[(data.Fare>60.3)&(data.Pclass==1), 'PassengerCat']=0
    data.loc[(data.Fare<=60.3)&(data.Pclass==1), 'PassengerCat']=1
    
    data.loc[(data.Fare>14.3)&(data.Pclass==2), 'PassengerCat']=2
    data.loc[(data.Fare<=14.3)&(data.Pclass==2), 'PassengerCat']=3
              
    data.loc[(data.Fare>8.1)&(data.Pclass==3), 'PassengerCat']=4
    data.loc[(data.Fare<=8.1)&(data.Pclass==3), 'PassengerCat']=5
    data['PassengerCat']=data['PassengerCat'].astype(int)
    return data


train=splitClass_Train(train)

def splitClass_Test(data):
    data.loc[(data.Fare>60.00)&(data.Pclass==1), 'PassengerCat']=0
    data.loc[(data.Fare<=60.00)&(data.Pclass==1), 'PassengerCat']=1
    
    data.loc[(data.Fare>15.8)&(data.Pclass==2), 'PassengerCat']=2
    data.loc[(data.Fare<=15.8)&(data.Pclass==2), 'PassengerCat']=3
              
    data.loc[(data.Fare>7.9)&(data.Pclass==3), 'PassengerCat']=4
    data.loc[(data.Fare<=7.9)&(data.Pclass==3), 'PassengerCat']=5
    data['PassengerCat']=data['PassengerCat'].astype(int)
    return data


test=splitClass_Test(test)

df1=train.groupby(['PassengerCat'])[['Survived']].mean()
df2=train.groupby(['PassengerCat'])[['Survived']].count()
barplots(df1, df2, "Surival Rate", "Count")
avgAges=train.groupby(['Title', 'Pclass'], as_index=False)['Age'].median()
print(avgAges)
avgAges = avgAges['Age']

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
#Before filling in NA values
train['Age'].hist(bins=20, ax=axis1).set_title('Before Imputing') 
def fillAgeNulls(data, avgAges):
    data.loc[(data.Age.isnull())&(data.Title==1)&(data.Pclass==1), 'Age']=avgAges[0]
    data.loc[(data.Age.isnull())&(data.Title==1)&(data.Pclass==2), 'Age']=avgAges[1]
    data.loc[(data.Age.isnull())&(data.Title==1)&(data.Pclass==3), 'Age']=avgAges[2]
    
    data.loc[(data.Age.isnull())&(data.Title==2)&(data.Pclass==1), 'Age']=avgAges[3]
    data.loc[(data.Age.isnull())&(data.Title==2)&(data.Pclass==2), 'Age']=avgAges[4]
    data.loc[(data.Age.isnull())&(data.Title==2)&(data.Pclass==3), 'Age']=avgAges[5]
    
    data.loc[(data.Age.isnull())&(data.Title==3)&(data.Pclass==1), 'Age']=avgAges[6]
    data.loc[(data.Age.isnull())&(data.Title==3)&(data.Pclass==2), 'Age']=avgAges[7]
    data.loc[(data.Age.isnull())&(data.Title==3)&(data.Pclass==3), 'Age']=avgAges[8]
    
    data.loc[(data.Age.isnull())&(data.Title==4), 'Age']=avgAges[9]
    
    
    return data


train=fillAgeNulls(train, avgAges)
test=fillAgeNulls(test, avgAges)

train['Age'].hist(bins=20, ax=axis2).set_title('After Imputing')

train['AgeBand']=pd.cut(train['Age'], (0, 6, 60, 80), labels=['0','1','2'])
test['AgeBand']=pd.cut(test['Age'], (0, 6, 60, 80), labels=['0','1','2'])
train['AgeBand'].astype(int);
test['AgeBand'].astype(int);

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

d=train.groupby(['AgeBand'])[['Survived']].mean()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5, ax=axis1).set_title('Surival Rate')

d=train.groupby(['AgeBand'])[['Survived']].count()
sns.barplot(x=d.index, y=d['Survived'], alpha=.5,ax=axis2).set_title('Count')
print(train.groupby(['AgeBand', 'Sex'])[['Survived']].mean())
print(train.groupby(['AgeBand', 'Sex'])[['Survived']].count())
train=train.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin', 'SibSp','Parch', 'Embarked', 'FamilyBin','FareBand', 'Pclass', 'Sex', 'AgeBand'])
test=test.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin','SibSp', 'Parch', 'Embarked', 'FamilyBin','FareBand', 'Pclass', 'Sex', 'AgeBand'])
train.head(10)
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
print(train.dtypes)
train=pd.get_dummies(train,columns=['Title', 'PassengerCat'])
test=pd.get_dummies(test,columns=['Title', 'PassengerCat'])
print(train.isnull().sum())
print("\nTest Set:")
print(test.isnull().sum())
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import f_regression
from sklearn.ensemble import GradientBoostingClassifier
train_X=train.drop('Survived', axis=1)
train_Y=train['Survived'].astype(int)
test_X=test
KF=KFold(n_splits=10, random_state=1)

models=[]
modelScores=[]
modelSTD=[]
LR_model = LogisticRegression(solver = 'lbfgs', max_iter = 3000)

CV=cross_val_score(LR_model,train_X,train_Y,cv=KF, scoring="accuracy")
CV.mean()
LR_model.fit(train_X, train_Y)
models.append('LogisticRegression')
modelScores.append(round(CV.mean(),3))
modelSTD.append(round(CV.std(),3))
from sklearn.model_selection import GridSearchCV

RFC = RandomForestClassifier()


RF_grid = {"max_depth": [None],
              "max_features": [4, 6, 8],
              "min_samples_split": [3, 5, 7],
              "min_samples_leaf": [5, 10, 15],
              "bootstrap": [True],
              "n_estimators" :[500],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = RF_grid, cv=10, scoring="accuracy", n_jobs= -1, verbose = 1)

gsRFC.fit(train_X,train_Y)

RFC_best = gsRFC.best_estimator_

# Best score
print(gsRFC.best_score_)
print(gsRFC.best_params_)
#Random Forest
RF_model = RandomForestClassifier(bootstrap= True,
 criterion = 'gini',
 max_depth = None,
 max_features=8,
 min_samples_leaf = 5,
 min_samples_split = 7,
 n_estimators = 500, random_state=1)

RF_model.fit(train_X, train_Y)


CV=cross_val_score(RF_model,train_X,train_Y,cv=KF, scoring="accuracy")
models.append('RandomForest')
#modelScores.append(round(RF_model.oob_score_,3))
modelScores.append(round(CV.mean(),3))
modelSTD.append('NA')

featureImportance = pd.concat((pd.DataFrame(train_X.columns, columns = ['Feature']), 
           pd.DataFrame(RF_model.feature_importances_, columns = ['Importance'])), 
          axis = 1).sort_values(by='Importance', ascending = False)[:20]
plt.subplots(figsize=(20,8))
sns.barplot(x=featureImportance['Feature'], y=featureImportance['Importance'], alpha=.5).set_title('Feature Importance')
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    
plt.figure()
cnf_matrix = confusion_matrix(train_Y, RF_model.predict(train_X))
np.set_printoptions(precision=2)
class_names = ['0', '1']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for RF')
plt.figure()
cnf_matrix = confusion_matrix(train_Y, LR_model.predict(train_X))
np.set_printoptions(precision=2)
class_names = ['0', '1']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for LR')
ModelComparison=pd.DataFrame({'CV Score':modelScores, 'Std':modelSTD}, index=models)
ModelComparison

test_ID = pd.read_csv('../input/test.csv')
test_ID = test_ID['PassengerId']

Survival_predictions = RF_model.predict(test)
ID=np.arange(892,1310,1)

submission=pd.DataFrame({
        "PassengerId": ID,
        "Survived": Survival_predictions
    })
submission.to_csv('submission.csv', index=False)