import warnings,os,math

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

train.head()
recordId='PassengerId'

target='Survived'

trainId=train[recordId]

testId=test[recordId]



# Dropping **PassengerId** (unique identifier) feature from train & test set.

train.drop(recordId,axis=1,inplace=True)

test.drop(recordId,axis=1,inplace=True)



# Checking Dataset shape

print('Train Set\t %d X %d'%(train.shape[0],train.shape[1]))

print('Test Set\t %d X %d'%(test.shape[0],test.shape[1]))
features=['Pclass','SibSp','Parch','Sex','Embarked','Age','Fare','Survived']

nrows=2

ncols=int(np.ceil(len(features)/nrows))

fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(14,5))

fig.subplots_adjust(wspace=0.4,hspace=0.4)

for row in range(nrows):

    for col in range(ncols):

        feature=features[row*ncols+col]

        if feature in ['Age','Fare']:

            sns.violinplot(train[target],train[feature],ax=ax[row,col])

        else:

            sns.barplot(train[feature],train[target],ax=ax[row,col])            
nTrain=train.shape[0]

nTest=test.shape[0]

trainY=train[target]

allData=pd.concat((train,test)).reset_index(drop=True)

allData.drop(target,axis=1,inplace=True)

print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))
count=allData.isnull().sum().sort_values(ascending=False)

percentage=(allData.isnull().sum()/allData.isnull().count()).sort_values(ascending=False)*100

dtypes=allData[count.index].dtypes

missingData=pd.DataFrame({'Count':count,'Percentage':percentage,'Type':dtypes})

missingData.drop(missingData[missingData['Count']==0].index,inplace=True)

missingData.head(10)
idx=allData[allData['Cabin'].isnull()].index

allData.loc[idx,'Cabin']='M'
allData.drop(columns=['Age'],inplace=True)
allData[allData['Embarked'].isnull()]
idx=allData[allData['Embarked'].isnull()].index

allData.loc[idx,'Embarked']='S'
allData[allData['Fare'].isnull()]
name=allData[allData['Fare'].isnull()].Name.values[0]

dataset=train if name in train['Name'].tolist() else test

groups=dataset.groupby(['Pclass','Sex','Embarked'])['Fare'].mean().to_frame('Mean Fare')

groups
idx=allData[allData['Fare'].isnull()].index

allData.loc[idx,'Fare']=groups.loc[3,'male','S'].values[0]
count=allData.isnull().sum().sort_values(ascending=False).to_frame(name='count')

count
# FamilySize

allData['FamilySize']=allData['SibSp']+allData['Parch']+1

# IsAlone

allData['IsAlone']=None

idx=allData[allData['FamilySize']==1].index

allData.loc[idx,'IsAlone']=1

idx=allData[allData['FamilySize']>1].index

allData.loc[idx,'IsAlone']=0

allData['IsAlone']=allData['IsAlone'].astype(int)

# Title

allData['Title']=allData['Name'].str.extract(" ([A-Za-z]+)\.")

titleNames=(allData['Title'].value_counts()<10)

allData['Title']=allData['Title'].apply(lambda title: 'Misc' if titleNames.loc[title]==True else title)

# Deck

allData['Deck']=allData['Cabin'].str[0]

# Dropping Name, Cabin and Ticket feature

allData.drop(columns=['Name','Cabin','Ticket'],inplace=True)
allData.head()
_train=allData[:nTrain]

features=['FamilySize','IsAlone','Title','Deck']

nrows=1

ncols=int(np.ceil(len(features)/nrows))

fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(14,2.5))

fig.subplots_adjust(wspace=0.4,hspace=0.4)

for col in range(ncols):

    feature=features[col]

    if feature is not 'Deck':

        sns.barplot(_train[feature],trainY,ax=ax[col])

    else:

        sns.barplot(_train[feature],trainY,ax=ax[col],order=['A','B','C','D','E','F','G','M','T'])
_train.groupby(['Deck','Pclass']).size().to_frame(name='Passenger Count')
allData['Deck']=allData['Deck'].replace(['A','B','C','T'],'ABC')

allData['Deck']=allData['Deck'].replace(['D','E'],'DE')

allData['Deck']=allData['Deck'].replace(['F','G'],'FG')

allData['Deck'].value_counts()
fig,ax=plt.subplots(figsize=(5,4))

corrMat=allData.corr()

sns.heatmap(corrMat,annot=True)
allData=pd.get_dummies(allData)

print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))

allData.sample(5)
trainX=allData[:nTrain]

testX=allData[nTrain:]
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression



# Splitting training set further into training and validation set

subTrainX,valX,subTrainY,valY=train_test_split(trainX,trainY,test_size=0.2,random_state=42)
classifiers={

    "Nearest Neighbors":KNeighborsClassifier(3), 

    "Linear SVM":SVC(kernel='linear'), 

    "RBF SVM":SVC(kernel='rbf'),

    "Decision Tree":DecisionTreeClassifier(max_features=10,max_depth=10), 

    "Random Forest":RandomForestClassifier(n_estimators=200,max_features=10,max_depth=10), 

    "Neural Net":MLPClassifier(alpha=0.001),

    "Logistic Regression":LogisticRegression()

}
# Training

models_accuracy=[]

i=0

for classifier_name,classifier in classifiers.items():

    i+=1

    print("Training classifiers{}".format("."*i),end='\r')

    classifier.fit(subTrainX,subTrainY)

    predictions=classifier.predict(subTrainX)

    train_accuracy=accuracy_score(subTrainY,predictions)

    predictions=classifier.predict(valX)

    test_accuracy=accuracy_score(valY,predictions)

    models_accuracy.append({

        'classifier':classifier_name,

        'train':train_accuracy,

        'test':test_accuracy

    })

models_accuracy=pd.DataFrame(models_accuracy)

models_accuracy=models_accuracy.sort_values(by=['test','train'],ascending=False)

df=models_accuracy.melt(

    id_vars='classifier',

    value_name='accuracy'

)

sns.barplot(x='accuracy',y='classifier',hue='variable',data=df);

models_accuracy
feature_importance=pd.DataFrame({'feature':subTrainX.columns,'importance':classifiers['Random Forest'].feature_importances_})

feature_importance=feature_importance.sort_values(by='importance',ascending=False)

feature_importance.head(10)
rfc=RandomForestClassifier(n_estimators=200,max_features=10,max_depth=10)

rfc.fit(trainX,trainY)

predictions=rfc.predict(trainX)

accuracy=accuracy_score(trainY,predictions)

print('TRAINING ACCURACY : {:.4f}'.format(accuracy))
predictions=rfc.predict(testX)

submission=pd.DataFrame()

submission[recordId]=testId

submission[target]=predictions

submission.head()
submission.to_csv('submission.csv',index=False)