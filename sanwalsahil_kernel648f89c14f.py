# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data= pd.read_csv('../input/titanic/train.csv')
data.head()
data.info()
data.describe()
for column in data.columns:

    print(data[column].value_counts())

for column in data.columns:

    print(column+': ',format(data[column].isnull().sum()))
data.head()
contColumnList = ['PassengerId','Age','Fare']

catColumnList = ['Pclass','Survived','Sex','SibSp','Parch','Cabin','Embarked']
for col in contColumnList:

    plt.figure()

    sns.distplot(data[col])
data.columns
figure,plot = plt.subplots()

sns.barplot(x=data['Survived'].value_counts().index,

           y = data['Survived'].value_counts())

plot.set_xticklabels(['Not Survived','Survived'])

for patch in plot.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    plot.text(label_x, label_y,

                #left - freq below - rel freq wrt population as a percentage

               str(int(patch.get_height())) + '(' +

               '{:.0%}'.format(patch.get_height()/len(data.Survived))+')',

               horizontalalignment='center', verticalalignment='center')
figure,plot = plt.subplots()

sns.barplot(x=data['Pclass'].value_counts().index,

           y = data['Pclass'].value_counts())

for patch in plot.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    plot.text(label_x, label_y,

                #left - freq below - rel freq wrt population as a percentage

               str(int(patch.get_height())) + '(' +

               '{:.0%}'.format(patch.get_height()/len(data.Pclass))+')',

               horizontalalignment='center', verticalalignment='center')
figure,plot = plt.subplots()

sns.barplot(x=data['Sex'].value_counts().index,

           y = data['Sex'].value_counts())

for patch in plot.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    plot.text(label_x, label_y,

                #left - freq below - rel freq wrt population as a percentage

               str(int(patch.get_height())) + '(' +

               '{:.0%}'.format(patch.get_height()/len(data.Sex))+')',

               horizontalalignment='center', verticalalignment='center')
def plotCatDistribution(col,size=(20,5)):

    figure,plot = plt.subplots(figsize=size)

    sns.barplot(x=data[col].value_counts().index,

               y = data[col].value_counts())

    for patch in plot.patches:

        label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

        label_y = patch.get_y() + patch.get_height()+10

        plot.text(label_x, label_y,

                    #left - freq below - rel freq wrt population as a percentage

                   str(int(patch.get_height())) + '(' +

                   '{:.0%}'.format(patch.get_height()/len(data[col]))+')',

                   horizontalalignment='center', verticalalignment='center')

    

plotCatDistribution('SibSp')

    
    

plotCatDistribution('Parch')
plotCatDistribution('Embarked',(5,5))
sns.countplot(x='Sex',hue='Survived',data=data)
sns.countplot(x='Pclass',hue='Survived',data=data)
sns.countplot(x='SibSp',hue='Survived',data=data)
sns.countplot(x='Parch',hue='Survived',data=data)
sns.countplot(x='Embarked',hue='Survived',data=data)
sns.catplot(x='Pclass',hue='Sex',col='Survived',data=data,kind='count')
#newData = data

sns.catplot(x='Parch',hue='Sex',col='Survived',data=data,kind='count')
for x in data.corr()['Survived'].index:

    print(x + '----' + format(data.corr()['Survived'][x]))
sns.barplot(x=data.corr()['Survived'].index,

             y = data.corr()['Survived'].value_counts().index)
figure,plot = plt.subplots(figsize=(10,10))

sns.heatmap(data=data.corr(),annot=True)
sns.violinplot(x='Survived',y='Fare',data=data,hue='Sex')
sns.boxplot(x='Survived',y='Fare',data=data)
#Plot 2: We can use a line plot:

sns.catplot(x='Pclass',hue='Embarked',col='Sex',row='Survived',kind='count',data=data)
figure,ax = plt.subplots(figsize=(10,5))

sns.kdeplot(data['Age'][data['Survived']==1 & data['Age'].notnull()],ax=ax,legend=True)

sns.kdeplot(data['Age'][data['Survived']==0 & data['Age'].notnull()],ax=ax,legend=True)

ax.legend(['Survived','Not Survived'])
figure,ax = plt.subplots(figsize=(10,5))

sns.kdeplot(data['Fare'][data['Survived']==1 & data['Fare'].notnull()],ax=ax,legend=True)

sns.kdeplot(data['Fare'][data['Survived']==0 & data['Fare'].notnull()],ax=ax,legend=True)

ax.legend(['Survived','Not Survived'])
sns.pairplot(data)
for col in data.columns:

    print(col + ' --- '+format(data[col].isnull().sum()))

    

train = data
#importing test data

Test = pd.read_csv('../input/titanic/test.csv')

for col in Test.columns:

    print(col + ' --- '+format(Test[col].isnull().sum()))
Test[Test['Fare'].isnull()]
median_fare = Test.groupby(['Pclass','Parch'])['Fare'].median()[3][0]
Test['Fare'].fillna(median_fare)
Test['Fare'].isnull().sum()
print(train['Embarked'].isnull().sum())

print(Test['Embarked'].isnull().sum())
train[train['Embarked'].isnull()]
train['Embarked'] = train['Embarked'].fillna('S')
print(train['Embarked'].isnull().sum())

print(Test['Embarked'].isnull().sum())
print(train['Age'].isnull().sum())

print(Test['Age'].isnull().sum())
#

train.corr()['Age']
trainCopy = train.copy()
agePclassSibsp = trainCopy.groupby(['Pclass','SibSp']).median()['Age']

agePclassSibsp
agePclassSibsp[3][8] = agePclassSibsp[3][5]
agePclassSibsp
for pc in range(1,4):

    for ss in agePclassSibsp[pc].index.tolist():

        print('median for Pclass '+str(pc)+' and SibSp ' + str(ss)+ ' is '+str(agePclassSibsp[pc][ss]))
trainCopy['Age'] = trainCopy.groupby(['Pclass','SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))

trainCopy['Age'] = trainCopy['Age'].fillna(11)

trainCopy.info()
train = trainCopy

train.info()
Test['Age'] = Test.groupby(['Pclass','SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))

Test.info()
Test[Test['Fare'].isnull()]
Test['Fare']=Test['Fare'].fillna(median_fare)
Test[Test['Fare'].isnull()]
from collections import Counter



def detect_outliers(df, n, features):

    outliers_indices = [] #create a empty list to keep track of the passenger row number.

    for col in features:

        # 1st quartile (25%)

        Q1 = np.nanpercentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.nanpercentile(df[col], 75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step)

                              | (df[col] > Q3 + outlier_step)].index

        

        #print(df[(df[col] < Q1 - outlier_step)

                              #| (df[col] > Q3 + outlier_step)].index)

        print(col,Q1-outlier_step,Q3+outlier_step)

        # append the found outlier indices for col to the list of outlier indices

        outliers_indices.extend(outlier_list_col)

        

    #print(outliers_indices)

    

    # select observations containing more than 2 outliers

    outliers_indices = Counter(outliers_indices)

    multiple_outliers = list(k for k, v in outliers_indices.items() if v > n)

    #print(outliers_indices)

    

    return multiple_outliers





Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])

Outliers_to_drop
train.loc[Outliers_to_drop]
for col in train.columns:

    print('Unique features in '+col+' are '+str(len(train[col].unique())))
train.drop(['PassengerId'],inplace=True,axis=1)

Test.drop(['PassengerId'],inplace=True,axis=1)
train.drop(['Ticket'],inplace=True,axis=1)

Test.drop(['Ticket'],inplace=True,axis=1)
train['Name'].head(10)
train.drop(['Name'],inplace=True,axis=1)

Test.drop(['Name'],inplace=True,axis=1)
train.head()
train['Cabin'].isnull().sum()

Test['Cabin'].isnull().sum()
btrain = train.copy()

btest = Test.copy()
btrain['Cabin']=btrain['Cabin'].fillna('M')

btrain['Cabin'].isnull().sum()
btrain.head()
btrain['Cabin'] = btrain['Cabin'].apply(lambda x:x[0])
btrain['Cabin'].value_counts()
btest['Cabin']=btest['Cabin'].fillna('M')

btest['Cabin'].isnull().sum()
btest['Cabin'] = btest['Cabin'].apply(lambda x:x[0])
btest.head()
from sklearn.preprocessing import OneHotEncoder

trial = btrain.copy()
trial.head()
# columns for one hot encoding - Survived,Pclass,Sex,Parch,Cabin,Embarked

trialEnc = trial[['Pclass','Sex','Parch','Cabin','Embarked']]

trialEnc.head()
ohe = OneHotEncoder()

trialEncFit = ohe.fit(trialEnc)

trialEnc = ohe.transform(trialEnc).toarray()
trialEnc = pd.DataFrame(trialEnc)
trialEnc.head()
trial = trial.join(trialEnc)
trial.head()
trial.drop(['Pclass','Sex','Parch','Cabin','Embarked'],inplace=True,axis=1)
trial.head()
trainF = trial.copy()
testTr = btest.copy()
testTr.head()
# columns for one hot encoding - Survived,Pclass,Sex,Parch,Cabin,Embarked

testEnc = testTr[['Pclass','Sex','Parch','Cabin','Embarked']]

testEnc.head()
import pickle

ohe = OneHotEncoder()

testEncFit = ohe.fit(testEnc)

testEnc = ohe.transform(testEnc).toarray()



with open('TestFit.pickle', 'wb') as f:

    pickle.dump(testEncFit, f)



from IPython.display import FileLink

FileLink('TestFit.pickle')



testEnc = pd.DataFrame(testEnc)

testEnc.head()
testF = testTr.join(testEnc)

testF.drop(['Pclass','Sex','Parch','Cabin','Embarked'],inplace=True,axis=1)

testF.head()
trainF.head()
y = trainF['Survived']

trainF.drop(['Survived'],inplace=True,axis=1)

x=trainF
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

scFit = sc.fit(x_train)

X_train = sc.transform(x_train)

X_test = sc.transform(x_test)

with open('scFit.pickle', 'wb') as f:

    pickle.dump(scFit, f)

    



from IPython.display import FileLink

FileLink('scFit.pickle')
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,precision_recall_curve,classification_report,roc_curve,r2_score,precision_score,recall_score,f1_score

from sklearn.model_selection import GridSearchCV, cross_val_score,StratifiedKFold, learning_curve
conMatList = []

prcList = []

clRep = []

rocDet = []

preScore = []

recScore = []

f1Score = []

yPred = []



def getClassModel(model):

    model = model()

    model_name = model.__class__.__name__

    

    model.fit(X_train,y_train)

    

    #getting prediction

    y_pred = model.predict(X_test)

    yPred.append([model_name,y_pred])

    

    #getting scores

    pre_score = precision_score(y_test,y_pred)

    rec_score = recall_score(y_test,y_pred)

    f1score = f1_score(y_test,y_pred)

    

    preScore.append([model_name,pre_score])

    recScore.append([model_name,rec_score])

    f1Score.append([model_name,f1score])

    

    ## getting confusion matrix

    cm = confusion_matrix(y_test,y_pred)

    matrix = pd.DataFrame(cm,columns=['predicted 0','predicted 1'],

                         index=['Actual 0','Actual 1'])

    conMatList.append([model_name,matrix])

    

    ## getting precision recall curve values

    precision,recall,thresholds = precision_recall_curve(y_test,y_pred)

    prcList.append([model_name,precision,recall,thresholds])

    

    ## roc details

    fpr,tpr,thresholds = roc_curve(y_test,y_pred)

    rocDet.append([model_name,fpr,tpr,thresholds])

    

    ## classification report

    

    classRep = classification_report(y_test,y_pred)

    clRep.append([model_name,classRep])
classModelList = [LogisticRegression,SVC,GaussianNB,DecisionTreeClassifier

                 ,RandomForestClassifier,KNeighborsClassifier]



for model in classModelList:

    getClassModel(model)
pd.DataFrame(X_train).head()
kfold = StratifiedKFold(n_splits=10)

#getting cross validation scores of each model

cv_result = []

for model in classModelList:

    cv_result.append(cross_val_score(model(),X_train,y_train,

                                     scoring='accuracy', cv=kfold,n_jobs=4))

    

cv_means = []

cv_std = []



for res in cv_result:

    cv_means.append(res.mean())

    cv_std.append(res.std())

    

model_name = []

for model in classModelList:

    modelIns = model()

    model_name.append(modelIns.__class__.__name__)

    

cv_res = pd.DataFrame({

    "CrossValMeans":cv_means,

    "CrossValErrors":cv_std,

    "Model":model_name

})

  

cv_res
fig,ax = plt.subplots(figsize=(20,10))

sns.distplot(y_test,hist=False,label='test_set',ax=ax)

for pred in yPred:

    try:

        sns.distplot(pred[1],hist=False,label=pred[0],ax=ax)

    except:

        print(pred[0])

for mat in conMatList:

    print(mat[0])

    print(' ')

    print(mat[1])

    print('---------------------------------------------------------')
precisionDf = pd.DataFrame(preScore,columns=['model','precisionScore'])

recallDf = pd.DataFrame(recScore,columns=['model','recallScore'])

f1Df = pd.DataFrame(f1Score,columns=['model','f1Score'])

precisionDf['f1Score'] = f1Df['f1Score']

precisionDf['recallScore'] = recallDf['recallScore']

precisionDf
for roc in rocDet:

    print(roc[0])

    fpr = roc[1]

    tpr = roc[2]

    plt.plot(fpr,tpr,label=roc[0])

    plt.legend()
for prc in prcList:

    precision = prc[1]

    recall = prc[2]

    plt.plot(precision,recall,label=prc[0])

    plt.legend()
logReg = LogisticRegression()

logReg.fit(X_train,y_train)
Test = sc.fit_transform(testF)

Test1 = pd.read_csv("../input/titanic/test.csv")



output3 = pd.DataFrame({"PassengerId": Test1.PassengerId, "Survived":logReg.predict(Test)})

output3.PassengerId = output3.PassengerId.astype(int)

output3.Survived = output3.Survived.astype(int)



output3.to_csv("output3.csv", index=False)

print("Your submission was successfully saved!")

output3.head(10)
with open('model.pickle', 'wb') as f:

    pickle.dump(logReg, f)

    



from IPython.display import FileLink

FileLink('model.pickle')