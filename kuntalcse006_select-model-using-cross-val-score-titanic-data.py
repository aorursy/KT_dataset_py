import os
import  pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline

print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
train_df.head()
test_df = pd.read_csv("../input/test.csv")
test_df.head()
train_df.isnull().sum()
train_df.columns
# check for null for dataframe
test_df.isnull().sum()
# showing header off the dataframe
test_df.columns
# summary of dataframe 
train_df.describe()
# checking structure of data
train_df.dtypes
#showing rows and column no
train_df.shape, test_df.shape
# So, for adding those dataset there should be same variable name. 
# thus we have to add Survived variable in test dataset.
test_df['Survived']=0
# for recognize training and test dataset separately. 
# add another variable IsTrainData.
train_df['IsTraindata']=1
test_df['IsTraindata']=0
# adding 2 dataset ignore_index is most importent thing
titanic = train_df.append(test_df, ignore_index=True)
titanic.shape
titanic[880:900]
title = list(titanic.Name)
title[0] 
mm = str(title[0])
mm
# split name data retrieve particular sub string for each value
tmp = []
for tl in (title):
    mm = str(tl)
    titleList = mm.split(',')[1].split('.')[0].replace(' ','')
    tmp = tmp+[titleList]
print(tmp[0:100])
# add a new column, name title 
stitle = pd.Series(tmp)
titanic['Title']=stitle
titanic.head()
# create frequency table using crosstab
pd.crosstab(index=titanic['Title'],columns=titanic['Sex'])
mmk = titanic['Title'][titanic['Title'].isin(['Mrs','Mr','Rev','Sir','Don','Col','Capt','Master','Major','Jonkheer','Dr','Miss','Mme']) &
               titanic['Sex'].isin(['male'])]
list(mmk.index)
titanic.iloc[list(mmk.index),titanic.columns.get_loc('Title')] = 'Mr'
mmf = titanic['Title'][titanic['Title'].isin(['Dr','Lady','Master','Miss','Mlle','Ms','Rev','theCountess']) &
               titanic['Sex'].isin(['female'])]
list(mmf.index)
titanic.iloc[list(mmf.index),titanic.columns.get_loc('Title')] = 'Miss'
pd.crosstab(index=titanic['Title'],columns=titanic['Sex'])
indx = titanic['Title'][titanic['Title'].isin(['Dona','Mme']) & titanic['Sex'].isin(['female'])]
titanic.iloc[list(indx.index),titanic.columns.get_loc('Title')] = 'Miss'
pd.crosstab(index=titanic['Title'],columns=titanic['Sex'])
titanic.isnull().sum()
emindx = titanic['Embarked'][titanic.Embarked.isnull()]
titanic.iloc[list(emindx.index),titanic.columns.get_loc('Embarked')] = 'S'
titanic.isnull().sum()
pd.crosstab(index=titanic['Embarked'],columns='counts')
titanic.Age.isnull().sum()
titanic.Age.describe()
# histrogram showing usin hist function of column of datafframe
titanic.Age.hist()
# extract value from dataframe column
titanic.Age.values
# boxplot using plot.box() function with is predefine for column
titanic.Age.plot.box()
# checking density using density plot
titanic.Age.plot.density()
#titanic.Age.isnull()
ageind = titanic['Age'][titanic.Age.isnull()]
lage = list(ageind.index)
agemean = int(titanic.Age.mean())
titanic.iloc[lage,titanic.columns.get_loc('Age')] = agemean
titanic['Fare'][titanic.Fare == 0 & titanic.Fare.isnull()]
titanic.groupby('Pclass')['Fare'].sum()
ff = titanic.groupby('Pclass')['Fare'].sum()[1]
ss = titanic.groupby('Pclass')['Fare'].sum()[2]
tt = titanic.groupby('Pclass')['Fare'].sum()[3]
frind = titanic['Fare'][(titanic['Pclass']==1) & (titanic['Fare'] == 0)]
titanic.iloc[list(frind.index),titanic.columns.get_loc('Fare')] = ff

srind = titanic['Fare'][(titanic['Pclass']==2) & (titanic['Fare'] == 0)]
titanic.iloc[list(srind.index),titanic.columns.get_loc('Fare')] = ss

trind = titanic['Fare'][(titanic['Pclass']==3) & (titanic['Fare'] == 0)]
titanic.iloc[list(trind.index),titanic.columns.get_loc('Fare')] = tt
nlfrind = titanic['Pclass'][titanic.Fare.isnull()]
titanic.iloc[list(nlfrind.index),titanic.columns.get_loc('Fare')] = tt
titanic.isnull().sum()
mdtrain_df = titanic[titanic.IsTraindata ==1]
mdtest_df = titanic[titanic.IsTraindata == 0]
mdtrain_df.shape
mdtest_df.shape
mdtrain_df.columns
mdtrain_df.Embarked.value_counts()
def embarked_convert(text):
    if text == 'S':
        return 0
    if text == 'C':
        return 1
    if text == 'Q':
        return 2   
mdtrain_df.Title.value_counts()
def title_convert(text):
    if text == 'Mr':
        return 0
    if text == 'Miss':
        return 1
    if text == 'Mrs':
        return 2 
mdtrain_df.columns
mdtrain_df['Embarked'] = mdtrain_df.Embarked.apply(embarked_convert)
mdtrain_df['Title'] = mdtrain_df.Title.apply(title_convert)

mdtrain_x = mdtrain_df.loc[:,['Age','Fare','Pclass','Sex']]
# mdtrain_x = mdtrain_df.loc[:,['Age','Fare','Pclass','Sex']]
mdtrain_y =  mdtrain_df.loc[:,['Survived']]
mdtest_df['Embarked'] = mdtest_df.Embarked.apply(embarked_convert)
mdtest_df['Title'] = mdtest_df.Title.apply(title_convert)
mdtest = mdtest_df.loc[:,['Age','Fare','Pclass','Sex']]
mdtrain_x['Sex'] = mdtrain_x['Sex'].apply(lambda sex:1 if sex=="male" else 0) 
#mdtrain_x['Sex'] 

mdtest['Sex'] = mdtest['Sex'].apply(lambda sex:1 if sex=="male" else 0) 
#mdtest['Sex'] 
mdtrain_x.columns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import sklearn.ensemble as ensm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score 
from xgboost import XGBClassifier
mdtrain_x.shape, mdtrain_y.shape
c, r = mdtrain_y.shape
c
# we have to extract anly label for cross validation
c, r = mdtrain_y.shape
label = mdtrain_y.Survived.reshape(c,)

modelselect = [LogisticRegression(),DecisionTreeClassifier(),GaussianNB(),svm.SVC(),
               ensm.RandomForestClassifier(),SGDClassifier(),ensm.GradientBoostingClassifier()]
modelscore = []

for estimator in modelselect:
    scorem = cross_val_score(estimator,mdtrain_x,label,cv=10,scoring='accuracy')
    modelscore.append(scorem.mean())
modelscore
modelscrore = np.array(modelscore)
modelscrore.argmax()
modelname = np.array(modelselect)
modelname = modelname[modelscrore.argmax()]
modelname
model = modelname
model.fit(mdtrain_x,mdtrain_y)
result = model.predict(mdtrain_x)
result
pd.crosstab(index=mdtrain_y.Survived,columns='counts')
predictedvalue = pd.DataFrame({'Predvalue':result})
pd.crosstab(index=predictedvalue.Predvalue,columns='counts')
pd.crosstab(index=predictedvalue.Predvalue,columns=mdtrain_y.Survived)
test_result = model.predict(mdtest)
test_result
sub_df = pd.read_csv('../input/gender_submission.csv')
sub_df.head()
sub_df['Survived'] = test_result
sub_df.head()
sub_df.to_csv('mytitanicsubmision.csv',index=False)
sub_df.shape
