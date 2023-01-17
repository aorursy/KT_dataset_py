# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import sklearn.preprocessing as preprocessing

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

#%matplotlib inline



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



print(train_data.shape)

print(test_data.shape)

full = train_data.append(test_data,ignore_index=True)

full.describe()

full.info()



#Age/Cabin/Embarked/Fare

#Cabin->1/4 miss
sns.barplot(data=train_data,x='Embarked',y='Survived')

print("Embarked->S,Survival rate%.2f"%full["Survived"][full['Embarked']=='S'].value_counts(normalize=True)[1])

print("Embarked->C,Survival rate%.2f"%full["Survived"][full['Embarked']=='C'].value_counts(normalize=True)[1])

print("Embarked->Q,Survival rate%.2f"%full["Survived"][full['Embarked']=='Q'].value_counts(normalize=True)[1])
sns.factorplot("Pclass",col="Embarked",data=train_data,kind='count',size=3)
sns.barplot(data=train_data,x="Parch",y="Survived")
sns.barplot(data=train_data,x="Pclass",y="Survived")

#Pclass->1の場合Survived　rate最も高い
#X軸

ageFacet = sns.FacetGrid(train_data,hue="Survived",aspect=3)



ageFacet.map(sns.kdeplot,"Age",shade=True)

ageFacet.set(xlim=(0,train_data["Age"].max()))

ageFacet.add_legend()
fareFacet=sns.FacetGrid(train_data,hue='Survived',aspect=3)

fareFacet.map(sns.kdeplot,'Fare',shade=True)

fareFacet.set(xlim=(0,150))

fareFacet.add_legend()
farePlot=sns.distplot(full['Fare'][full['Fare'].notnull()],label='skewness:%.2f'%(full['Fare'].skew()))

farePlot.legend(loc='best')

full['Fare']=full['Fare'].map(lambda x: np.log(x) if x>0 else 0)

#fare的分布呈左偏的形态，其偏度skewness=4.37较大，说明数据偏移平均值较多，

#因此我们需要对数据进行对数化处理，防止数据权重分布不均匀。
#データの前処理

full["Cabin"] = full["Cabin"].fillna("U")

full["Cabin"].head()
full[full["Embarked"].isnull()]



full["Embarked"] = full["Embarked"].fillna("S")

full["Embarked"].value_counts()
full[full['Fare'].isnull()]

full['Fare']=full['Fare'].fillna(full[(full['Pclass']==3)&(full['Embarked']=='C')&(full['Cabin']=='U')]['Fare'].mean())
#feature engineering

#构造新特征Title

full['Title']=full['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())

#查看title数据分布

full['Title'].value_counts()



TitleDict={}

TitleDict['Mr']='Mr'

TitleDict['Mlle']='Miss'

TitleDict['Miss']='Miss'

TitleDict['Master']='Master'

TitleDict['Jonkheer']='Master'

TitleDict['Mme']='Mrs'

TitleDict['Ms']='Mrs'

TitleDict['Mrs']='Mrs'

TitleDict['Don']='Royalty'

TitleDict['Sir']='Royalty'

TitleDict['the Countess']='Royalty'

TitleDict['Dona']='Royalty'

TitleDict['Lady']='Royalty'

TitleDict['Capt']='Officer'

TitleDict['Col']='Officer'

TitleDict['Major']='Officer'

TitleDict['Dr']='Officer'

TitleDict['Rev']='Officer'



full['Title']=full['Title'].map(TitleDict)

full['Title'].value_counts()

sns.barplot(data=full,x='Title',y='Survived')
full['familyNum']=full['Parch']+full['SibSp']+1



def familysize(familyNum):

    if familyNum==1:

        return 0

    elif (familyNum>=2)&(familyNum<=4):

        return 1

    else:

        return 2



full['familySize']=full['familyNum'].map(familysize)

full['familySize'].value_counts()



sns.barplot(data=full,x='familySize',y='Survived')
full[full["Age"].isnull()].head()

#筛选数据集

AgePre=full[['Age','Parch','Pclass','SibSp','Title','familyNum']]



AgePre=pd.get_dummies(AgePre)

ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')

SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')

PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')



#相关性

AgeCorrDf=pd.DataFrame()

AgeCorrDf=AgePre.corr()

AgeCorrDf['Age'].sort_values()



AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)

AgePre.head()
#randomForestRegressor to pred AGE

AgeKnown=AgePre[AgePre['Age'].notnull()]

AgeUnKnown=AgePre[AgePre['Age'].isnull()]



#label

AgeKnown_X=AgeKnown.drop(['Age'],axis=1)

AgeKnown_y=AgeKnown['Age']



AgeUnKnown_X=AgeUnKnown.drop(['Age'],axis=1)



from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)

rfr.fit(AgeKnown_X,AgeKnown_y)



rfr.score(AgeKnown_X,AgeKnown_y)
AgeUnKnown_y = rfr.predict(AgeUnKnown_X)

full.loc[full['Age'].isnull(),['Age']]=AgeUnKnown_y

full.info() 
###!!!!!!point!!!!!

#找出具有明显同组效应且违背整体规律的数据，对其数据进行修正

full['Surname']=full['Name'].map(lambda x:x.split(',')[0].strip())

SurNameDict={}

SurNameDict=full['Surname'].value_counts()

full['SurnameNum']=full['Surname'].map(SurNameDict)



MaleDf=full[(full['Sex']=='male')&(full['Age']>12)&(full['familyNum']>=2)]

FemChildDf=full[((full['Sex']=='female')|(full['Age']<=12))&(full['familyNum']>=2)]
MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()

MSurNamDf.head()

MSurNamDf.value_counts()



MSurNamDict={}

MSurNamDict=MSurNamDf[MSurNamDf.values==1].index

MSurNamDict

FCSurNamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()

FCSurNamDf.head()

FCSurNamDf.value_counts()



FCSurNamDict={}

FCSurNamDict=FCSurNamDf[FCSurNamDf.values==0].index

FCSurNamDict
#从数据我们发现一般性别为男性的乘客幸存率较低，而女性及幼童的幸存率较高，

#因此我们将具有同组识别为幸存的男性的数据进行修饰，以提升模型将其预测为幸存的概率。



#for male have same family name -> girl 5 years old 

full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Age']=5

full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Sex']='female'



full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Age']=60

full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Sex']='male'

fullSel=full.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)

corrDf=pd.DataFrame()

corrDf=fullSel.corr()

corrDf['Survived'].sort_values(ascending=True)
plt.figure(figsize=(8,8))

sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass',

                    'Sex','SibSp','Title','familyNum','familySize',]].corr(),cmap='BrBG',annot=True,

           linewidths=.5)

plt.xticks(rotation=45)
fullSel=fullSel.drop(['familyNum','SibSp','Parch'],axis=1)

#one-hot编码

fullSel=pd.get_dummies(fullSel)

PclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')

familySizeDf=pd.get_dummies(full['familySize'],prefix='familySize')



fullSel=pd.concat([fullSel,PclassDf,familySizeDf],axis=1)
##point2 选择多种模型进行比较从而判断模型更适合用哪个算法



experData=fullSel[fullSel['Survived'].notnull()]

preData=fullSel[fullSel['Survived'].isnull()]



experData_X=experData.drop('Survived',axis=1)

experData_y=experData['Survived']

preData_X=preData.drop('Survived',axis=1)



from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold



kfold=StratifiedKFold(n_splits=10)



classifiers=[]

classifiers.append(SVC())

classifiers.append(DecisionTreeClassifier())

classifiers.append(RandomForestClassifier())

classifiers.append(ExtraTreesClassifier())

classifiers.append(GradientBoostingClassifier())

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression())

classifiers.append(LinearDiscriminantAnalysis())



cv_results=[]

for classifier in classifiers:

    cv_results.append(cross_val_score(classifier,experData_X,experData_y,

                                      scoring='accuracy',cv=kfold,n_jobs=-1))

cv_means=[]

cv_std=[]

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cvResDf=pd.DataFrame({'cv_mean':cv_means,

                     'cv_std':cv_std,

                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',

                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})



cvResDf
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }

modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, 

                                     scoring="accuracy", n_jobs= -1, verbose = 1)

modelgsGBC.fit(experData_X,experData_y)

modelLR=LogisticRegression()

LR_param_grid = {'C' : [1,2,3],

                'penalty':['l1','l2']}

modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 

                                     scoring="accuracy", n_jobs= -1, verbose = 1)

modelgsLR.fit(experData_X,experData_y)



LRpreData_y = modelLR.predict(preData_X)

LRpreData_y = LRpreData_y.astyoe(int)



LRpreResultDf=pd.DataFrame()

LRpreResultDf['PassengerId']=full['PassengerId'][full['Survived'].isnull()]

LRpreResultDf['Survived']=LRpreData_y

LRpreResultDf



LRpreResultDf.to_csv('/kaggle/working/LR_predictions.csv',index = False)
%%time

GBCpreData_y=modelgsGBC.predict(preData_X)

GBCpreData_y=GBCpreData_y.astype(int)



GBCpreResultDf=pd.DataFrame()

GBCpreResultDf['PassengerId']=full['PassengerId'][full['Survived'].isnull()]

GBCpreResultDf['Survived']=GBCpreData_y

GBCpreResultDf



GBCpreResultDf.to_csv('/kaggle/working/GBC_predictions.csv',index = False)