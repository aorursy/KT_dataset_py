#load relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
import warnings
warnings.filterwarnings('ignore')
#use online dataset to your notebook input 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#read data
train = pd.read_csv('/kaggle/input/titanic/train.csv') 
test = pd.read_csv('/kaggle/input/titanic/test.csv') 
#check basic info of data, understand variables
print(train.info())
print(test.info())
print("train dataset size:",train.shape)
print("test dataset size:",test.shape)
train.head()
test.head()
#merge the dataset of train and test for data processing conveniently
full_dataset=train.append(test,ignore_index=True)
full_dataset.describe()#check if there are obvious abnormal value, according the "mean","min","max"
full_dataset.info()#check missing value,the result show us,"age","fare","cabin","embarked" exist missing value
sns.barplot(data=train,x='Embarked',y='Survived')
#to calculate the survival rate of different port
print('the passengers embarked in Southampton，the survival rate is %.2f'%full['Survived'][full['Embarked']=='S'].value_counts(normalize=True)[1])
print('the passengers embarked in Cherbourg，the survival rate is %.2f'%full['Survived'][full['Embarked']=='C'].value_counts(normalize=True)[1])
print('the passengers embarked in Queenstown，the survival rate is %.2f'%full['Survived'][full['Embarked']=='Q'].value_counts(normalize=True)[1])
sns.factorplot('Pclass',col='Embarked',data=train,kind='count',size=3)
#the result shows us there is the highest proportion of 1st class in Cherbourg.
sns.barplot(data=train,x='Parch',y='Survived')
sns.barplot(data=train,x='SibSp',y='Survived')
sns.barplot(data=train,x='Pclass',y='Survived')
sns.barplot(data=train,x='Sex',y='Survived')
#Create axes
ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
#Draw a graph and select the graph type
ageFacet.map(sns.kdeplot,'Age',shade=True)
#range of axes, labels, etc.
ageFacet.set(xlim=(0,train['Age'].max()))
ageFacet.add_legend()
fareFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
fareFacet.map(sns.kdeplot,'Fare',shade=True)
fareFacet.set(xlim=(0,150))
fareFacet.add_legend()
# View the distribution of "Fare"
farePlot=sns.distplot(full_dataset['Fare'][full_dataset['Fare'].notnull()],label='skewness:%.2f'%(full_dataset['Fare'].skew()))
farePlot.legend(loc='best')
#process the "Fare" logarithmically
full_dataset['Fare']=full_dataset['Fare'].map(lambda x: np.log(x) if x>0 else 0)
# Process the missing values inside "Cabin" and fill them with U (Unknown)
full_dataset['Cabin']=full_dataset['Cabin'].fillna('U')
full_dataset['Cabin'].head()
#the missing value processing of "Embarked", check the missing value
full_dataset[full_dataset['Embarked'].isnull()]
full_dataset['Embarked'].value_counts()
#Inspection of the data distribution revealed the highest likelihood of boarding at Southampton, and therefore filled in the missing value.
full_dataset['Embarked']=full_dataset['Embarked'].fillna('S')
#the missing value processing of "Fare",check the missing value
full_dataset[full_dataset['Fare'].isnull()]
full_dataset['Fare']=full_dataset['Fare'].fillna(full_dataset[(full_dataset['Pclass']==3)&(full_dataset['Embarked']=='S')&(full_dataset['Cabin']=='U')]['Fare'].mean())
#Creat new feature - title
full_dataset['Title']=full_dataset['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
#check the distribution of "title"
full_dataset['Title'].value_counts()
#merge the similar title
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

full_dataset['Title']=full_dataset['Title'].map(TitleDict)
full_dataset['Title'].value_counts()
#view the relation between "title" and "survived"
sns.barplot(data=full_dataset,x='Title',y='Survived')
full_dataset['familyNum']=full_dataset['Parch']+full_dataset['SibSp']+1
#view familyNum with Survived
sns.barplot(data=full_dataset,x='familyNum',y='Survived')
#according to the number of family members, family size is divided into three categories: small, medium and large:

def familysize(familyNum):
    if familyNum==1:
        return 0
    elif (familyNum>=2)&(familyNum<=4):
        return 1
    else:
        return 2

full_dataset['familySize']=full_dataset['familyNum'].map(familysize)
full_dataset['familySize'].value_counts()
#view familySize with Survived
sns.barplot(data=full_dataset,x='familySize',y='Survived')
#Extract the first letter of "Cabin"
full_dataset['Deck']=full_dataset['Cabin'].map(lambda x:x[0])
#check the survival rate of different type of "Deck
sns.barplot(data=full_dataset,x='Deck',y='Survived')
#Extract the number of passengers for each ticket number
TickCountDict={}
TickCountDict=full_dataset['Ticket'].value_counts()
TickCountDict.head()
# put the same-ticket passenger number into the dataset
full_dataset['TickCot']=full_dataset['Ticket'].map(TickCountDict)
full_dataset['TickCot'].head()
#view TickCot with Survived
sns.barplot(data=full_dataset,x='TickCot',y='Survived')
#TickCot group is divided into three categories according to the size of TickCot.

def TickCountGroup(num):
    if (num>=2)&(num<=4):
        return 0
    elif (num==1)|((num>=5)&(num<=8)):
        return 1
    else :
        return 2
    
#get TickGroup type for each passenger
full_dataset['TickGroup']=full_dataset['TickCot'].map(TickCountGroup)
#view TickGroup with Survived
sns.barplot(data=full_dataset,x='TickGroup',y='Survived')
#check the missing value
full_dataset[full_dataset['Age'].isnull()].head()
#select variables
AgePre=full_dataset[['Age','Parch','Pclass','SibSp','Title','familyNum','TickCot']]

#one-hot enconding
AgePre=pd.get_dummies(AgePre)
ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')

#calculate the correlations between variables
AgeCorrDf=pd.DataFrame()
AgeCorrDf=AgePre.corr()
AgeCorrDf['Age'].sort_values()
#concat data
AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)
AgePre.head()
#split into train dataset and test dataset
AgeKnown=AgePre[AgePre['Age'].notnull()]
AgeUnKnown=AgePre[AgePre['Age'].isnull()]

#generate the feature and label of train dataset
AgeKnown_X=AgeKnown.drop(['Age'],axis=1)
AgeKnown_y=AgeKnown['Age']

#generate the feature of test dataset
AgeUnKnown_X=AgeUnKnown.drop(['Age'],axis=1)

#modeling in RandomForest
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)
rfr.fit(AgeKnown_X,AgeKnown_y)
#score of the model
rfr.score(AgeKnown_X,AgeKnown_y)
#predict "Age"
AgeUnKnown_y=rfr.predict(AgeUnKnown_X)
#fill in missing value
full_dataset.loc[full_dataset['Age'].isnull(),['Age']]=AgeUnKnown_y
full_dataset.info()  #no missing data
#Extract the passenger's last name and the corresponding number of passengers
full_dataset['Surname']=full_dataset['Name'].map(lambda x:x.split(',')[0].strip())
SurNameDict={}
SurNameDict=full_dataset['Surname'].value_counts()
full_dataset['SurnameNum']=full_dataset['Surname'].map(SurNameDict)

#split the data into two groups
MaleDf=full_dataset[(full_dataset['Sex']=='male')&(full_dataset['Age']>12)&(full_dataset['familyNum']>=2)]
FemChildDf=full_dataset[((full_dataset['Sex']=='female')|(full_dataset['Age']<=12))&(full_dataset['familyNum']>=2)]
#analysis on Male Goup
MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
MSurNamDf.head()
MSurNamDf.value_counts()
#Obtain surnames with survival rate of 1
MSurNamDict={}
MSurNamDict=MSurNamDf[MSurNamDf.values==1].index
MSurNamDict
#analysis on Female and Children Goup
FCSurNamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
FCSurNamDf.head()
FCSurNamDf.value_counts()
#Obtain surnames with survival rate of 0
FCSurNamDict={}
FCSurNamDict=FCSurNamDf[FCSurNamDf.values==0].index
FCSurNamDict
#Modify the male data of these surnames in the data set: 1. Change gender to female; 2. Change the age to 5.
full_dataset.loc[(full_dataset['Survived'].isnull())&(full_dataset['Surname'].isin(MSurNamDict))&(full_dataset['Sex']=='male'),'Age']=5
full_dataset.loc[(full_dataset['Survived'].isnull())&(full_dataset['Surname'].isin(MSurNamDict))&(full_dataset['Sex']=='male'),'Sex']='female'

#Correct the data of female and children with these surnames in the data set: 1. Change gender to male; 2. Change the age to 60.
full_dataset.loc[(full_dataset['Survived'].isnull())&(full_dataset['Surname'].isin(FCSurNamDict))&((full_dataset['Sex']=='female')|(full_dataset['Age']<=12)),'Age']=60
full_dataset.loc[(full_dataset['Survived'].isnull())&(full_dataset['Surname'].isin(FCSurNamDict))&((full_dataset['Sex']=='female')|(full_dataset['Age']<=12)),'Sex']='male'
# select with experience
fullSel=full_dataset.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)
#check the correlation between each variable with "Survived"
corrDf=pd.DataFrame()
corrDf=fullSel.corr()
corrDf['Survived'].sort_values(ascending=True)
#Thermal diagram, to see the correlation between the "Survived" and the other characteristics
plt.figure(figsize=(8,8))
sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass',
                    'Sex','SibSp','Title','familyNum','familySize','Deck',
                     'TickCot','TickGroup']].corr(),cmap='BrBG',annot=True,
           linewidths=.5)
plt.xticks(rotation=45)
fullSel=fullSel.drop(['familyNum','SibSp','TickCot','Parch'],axis=1)
#one-hot encoding
fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(full_dataset['Pclass'],prefix='Pclass')#?fullSel
TickGroupDf=pd.get_dummies(full_dataset['TickGroup'],prefix='TickGroup')
familySizeDf=pd.get_dummies(full_dataset['familySize'],prefix='familySize')

fullSel=pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)
#split dataset into train and test dataset
experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)
#import Machine learning algorithm packages
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
#set k-fold，Cross - sampling method to split dataset
kfold=StratifiedKFold(n_splits=10)

#Summarize different model algorithms
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
#Summary of cross-validation results for different machine learning
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))
#calculate the mean and standard deviation of the model scores
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
#summary results
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})

cvResDf
#Visualize the performing of algorithms 
sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})
#The optimization of model

#GradientBoostingClassifier
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

#LogisticRegression
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)
#modelgsGBC
print('the score of modelgsGBC：%.3f'%modelgsGBC.best_score_)
#modelgsLR
print('the scor of modelgsLR：%.3f'%modelgsLR.best_score_)
#view ROC curve
#Calculate the predicted value of the test data model
modelgsGBCtestpre_y=modelgsGBC.predict(experData_X).astype(int)

#drawing
from sklearn.metrics import roc_curve, auc  ###calculate roc,auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(experData_y, modelgsGBCtestpre_y) ###TP,FP
roc_auc = auc(fpr,tpr) ###auc

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###FP-x_axis，TP-y_axis
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic GradientBoostingClassifier Model')
plt.legend(loc="lower right")
plt.show()
#view ROC curve
#Calculate the predicted value of the test data model
modelgsLRtestpre_y=modelgsLR.predict(experData_X).astype(int)

#drawing
from sklearn.metrics import roc_curve, auc  ###calculate roc,auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(experData_y, modelgsLRtestpre_y) ###TP,FP
roc_auc = auc(fpr,tpr) ###auc

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###FP-x_axis，TP-y_axis
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic LogisticRegression Model')
plt.legend(loc="lower right")
plt.show()
#Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion matrix of GradientBoostingClassifier :\n',confusion_matrix(experData_y.astype(int).astype(str),modelgsGBCtestpre_y.astype(str)))
print('Confusion matrix of LinearRegression:\n',confusion_matrix(experData_y.astype(int).astype(str),modelgsLRtestpre_y.astype(str)))
#predict

#TitanicGBSmodle
GBCpreData_y=modelgsGBC.predict(preData_X)
GBCpreData_y=GBCpreData_y.astype(int)
#output the result of prediction
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=full_dataset['PassengerId'][full_dataset['Survived'].isnull()]
GBCpreResultDf['Survived']=GBCpreData_y
GBCpreResultDf
#output the result as filename.csv
GBCpreResultDf.to_csv('TitanicGBSmodle.csv',index=False)