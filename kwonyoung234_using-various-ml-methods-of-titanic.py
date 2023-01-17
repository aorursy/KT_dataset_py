# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier #KNearestNeighbor
from sklearn.linear_model import LogisticRegression #LogisticRegression -- Lienar classification method
from sklearn.svm import LinearSVC # Linear classification method
from sklearn.naive_bayes import GaussianNB # classification specialized one Among naive bayes
from sklearn.tree import DecisionTreeClassifier # DecisionTree 
from sklearn.ensemble import RandomForestClassifier # RandomForest of ensemble method
from sklearn.ensemble import GradientBoostingClassifier # GradientBoosting of ensemble method
from sklearn.svm import SVC #Kernel Support Vector Machine(it's different with Linear SVC. SVC could make multidimensional decision border)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # for finding proper value of our Analysising method parameters

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import dataset to Kernel
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#check the shape of each DataFrame
print(train.head())
print(test.head())
print(train.columns)
print(test.columns)
#fuction for finding Missing value
def getMissingValue(column):
    isnull = column.isnull()
    nullsum = isnull.sum()
    return nullsum

print(train.apply(getMissingValue))
print(test.apply(getMissingValue))
#Let's see the relations btw columns of train dataset
fig = plt.figure(figsize=[12,10])
sns.heatmap(train.drop('PassengerId',axis=1).corr(),vmax=0.6,square=True,annot=True)
#Looking at it more in detail
sns.pairplot(train.dropna(),vars=["Survived","Pclass","Age","SibSp","Parch","Fare"],hue="Survived",palette=['red','blue'])
#Compare each columns with Survived column using survivors and the deads
surv = train[train.Survived == 1]
nosurv = train[train.Survived == 0]
surv_col = 'blue'
nosurv_col = 'red'
#Age of those who are survived or not
fig = plt.figure(figsize=[12,12])
ax1 = fig.add_subplot(3,3,1)
ax1 = sns.distplot(surv.Age.dropna().values,bins=range(0,train.Age.max().astype('int'),1),kde=False,color=surv_col)
ax1 = sns.distplot(nosurv.Age.dropna().values,bins=range(0,train.Age.max().astype('int'),1),kde=False,color=nosurv_col)
ax1.set_xlabel('Age')

ax2 = fig.add_subplot(3,3,2)
ax2 = sns.barplot(x="Pclass",y="Survived",data=train)
ax2.set_xlabel('Pclass')

ax3 = fig.add_subplot(3,3,3)
ax3 = sns.barplot(x='Sex',y='Survived',data=train)
ax3.set_xlabel('Sex')

ax4 = fig.add_subplot(3,3,4)
ax4 = sns.barplot(x='SibSp',y='Survived',data=train)
ax4.set_xlabel('SibSp')

ax5 = fig.add_subplot(3,3,5)
ax5 = sns.barplot(x='Parch',y='Survived',data=train)
ax5.set_xlabel('Parch')

ax6 = fig.add_subplot(3,3,6)
ax6 = sns.barplot(x='Embarked',y='Survived',data=train)
ax6.set_xlabel('Embarked')

ax7 = fig.add_subplot(3,3,7)
ax7 = sns.distplot(np.log10(surv.Fare.dropna()+1),kde=False,color=surv_col)
ax7 = sns.distplot(np.log10(nosurv.Fare.dropna()+1),kde=False,color=nosurv_col)
ax7.set_xlabel('Fare')
#We could know some facts what we got
#1 Children could get higher chances to be survived
#2 The higher Pclass people got, The higher possibility they got (Pclass와 Survived 는 역의관계)
#3 Females were higher than males
#4 people who got family were good to be survived
#5 people who boarded at the C port were good to be survived
#6 The higher Fare people paid, The higher possibility they could get

#Let's investigate much deeper using this facts
#See much more detail about Age and Sex at the same time
fsurv = train[np.logical_and(train.Sex=='female',train.Survived == 1)]
fnosurv = train[np.logical_and(train.Sex=='female',train.Survived == 0)]
msurv = train[np.logical_and(train.Sex=='male',train.Survived == 1)]
mnosurv =train[np.logical_and(train.Sex=='male',train.Survived == 0)]

#distribution of Age
fig = plt.figure(figsize=[12,5])
ax1 = fig.add_subplot(1,2,1)
ax1 = sns.distplot(fsurv.Age.dropna().values,bins=range(0,train.Age.max().astype('int'),1),kde=False,color=surv_col)
ax1 = sns.distplot(fnosurv.Age.dropna().values,bins=range(0,train.Age.max().astype('int'),1),kde=False,color=nosurv_col)
ax1.set_xlabel('Female Age')

ax2 = fig.add_subplot(1,2,2)
ax2 = sns.distplot(msurv.Age.dropna().values,bins=range(0,train.Age.max().astype('int'),1),kde=False,color=surv_col)
ax2 = sns.distplot(mnosurv.Age.dropna().values,bins=range(0,train.Age.max().astype('int'),1),kde=False,color=nosurv_col)
ax2.set_xlabel('Male Age')
# Age Variation of those who boarded on titanic by plcass
fig = plt.figure(figsize=[12,10])
sns.violinplot(x="Pclass",y="Age",hue="Survived",data=train,split=True)
plt.hlines([0,10],xmin=-1,xmax=3,linestyles='dotted')

#1,2,3,5
fig = plt.figure(figsize=[12,10])
sns.factorplot(x='Pclass',y='Survived',hue='Sex',col='Embarked',data=train)
#What Sex proportion each Pclass has?
tab = pd.crosstab(train.Pclass,train.Sex)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train)
# What Sex ratio each Embarked place has
tab = pd.crosstab(train.Embarked,train.Sex)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
#Pclass ratio by Embarked Place
tab = pd.crosstab(train.Embarked,train.Pclass)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
#Embarked place of Q has lot of error about 1st and 2nd class due to that these didn't have enough amount of samples
sns.barplot(x='Embarked',y='Survived',hue='Pclass',data=train)
#Family or Alone by Sex
tab = pd.crosstab(train.SibSp,train.Sex)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
tab = pd.crosstab(train.Parch,train.Sex)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
# According to two graph about Parch and SibSp, the probability that that Female would be with is higher than male 
# The relation btw Fare and Pclass
surv_class3 = train[np.logical_and(train.Survived==1,train.Pclass==3)]
nosurv_class3 = train[np.logical_and(train.Survived==0,train.Pclass==3)]
surv_class2 = train[np.logical_and(train.Survived==1,train.Pclass==2)]
nosurv_class2 = train[np.logical_and(train.Survived==0,train.Pclass==2)]
surv_class1 = train[np.logical_and(train.Survived==1,train.Pclass==1)]
nosurv_class1 = train[np.logical_and(train.Survived==0,train.Pclass==1)]


fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(3,1,1)
ax1 = sns.distplot(np.log10(surv_class3.Fare.dropna().values+1),kde=False,color=surv_col)
ax1 = sns.distplot(np.log10(nosurv_class3.Fare.dropna().values+1),kde=False,color=nosurv_col)
ax1.set_xlabel('Pclass 3')

fig = plt.figure(figsize=[12,10])
ax2 = fig.add_subplot(3,1,2)
ax2 = sns.distplot(np.log10(surv_class2.Fare.dropna().values+1),kde=False,color=surv_col)
ax2 = sns.distplot(np.log10(nosurv_class2.Fare.dropna().values+1),kde=False,color=nosurv_col)
ax2.set_xlabel('Pclass 2')

fig = plt.figure(figsize=[12,10])
ax3 = fig.add_subplot(3,1,3)
ax3 = sns.distplot(np.log10(surv_class1.Fare.dropna().values+1),kde=False,color=surv_col)
ax3 = sns.distplot(np.log10(nosurv_class1.Fare.dropna().values+1),kde=False,color=nosurv_col)
ax3.set_xlabel('Pclass 1')
#Filling Missing value of train dataset
train.apply(getMissingValue)
train[train.Embarked.isnull()] #getting dataframe what Embarked attr is null
#train.Embarked.value_counts() #=> S is the most Embarked place of them
train.Embarked.iloc[61]= 'S'
train.Embarked.iloc[829] = 'S'


#Filling Missing value of test dataset
test.apply(getMissingValue)
#test[test.Fare.isnull()] #dataframe what Fare attr is null
test.Fare.iloc[152] = test.Fare.dropna().median()
print(test.iloc[152])
#Let's make derived columns using the facts what we found above
#train.isnull().sum()
#test.isnull().sum()
combine = pd.concat([train.drop('Survived',1),test])
survived = train.Survived
#combine.isnull().sum()

combine.head()
combine['Child'] = combine.Age <= 10
combine['Family'] = combine.SibSp + combine.Parch
combine['Alone'] = (combine.SibSp + combine.Parch) == 0
combine['Large_Family'] = (np.logical_or(combine['SibSp']>2,combine['Parch']>3))
combine['Title'] = combine.Name.str.split(",",expand=True)[1].str.split(".",expand=True)[0]
combine['Young'] = (np.logical_or(combine.Age <=30, combine.Title.isin(['Master','Miss','Mlle'])))
combine['Cabin_known'] = combine.Cabin.isnull() == False
combine['Age_known'] = combine.Age.isnull() == False

combine['Deck'] = combine.Cabin.str[0];
combine['Deck'] = combine['Deck'].fillna(value = "U")

combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine.Fare+1))).astype('int')
combine['Shared_ticket'] = np.where(combine.groupby('Ticket')['Name'].transform('count') >1 ,1,0)

train = combine.iloc[:len(train),]
test = combine.iloc[len(train):,]
train['Survived'] = survived
#Additionally, find relations with derived attrs
#Child

tab = pd.crosstab(train.Child,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
#Family
tab = pd.crosstab(train.Family,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
sns.factorplot(x='Family',y="Survived",hue='Sex',data=train)
#if passengers had proper number of family(1~3), they would be survived well than others 
#Alone
tab = pd.crosstab(train.Alone,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
sns.factorplot(x="Alone",y="Survived",hue="Sex",data=train)
#Large_Family
tab = pd.crosstab(train.Large_Family,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
sns.factorplot(x="Large_Family",y='Survived',hue='Sex',data=train)
#Cabin_known
tab = pd.crosstab(train.Cabin_known,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
#Deck
tab = pd.crosstab(train.Deck,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
sns.factorplot(x='Deck',y='Survived',hue='Sex',data=train)
#Shared_ticket
tab = pd.crosstab(train.Shared_ticket,train.Survived)
print(tab)
tab.div(tab.sum(1),axis=0).plot(kind='bar',stacked=True)
sns.factorplot(x='Shared_ticket',y='Survived',hue='Sex',data=train)
#for putting our dataset into learning model, we should change our dataset to numeric values e.g) Sex, Deck, Embarked  

combine = pd.concat([train.drop('Survived',1),test])
survived = train.Survived

#Sex
combine['Sex'] = combine.Sex.astype('category')
combine.Sex.cat.categories = [0,1]
combine['Sex'] = combine.Sex.astype('int')

#Deck
combine['Deck'] = combine.Deck.astype('category')
combine.Deck.cat.categories = range(0,combine.Deck.nunique())
combine['Deck'] = combine.Deck.astype('int')

#Embarked
combine['Embarked'] = combine.Embarked.astype('category')
combine.Embarked.cat.categories = range(0,3)
combine['Embarked'] = combine.Embarked.astype('int')

train = combine.iloc[:len(train),]
test = combine.iloc[len(train):,]
train['Survived'] = survived
#drawing heatmap using adjusted dataset for learing to classifier
plt.figure(figsize=[12,12])
sns.heatmap(train.drop('PassengerId',1).corr(),vmax=0.6,square=True,annot=True)
#training,testing = train_test_split(train,test_size=0.2,random_state=0)
#Pclass and Sex is main factor of gaining survived value
#According to the heatmap above, we gained the columns what we would use for learning classifiers

cols = ['Pclass','Sex','SibSp','Parch','Embarked','Child','Young',"Age_known",'Shared_ticket']
tcols = np.append(['Survived'],cols)
df = train.loc[:,tcols].dropna()

X = train.loc[:,cols]
y = np.ravel(df.loc[:,"Survived"])


#X_test = testing.loc[:,cols]
#y_test = np.ravel(df.loc[:,"Survived"])

#X.shape
#X_test.shape
submission = pd.read_csv('../input/gender_submission.csv')

X_test = test.loc[:,cols]
y_test = submission['Survived']
#KNearestNeighbor
knn = KNeighborsClassifier(n_neighbors=10,weights='distance')
knn.fit(X,y)
knn_train_score = knn.score(X,y)
knn_test_score = knn.score(X_test,y_test)
print('the score of traning set when using KNearestNeighbor is {}'.format(knn_train_score))
print('the score of test set when using KNearestNeighbor is {}'.format(knn_test_score))
lr = LogisticRegression()
print(lr)
#LogisticRegrssion

#cvalues = [0.001,0.01,0.1,1,10,100,1000]
#lr_train_scores = []
#lr_test_scores = []
#for cvalue in cvalues:
#    lr = LogisticRegression(C=cvalue)
#    lr.fit(X,y)
#    lr_train_scores.append(lr.score(X,y))
#    lr_test_scores.append(lr.score(X_test,y_test))
#print(lr_train_scores)
#print(lr_test_scores)
#when using 1000 for C value in lr is good perfomance

lr = LogisticRegression()
lr_params_ = {'C':[0.001,0.01,0.1,1,10,100,1000]}
grid_lr = GridSearchCV(lr,lr_params_,scoring='accuracy',cv=5)
grid_lr.fit(X,y)
lr_train_score = grid_lr.score(X,y)
lr_test_score = grid_lr.score(X_test,y_test)

print(grid_lr.best_params_)
print('the score of traning set when using LogisticRegression is {}'.format(lr_train_score))
print('the score of test set when using LogisticRegression is {}'.format(lr_test_score))
#Linear Support Vector Classifier

#cvalues = [0.001,0.01,0.1,1,10,100,1000]
#svc_train_scores = []
#svc_test_scores = []
#for cvalue in cvalues:
#    svc = LinearSVC(C=cvalue)
#    svc.fit(X,y)
#    svc_train_scores.append(svc.score(X,y))
#    svc_test_scores.append(svc.score(X_test,y_test))
#print(svc_train_scores)
#print(svc_test_scores)

# The best performance is showed when C value is 0.1[X] or 1
# via GridSearchCV we could find which C value is the best in this analaysis
# The answer is C=1

lsvc = LinearSVC()
lsvc_params_ = {'C':[0.001,0.01,0.1,1,10,100,1000]}
grid_lsvc = GridSearchCV(lsvc,lsvc_params_,scoring='accuracy',cv=5)

grid_lsvc.fit(X,y)
lsvc_train_score = grid_lsvc.score(X,y)
lsvc_test_score = grid_lsvc.score(X_test,y_test)

print(grid_lsvc.best_params_)
print('the score of traning set when using LinearSVC is {}'.format(lsvc_train_score))
print('the score of test set when using LinearSVC is {}'.format(lsvc_test_score))
#Naive Bayes
NB = GaussianNB()
NB.fit(X,y)
NB_train_score = NB.score(X,y)
NB_test_score = NB.score(X_test,y_test)
print(NB_train_score)
print(NB_test_score)
#DecisionTreeClassifier

#depths = range(1,11)
#tree_train_scores = []
#tree_test_scores = []
#for depth in depths:
#    tree = DecisionTreeClassifier(max_depth = depth)
#    tree.fit(X,y)
#    tree_train_scores.append(tree.score(X,y))
#    tree_test_scores.append(tree.score(X_test,y_test))

#print(tree_train_scores)
#print(tree_test_scores)

# The best max_depth is 6
tree = DecisionTreeClassifier(max_depth = 6)
tree_params_ = {'max_depth':range(1,11)}
grid_tree = GridSearchCV(tree,tree_params_,scoring='accuracy',cv=5)
grid_tree.fit(X,y)
tree_train_score = grid_tree.score(X,y)
tree_test_score = grid_tree.score(X_test,y_test)

print(grid_tree.best_params_)
print('the score of traning set when using DecisionTree is {}'.format(tree_train_score))
print('the score of test set when using DecisionTree is {}'.format(tree_test_score))
#RandomForest
rf = RandomForestClassifier(random_state=0)
rf.fit(X,y)
rf_train_score = rf.score(X,y)
rf_test_score = rf.score(X_test,y_test)
print('the score of traning set when using RandomForest is {}'.format(rf_train_score))
print('the score of test set when using RandomForest is {}'.format(rf_test_score))
#GradientBoosting

#learning_rates = [0.001,0.01,0.1,1,10,100,1000]
#GB_train_scores = []
#GB_test_scores = []
#for rate in learning_rates:
#    GB = GradientBoostingClassifier(learning_rate = rate)
#    GB.fit(X,y)
#    GB_train_scores.append(GB.score(X,y))
#    GB_test_scores.append(GB.score(X_test,y_test))

#print(GB_train_scores)
#print(GB_test_scores)

#best one is when the learning rate is 0.1

#depths = range(1,11)

#for depth in depths:
#    GB = GradientBoostingClassifier(max_depth = depth)
#    GB.fit(X,y)
#    GB_train_scores.append(GB.score(X,y))
#    GB_test_scores.append(GB.score(X_test,y_test))
    
#def get_index_value(List):
#    maxValue = 0
#    idx = 0;
#    for index,value in enumerate(List):
#        if value > maxValue:
#            maxValue = value
#            idx = index
#    return idx+1,maxValue

#train_idx,train_max = get_index_value(GB_train_scores)
#test_idx,test_max = get_index_value(GB_test_scores)
#print('The best performance of trainig set {} could achieve when max_depths is {}'.format(train_max,train_idx))
#print('The best performance of test set {} could achieve when max_depths is {}'.format(test_max,test_idx))

"""
The best performance of trainig set 0.8720538720538721 could achieve when max_depths is 7
The best performance of test set 0.9832535885167464 could achieve when max_depths is 1
"""
GB = GradientBoostingClassifier()
GB_params = {'learning_rate':[0.001,0.01,0.1,1,10,100,1000],'max_depth':range(1,11)}
grid_GB = GridSearchCV(GB,GB_params,scoring='accuracy',cv=5)
grid_GB.fit(X,y)
GB_train_score = grid_GB.score(X,y)
GB_test_score = grid_GB.score(X_test,y_test)

print(grid_GB.best_params_)
print('the score of traning set when using Gradient Boosting is {}'.format(GB_train_score))
print('the score of test set when using Gradient Boosting is {}'.format(GB_test_score))
#SVC

#showing diff whether cleaning data or not for using SVC
svc = SVC()
svc.fit(X,y)
print(svc.score(X,y))
print(svc.score(X_test,y_test))
#SVC

#for using SVC. We need to clean our data to the range that it has btw 0 and 1
minValueOfTrainingset = X.min()
rangeOfTrainingset = (X - minValueOfTrainingset).max()
adj_train_df = (X - minValueOfTrainingset) / rangeOfTrainingset

minValueOfTestset = X_test.min()
rangeOfTestset = (X_test-minValueOfTestset).max()
adj_test_df = (X_test - minValueOfTestset)/ rangeOfTestset

svc = SVC()
svc.fit(adj_train_df,y)
print(svc.score(adj_train_df,y))
print(svc.score(adj_test_df,y_test))
"""
This was replaced with the result of GridSearchCV (18.11.25)

In SVC, C value and gamma could determine the result of our prediction
cvalues=[0.001,0.01,0.1,1,10,100,1000]
gammas=[0.01,0.1,1,10,100,1000]

#svc_train_scores = []
#svc_test_scores = []


for cvalue in cvalues:
    svc = SVC(C=cvalue)
    svc.fit(adj_train_df,y)
    svc_train_scores.append(svc.score(adj_train_df,y))
    svc_test_scores.append(svc.score(adj_test_df,y_test))
    
df_svc_train_c = pd.DataFrame({'cvalue':cvalues,'train_performance':svc_train_scores})
df_svc_test_c = pd.DataFrame({'cvalue':cvalues,'test_performance':svc_test_scores})
    
svc_train_scores = []
svc_test_scores = []

for gam in gammas:
    svc = SVC(gamma=gam)
    svc.fit(adj_train_df,y)
    svc_train_scores.append(svc.score(adj_train_df,y))
    svc_test_scores.append(svc.score(adj_test_df,y_test))

df_svc_train_gam = pd.DataFrame({'gamma':gammas,'train_performance':svc_train_scores})
df_svc_test_gam = pd.DataFrame({'gamma':gammas,'test_performance':svc_test_scores})

df_svc_c = pd.merge(df_svc_train_c,df_svc_test_c,on='cvalue',copy=False)
df_svc_gam = pd.merge(df_svc_train_gam,df_svc_test_gam,on='gamma',copy=False)
"""
"""fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x="cvalue",y="train_performance", data=df_svc_c)
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x="cvalue",y="test_performance", data=df_svc_c)
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x="gamma",y="train_performance", data=df_svc_gam)
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x="gamma",y="test_performance", data=df_svc_gam)

#We could find values (C,gamma) for increasing the predicting performance 
# C=0.1 or gamma = 0.01
"""
svc = SVC()
svc_params_ ={'C':[0.001,0.01,0.1,1,10,100,1000],'gamma':[0.01,0.1,1,10,100,1000]}
grid_svc = GridSearchCV(svc,svc_params_,scoring='accuracy',cv=5)
grid_svc.fit(X,y)
svc_train_score = grid_svc.score(X,y)
svc_test_score = grid_svc.score(X_test,y_test)

print(grid_svc.best_params_)
print('the score of traning set when using Gradient Boosting is {}'.format(svc_train_score))
print('the score of test set when using Gradient Boosting is {}'.format(svc_test_score))
#Let's gathering the result of our ML methods
df_prediction = pd.DataFrame({'method':['knn_test_score','lr_test_score','lsvc_test_score','NB_test_score','tree_test_score','rf_test_score','GB_test_score','svc_test_score'],
                              'value':[knn_test_score,lr_test_score,lsvc_test_score,NB_test_score,tree_test_score,rf_test_score,GB_test_score,svc_test_score]})
#Visualizing the values of ML methods
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=[12,10])
sns.barplot(x='method',y='value',data=df_prediction)

#Accoring to our prediction, svc has the best performance 
df_prediction
prediction = grid_lsvc.predict(X_test)
submission['Survived'] = prediction
submission.to_csv('../working/submit.csv',index=False)