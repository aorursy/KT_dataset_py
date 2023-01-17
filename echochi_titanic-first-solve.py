import matplotlib

from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline

import seaborn as sns



import numpy as np

import pandas as pd



pd.options.display.max_columns = 100

pd.options.display.max_rows = 100



from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing

from sklearn.cross_validation import cross_val_score

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Combining both the test and trainig data so that all the manipulations which are done

# happen on both the data sets.

# Also if test set has any missing values, it will easily come to notice here

def get_combined_data():

      

    combined = pd.concat([train, test], ignore_index=True)

    #keep columns in the same order

    combined = combined[list(train.columns.values)]

    

    #combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index',inplace=True,axis=1)

    return combined

combined = get_combined_data()
combined.tail()
combined.info()
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

combined.head()
grouped_fare = combined.head(891).groupby(['Pclass','FamilySize'])

grouped_fare.median()
col_admit = pd.DataFrame(combined.groupby('Ticket').size().reset_index(name='Admit'))



combined = pd.merge(combined, col_admit, on='Ticket', how='outer')

combined = combined.sort_values(by='PassengerId', ascending=True).reset_index(drop=True)



combined['FarePp'] = combined['Fare'] / combined['Admit']



combined.head()
grouped_title = combined.head(891).groupby(['Pclass'])

grouped_title.describe()
def get_titles():



    global combined

    

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

    Title_Dictionary = {

                       "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Mrs",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"



                        }

    

    # we map each title

    combined['Title'] = combined.Title.map(Title_Dictionary)

    

get_titles()

    

combined.head()
grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])

grouped_train.median()
#fill NaN ages with grouped medians based on social status as determined by titles

combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))

#fill 0 fares with NaN so they do not feature in median calculations and get replaced

combined['Fare'] = combined.Fare.apply(lambda x: x if x>0 else pd.np.nan)

#fill 0 fares and NaN fares with grouped medians based on social status

combined["Fare"] = combined.groupby(['Sex','Pclass','Title'])['Fare'].transform(lambda x: x.fillna(x.median()))



combined['FarePp'] = combined['Fare'] / combined['Admit']



grouped_title = combined.head(891).groupby(['Pclass'])

grouped_title.describe()

combined.info()
def get_num():

    

    global combined

    

    # some cabins have no digits, causing trouble when we use 'findall'. Add a digit where none is found

    combined['Cabin'] = combined['Cabin'].map(lambda x: x+'1' if re.findall(r"(\d+)",x) == [] else x)

    # the Num feature will be a single complex feature

    combined['Num'] = combined['Cabin'].apply(lambda x : re.findall(r"(\d+)", x )[0])

    # we can convert it into a few dunny variables

    combined['Section'] = combined['Num'].transform(lambda x : math.ceil((float(x))/20))

    combined['Section'] = combined['Section'].transform(lambda x : str(x))

    

    Boat_Section = {

                        "0":        "unknown",

                        "1":        "stern",

                        "2":        "sternpropeller",

                        "3":        "sternengine",

                        "4":        "midboiler",

                        "5":        "foreboiler",

                        "6":        "foreluggage",

                        "7":        "forecargo",

                        "8":        "fore"

                       

                        }

    # we map each group of cabin numbers onto a boat section which will give us 

    # names for section dummy features 

    combined['Section'] = combined.Section.map(Boat_Section)



def get_decks():

    

    global combined 

    global fillcombined

    # the Deck feature will become dummy variables

    combined['Deck'] = combined['Cabin'].map(lambda c : str(c)[0])

    # a single cabin number was reported as "T", but as that is the Tank Top which has no cabins

    # the next deepest deck level is substituted.

    combined['Deck'] = combined['Deck'].map(lambda c : 'U' if c == 'T' else c)



     # a map of titanic decks with 0 being the boat deck where the lifeboats were found,

    # and higher numbers indicate being furhter below decks

    Deck_Level = {

                        "A":        "7",

                        "B":        "6",

                        "C":        "5",

                        "D":        "4",

                        "E":        "3",

                        "F":        "2",

                        "G":        "1",

                        "U":        "0"

                        }

    

    # we map each deck onto a number level which will give us one complex Level feature 

    combined['Level'] = combined.Deck.map(Deck_Level)



import math

import re

#fillcombined = combined[['PassengerId', 'Pclass', 'Cabin', 'Embarked']]

#get_cabin_embarked()   

combined.Cabin.fillna('U0',inplace=True)

get_num()

get_decks()



combined.head()

data = combined
survived_sex = data[data['Survived']==1]['Sex'].value_counts()

dead_sex = data[data['Survived']==0]['Sex'].value_counts()

#plot the survived male , female and dead male,female

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar', figsize=(15,8))
# dead and survived based on age of people

figure = plt.figure(figsize=(15,8))

plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], color = ['g','r'],

         bins = 15,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
# plotting number of survivors based on the fare they gave

figure = plt.figure(figsize=(15,8))

plt.hist([data[data['Survived']==1]['FarePp'],data[data['Survived']==0]['FarePp']], color = ['g','r'],

         bins = 40,label = ['Survived','Dead'])

plt.xlabel('FarePp')

plt.ylabel('Number of passengers')

plt.legend()
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['FarePp'],c='green',s=40, alpha=0.4)

ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['FarePp'],c='red',s=40,  alpha=0.4)

ax.set_xlabel('Age')

ax.set_ylabel('FarePp')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=20,)
# The size of families on the training set

plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('FamilySize')

ax.hist([data[data['Survived']==1]['FamilySize'],data[data['Survived']==0]['FamilySize']],color = ['g','r'],)
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('Pclass')

ax.hist([data[data['Survived']==1]['Pclass'],data[data['Survived']==0]['Pclass']],color = ['g','r'],)
dataF = data[data['Sex']=='female']

dataM = data[data['Sex']=='male']
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('Pclass')

ax.hist([dataF[dataF['Survived']==1]['Pclass'],dataF[dataF['Survived']==0]['Pclass']],color = ['g','r'],)
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.set_ylabel('Survived')

ax.set_xlabel('Pclass')

ax.hist([dataM[dataM['Survived']==1]['Pclass'],dataM[dataM['Survived']==0]['Pclass']],color = ['g','r'],)
#is there a better ratio of survival based on where you embarked?

embark_S = data[data['Embarked']=='S']['Survived'].value_counts()

embark_C = data[data['Embarked']=='C']['Survived'].value_counts()

embark_Q = data[data['Embarked']=='Q']['Survived'].value_counts()

df = pd.DataFrame([embark_S,embark_C,embark_Q])

df.index = ['S','C','Q']

df.plot(kind='bar', figsize=(15,8))

embarked = data

embarked['Embarked'] = embarked['Embarked'].map({'S':0,'C':1, 'Q':2})
embark_1 = embarked[embarked['Pclass']==1]['Embarked']

embark_2 = embarked[embarked['Pclass']==2]['Embarked']

embark_3 = embarked[embarked['Pclass']==3]['Embarked']



plt.figure(figsize=(15,8))

plt.xlabel('Embarked')

plt.hist([embark_1, embark_2 ,embark_3], bins=3, label=['1st', '2nd', '3rd'])

plt.legend(loc='upper right')

plt.show()


plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(embarked[embarked['Pclass']==1]['Embarked'],embarked[embarked['Pclass']==1]['PassengerId'],c='green',s=40, alpha=0.7)

ax.scatter(embarked[embarked['Pclass']==2]['Embarked'],embarked[embarked['Pclass']==2]['PassengerId'],c='red',s=40,  alpha=0.3)

ax.scatter(embarked[embarked['Pclass']==3]['Embarked'],embarked[embarked['Pclass']==3]['PassengerId'],c='yellow',s=40,  alpha=0.1)

ax.set_xlabel('Embarked')

ax.set_ylabel('PassengerId')

ax.legend(('1st Class','2nd Class','3rd Class'),scatterpoints=1,loc='upper right',fontsize=20,)

plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(dataF[dataF['Survived']==1]['Num'],dataF[dataF['Survived']==1]['Level'],c='green',s=40, alpha=0.6)

ax.scatter(dataF[dataF['Survived']==0]['Num'],dataF[dataF['Survived']==0]['Level'],c='red',s=40,  alpha=0.6)

ax.set_xlabel('Num')

ax.set_ylabel('Level')

ax.legend(('female survived','female dead'),scatterpoints=1,loc='upper right',fontsize=20,)
plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(dataM[dataM['Survived']==1]['Num'],dataM[dataM['Survived']==1]['Level'],c='green',s=40, alpha=0.5)

ax.scatter(dataM[dataM['Survived']==0]['Num'],dataM[dataM['Survived']==0]['Level'],c='red',s=40,  alpha=0.5)

ax.set_xlabel('Num')

ax.set_ylabel('Level')

ax.legend(('male survived','male dead'),scatterpoints=1,loc='upper right',fontsize=20,)
combined.info()
# new columns showing age ranges

# based on the histograms that showed different survival rates. Tweens did surprisingly poorly,

# so distinguish as a separate feature

combined['Age_Child'] = combined['Age'].apply(lambda x: 1 if x>=0 and x<=7 else 0)

combined['Age_Tween'] = combined['Age'].apply(lambda x: 1 if x>=8 and x<=12 else 0)

combined['Age_Older'] = combined['Age'].apply(lambda x: 1 if x>=15 else 0)

combined['Age_Teen'] = combined['Age'].apply(lambda x: 1 if x>=13 and x<=17 else 0)

combined['Age_Twenties'] = combined['Age'].apply(lambda x: 1 if x>=18 and x<=32 else 0)

combined['Age_MiddleAged'] = combined['Age'].apply(lambda x: 1 if x>=33 and x<=48 else 0)

combined['Age_Elderly'] = combined['Age'].apply(lambda x: 1 if x>=49 else 0)



combined.head()
combined['Family_Single'] = combined['FamilySize'].apply(lambda x: 1 if x<=1 else 0)

combined['Family_Small'] = combined['FamilySize'].apply(lambda x: 1 if x>=2 and x<=4 else 0)

combined['Family_Large'] = combined['FamilySize'].apply(lambda x: 1 if x>=5 else 0)



combined.head()
combined['Poor'] = combined['FarePp'].apply(lambda x: 1 if x<=22 else 0)

combined.head()
def get_one_hot_encoding(dt, features):

    for feature in features:

        if feature in dt.columns:

            dummies = pd.get_dummies(dt[feature],prefix=feature)

            dt = pd.concat([dt,dummies],axis=1)

    return dt
combined = get_one_hot_encoding(combined,['Pclass', 'Title', 'Deck', 'Section']) 

#'Embarked' was removed for one hot coding since it mostly corresponds to class



combined['Sex'] = combined['Sex'].map({'male':0,'female':1})



combined.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'Section', 'Deck'],inplace=True,axis=1)



combined.head()
columns = combined.columns

combined_new = pd.DataFrame(preprocessing.normalize(combined, axis=0, copy=True), columns=columns)

combined = combined_new
combined.head()
trainY = train.Survived

trainX = combined[0:891]

testX = combined[891:]

trainX.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')

#clf = clf.fit(train, targets)

#clf = ExtraTreesClassifier(n_estimators=500)

clf = clf.fit(trainX, trainY)

features = pd.DataFrame()

features['feature'] = trainX.columns

features['importance'] = clf.feature_importances_

#this one we've done manually for now

#cols =  features.sort(['importance'],ascending=False)['feature']

cols =  pd.DataFrame({'features':['FarePp','Age','Title_Mr','Sex','Admit','Title_Mrs','Title_Miss','Pclass','Num','Deck_U']})

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))

model = SelectFromModel(clf, prefit=True)

train_new = model.transform(trainX)

test_new = model.transform(testX)

train_new.shape
train_select = trainX[['FarePp','Age','Title_Mr','Sex','Admit','Title_Mrs','Title_Miss','Pclass','Num','Deck_U']]

test_select = testX[['FarePp','Age','Title_Mr','Sex','Admit','Title_Mrs','Title_Miss','Pclass','Num','Deck_U']]

train_select.head()
forest = RandomForestClassifier()



parameter_grid = {

                 'max_depth' : [4, 6, 8, 10],

                 'n_estimators': [50, 10],

                 'criterion': ['gini','entropy'],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [2, 3, 10],

                 'bootstrap': [True, False],

                 }



cross_validation = StratifiedKFold(trainY, n_folds=5)



grid_search = GridSearchCV(forest,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_select, trainY)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
ext = ExtraTreesClassifier()



parameter_grid = {

                 'max_depth' : [4,5,6,7,8,9,10,11,12],

                 'n_estimators': [100, 200, 300],

                 'criterion': ['gini','entropy']

                 }



cross_validation = StratifiedKFold(trainY, n_folds=5)



grid_search = GridSearchCV(ext,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_select, trainY)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
lr = LogisticRegression(penalty='l2')



parameter_grid = {

                 'tol' : [0.1,0.01,0.001,1,10],

                 'max_iter': [200, 300],

                 }



cross_validation = StratifiedKFold(trainY, n_folds=5)



grid_search = GridSearchCV(lr,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_select, trainY)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=300,learning_rate=0.6)



cross_validation = StratifiedKFold(trainY, n_folds=5)

adaboost.fit(train_select, trainY)



print('Best score: {}'.format(cross_val_score(adaboost,train_select,trainY,cv=10)))
import xgboost as xgb



parameter_grid = {

    'n_estimators': 300,

    'learning_rate' : 0.5,

    'max_depth': 6,

    'booster': 'gbtree',

    'min_child_weight': 1

}



gbm = xgb.XGBClassifier(**parameter_grid)



gbm.fit(train_select, trainY,

    eval_set = [(train_select, trainY),(train_select, trainY)], eval_metric='error', early_stopping_rounds = 200, verbose=False)



evals_result = gbm.evals_result()



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)

knn.fit( train_select , trainY )



print (knn.score( train_select , trainY ))
from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[

        ('rf', forest),('etc',ext),('knn', knn), ('adb', adaboost)], voting='soft',

                        weights=[3,1,1,1])

eclf1 = eclf1.fit(train_select, trainY)

predictions=eclf1.predict(test_select)

predictions

cross_validation = StratifiedKFold(trainY, n_folds=5)



test_predictions=eclf1.predict(test_select)

test_predictions=test_predictions.astype(int)

print('Best score: {}'.format(cross_val_score(eclf1,train_select,trainY,cv=10)))
test_predictions = eclf1.predict(test_select)

#df_output = pd.DataFrame()

#df_output['PassengerId'] = test['PassengerId']

#df_output['Survived'] = test_predictions

#df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

test_predictions
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

    'warm_start': True, 

    'bootstrap': True, 

    'criterion': 'gini',

    'max_depth': 10,

    'min_samples_split': 10,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    'criterion': 'entropy',

    'max_depth': 10,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.6

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 10,

    'min_samples_leaf': 2,

    'verbose': 0

}

NFOLDS = 5

kf = StratifiedKFold(trainY,n_folds= NFOLDS)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        return self.clf.fit(x,y).feature_importances_
# Create 5 objects that represent our 4 models

SEED=0



rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

def get_oof(clf, x_train, y_train, x_test):

    ntrain = train.shape[0]

    ntest = test.shape[0]

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        # get the training of fold number i from training set

        x_tr = train_select.iloc[train_index]

        # get the targets of fold i from training set

        y_tr = trainY.iloc[train_index]

        # get the remaining 10% test set from the ith fold 

        x_te = train_select.iloc[test_index]



        # train the classifier on the training set

        clf.train(x_tr, y_tr)

        

        # store results of predictions over the ith test set at proper locations

        # oof_train will contain all the predictions over the test set once all n_fold iterations are over

        oof_train[test_index] = clf.predict(x_te)

        # over the complete test set classifier trained so far will predict

        # ith entry of oof_test_skf will contain predictions from classifier trained till ith fold

        oof_test_skf[i, :] = clf.predict(x_test)



    # calculate mean of all the predictions done in the i folds and store them as final results in oof_test

    oof_test[:] = oof_test_skf.mean(axis=0)

    # predictions on training set, mean predictions on the test set

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, train_select, trainY, test_select) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,train_select, trainY, test_select) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, train_select, trainY, test_select) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,train_select, trainY, test_select) # Gradient Boost

#gbm_oof_train, gbm_oof_test = get_oof(gbm,train_select, trainY, test_select) # XGBoost



print("Training is complete")
rf_feature = rf.feature_importances(train_select, trainY)

et_feature = et.feature_importances(train_select, trainY)

ada_feature = ada.feature_importances(train_select, trainY)

gb_feature = gb.feature_importances(train_select, trainY)

#gbm_feature = gbm.feature_importances(train_select, trainY)
cols_new = cols['features'].values

print(cols_new)
# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols_new,

     'Random Forest feature importances': rf_feature,

     'Extra Trees  feature importances': et_feature,

     'AdaBoost feature importances': ada_feature,

     'Gradient Boost feature importances': gb_feature

    })
# The final dataframe

feature_dataframe.head(10)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

     'GradientBoost': gb_oof_train.ravel()

         })

base_predictions_train.head(10)
import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)

data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Portland',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
#converted into a single array of training set(891) X 4 columns(number of classifiers)

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
from sklearn import tree

cross_validation = StratifiedKFold(trainY, n_folds=5)

clf = tree.DecisionTreeClassifier(max_depth=10, max_features='sqrt').fit(x_train, trainY)



print('Best score: {}'.format(cross_val_score(clf,train_select,trainY,cv=10)))

predictions = clf.predict(x_test)
#our stacked output

predictions
df_output = pd.DataFrame()

df_output['PassengerId'] = test['PassengerId']

df_output['Survived'] = predictions

df_output[['PassengerId','Survived']].to_csv('titanic_predictions.csv',index=False)