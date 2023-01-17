# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fileNameTrain = '/kaggle/input/titanic/train.csv'

fileNameTest  = '/kaggle/input/titanic/test.csv'



dfTrain = pd.read_csv(fileNameTrain)

dfTest  = pd.read_csv(fileNameTest)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

import lightgbm as lgb





from xgboost import XGBRegressor

from xgboost import XGBClassifier



import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns; sns.set(style="ticks", color_codes=True)

import matplotlib.pyplot as plt

%matplotlib inline



print(lgb.__version__)
dfTrain.head()
for df in [dfTrain,dfTest]:

    lastName = [x.split(', ')[0] for x in df.Name] 

    title    = [x.split(', ')[1].split('. ')[0] for x in df.Name]

    #replace some titles

    title    = [x.replace('Mme', 'Mrs') for x in title]

    title    = [x.replace('Ms', 'Miss') for x in title]

    title    = [x.replace('Mlle', 'Miss') for x in title]

    title    = [x.replace('the Countess', 'Lady') for x in title]

    title    = [x.replace('Jonkheer', 'Sir') for x in title]

    title    = [x.replace('Capt', 'Col') for x in title]

    title    = [x.replace('Dona', 'Mrs') for x in title]

    title    = [x.replace('Don', 'Mr') for x in title]

    

    title    = [x.replace('Dr', 'Col') for x in title]

    title    = [x.replace('Major', 'Col') for x in title]

    title    = [x.replace('Rev', 'Col') for x in title]

    title    = [x.replace('Sir', 'Col') for x in title]



    df['Lastname'] = lastName

    df['Title']    = title

    

print(dfTrain['Title'].unique())

print(dfTest['Title'].unique())
for df in [dfTrain,dfTest]:

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    

for df in [dfTrain,dfTest]:    

    df['IsAlone'] = 0

    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
for df in [dfTrain,dfTest]:

    ticketClean = [x for x in df.Ticket]

    ticketClean = [x.replace('W./C.', 'W/C') for x in ticketClean]

    ticketClean = [x.replace('STON/O 2', 'STON/O2') for x in ticketClean]

    ticketClean = [x.replace('C.A.', 'CA') for x in ticketClean]

    ticketClean = [x.replace('CA.', 'CA') for x in ticketClean]

    ticketClean = [x.replace('A.', 'A') for x in ticketClean]

    ticketClean = [x.replace('A/', 'A') for x in ticketClean]

    ticketClean = [x.replace('SOTON/O.Q.', 'SOTON/OQ') for x in ticketClean]

    ticketClean = [x.replace('A4.', 'A4') for x in ticketClean]

    ticketClean = [x.replace('A5.', 'A5') for x in ticketClean]

    ticketClean = [x.replace('W.E.P.', 'WEP') for x in ticketClean]

    ticketClean = [x.replace('WE/P', 'WEP') for x in ticketClean]

    ticketClean = [x.replace('A 2.', 'A2') for x in ticketClean]



    df['TicketClean'] = ticketClean
freq_port = dfTrain.Embarked.dropna().mode()[0]

dfTrain['Embarked'] = dfTrain['Embarked'].fillna(freq_port)

dfTest['Embarked']  = dfTest['Embarked'].fillna(freq_port)
label_encoder = LabelEncoder()

for col in sorted(['Sex','Ticket','Lastname','Title','TicketClean','Embarked']):

    label_encoder.fit(dfTrain[col].append(dfTest[col]))

    dfTrain[col] = label_encoder.transform(dfTrain[col])

    dfTest[col] = label_encoder.transform(dfTest[col])

    print(col,' - done')



# Manuel version    

#for dataset in [dfTrain, dfTest]:

#    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
dfTrain['XLastname'] = dfTrain['Lastname'].to_numpy() -430

dfTest['XLastname'] = dfTest['Lastname'].to_numpy() -430



dfTrain['XPclass'] = dfTrain['Pclass'].to_numpy() -2

dfTest['XPclass'] = dfTest['Pclass'].to_numpy() -2
cols = dfTrain.columns.to_list()

cols.remove('Survived')

dfCombined = dfTrain[cols].append(dfTest[cols])



df=dfCombined

print('Train data: ',dfTrain.shape[0])

print('Test  data: ',dfTest.shape[0])

print('all   data: ',dfCombined.shape[0])

dfCombinedAgeMissing = dfCombined[dfCombined.Age.isna()]

dfCombinedAgeIn      = dfCombined[dfCombined.Age > 0]

print('all   data - age NOT missing:    ',dfCombinedAgeIn.shape[0])

print('all   data - age missing:        ',dfCombinedAgeMissing.shape[0])
pd.set_option("display.max_rows", None)

pd.set_option("display.max_rows", 40)

dfCombined
allColls = dfTrain.columns.to_list()

print(allColls)

feature_names = allColls

feature_names.remove('Survived')

feature_names.remove('Fare')

feature_names.remove('Cabin')

#feature_names.remove('Embarked')

feature_names.remove('Name')

feature_names.remove('Age')



Xage = dfCombinedAgeIn[feature_names]

yage = dfCombinedAgeIn['Age']



train_Xage, valid_Xage, train_yage, valid_yage = train_test_split(Xage, yage, random_state = 1)

print('SampleSize of Age training set: ',Xage.shape[0],train_Xage.shape[0],valid_Xage.shape[0],train_Xage.shape[0]/Xage.shape[0])

#train_Xage.head()



test_Xage = dfCombinedAgeMissing[feature_names]

test_Xage.head()
nbrOfNodes = []

errors     = []

for k in range(3,50,1):

    treeModel = DecisionTreeRegressor(max_leaf_nodes=k, random_state=0)

    treeModel.fit(train_Xage, train_yage)

    preds_val = treeModel.predict(valid_Xage)

    mae = mean_absolute_error(valid_yage, preds_val)

    nbrOfNodes.append(k)

    errors.append(mae)



plt.plot(nbrOfNodes,errors)
nbrOfNodes = []

errors     = []

for k in range(25,400,225):

    print(k,end=' ')

    forestModel = RandomForestRegressor(n_estimators=k, 

                                        random_state=0, 

                                        criterion='mae', 

                                        min_samples_split=5,

                                        max_depth=10

                                        )

    forestModel.fit(train_Xage, train_yage)

    preds_val = forestModel.predict(valid_Xage)

    mae = mean_absolute_error(valid_yage, preds_val)

    nbrOfNodes.append(k)

    errors.append(mae)



plt.plot(nbrOfNodes,errors)
xgboostModel = XGBRegressor(objective ='reg:squarederror',

                        n_estimators=400,

                        learning_rate=0.15,

                        n_jobs=4)



xgboostModel.fit(train_Xage, train_yage, 

             early_stopping_rounds=15, 

             eval_set=[(valid_Xage, valid_yage)],

             verbose=False)



preds_val = xgboostModel.predict(valid_Xage)

mae = mean_absolute_error(valid_yage, preds_val)

print(mae)
forestModel = RandomForestRegressor(n_estimators=200, 

                                        random_state=0, 

                                        criterion='mae', 

                                        min_samples_split=5,

                                        max_depth=10

                                        )



forestModel.fit(train_Xage, train_yage)

predictesAges = forestModel.predict(test_Xage)

#dfCombinedAgeMissing['estAge']=predictesAges
passId = test_Xage['PassengerId'].to_numpy()

predAges = predictesAges

dictPassId_predAges = {}

for k in range(len(passId)):

    dictPassId_predAges[passId[k]]=predAges[k]

#dictPassId_predAges
allColls = dfTrain.columns.to_list()

print(allColls)

feature_names2 = allColls

feature_names2.remove('Survived')

feature_names2.remove('Cabin')

feature_names2.remove('Name')

#feature_names2.remove('Embarked')

#feature_names2.remove('Fare')



dfTrain2 = dfTrain[feature_names2].copy()

dfTrain2['estAge'] = 'NA'



dfTest2 = dfTest[feature_names2].copy()

dfTest2['estAge'] = 'NA'
for k in range(1,dfTrain2.shape[0]+1,1):

    dfTrain2.at[k-1,'estAge'] = dictPassId_predAges.get(k,dfTrain2.iloc[k-1]['Age'])



for k in range(1,dfTest2.shape[0]+1,1):

    j = dfTest2.loc[k-1].PassengerId

    dfTest2.loc[k-1,'estAge'] = dictPassId_predAges.get(j,dfTest2.iloc[k-1]['Age'])
bandSize = 16

dfTrain2['estAgeBand'] = (dfTrain2['estAge']/bandSize).astype('int') -3

dfTest2['estAgeBand'] = (dfTest2['estAge']/bandSize).astype('int') -3



set(dfTrain2['estAgeBand'])
bandSize = 16

freq_Fare = dfTrain.Fare.dropna().mode()[0]

dfTrain2['Fare'] = dfTrain2['Fare'].fillna(freq_Fare)



dfTest2['Fare']  = dfTest2['Fare'].fillna(freq_Fare)



dfTrain2['FareBand'] = (dfTrain2['Fare']/bandSize).astype('int') 

dfTest2['FareBand'] = (dfTest2['Fare']/bandSize).astype('int') 



thresh = 4

dfTrain2['FareBand'] = [x if x < thresh else thresh for x in dfTrain2['FareBand'].to_numpy()] 

dfTest2['FareBand'] = [x if x < thresh else thresh for x in dfTest2['FareBand'].to_numpy()] 



set(dfTrain2['FareBand'])
dfTrain2['ChildP12']=((dfTrain2['estAge']<16) & (dfTrain2['Pclass']<3)).map( {True: 1, False: 0} ).astype(int)

dfTest2['ChildP12'] =((dfTest2['estAge']<16)  & (dfTest2['Pclass']<3)).map( {True: 1, False: 0} ).astype(int)
pd.set_option("display.max_rows", None)

pd.set_option("display.max_rows", 40)

dfTrain2

#dfTest2
feature_names3=['PassengerId','XPclass','Sex','SibSp','Parch','Ticket','TicketClean','XLastname','Title','estAge','estAgeBand','Embarked','FamilySize','IsAlone','FareBand','ChildP12']

feature_names3=['IsAlone','Sex','XPclass','FareBand','estAgeBand','XLastname','ChildP12']

#feature_names3=['Sex','FareBand','IsAlone','ChildP12','estAgeBand','Title']



dfTrain2['estAge'] = pd.to_numeric(dfTrain2['estAge'])

dfTest2['estAge'] = pd.to_numeric(dfTest2['estAge'])



X = dfTrain2[feature_names3]

y = dfTrain['Survived']



train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 1)

print('SampleSize of Age training set: ',X.shape[0],train_X.shape[0],valid_X.shape[0],train_X.shape[0]/X.shape[0])



XTest = dfTest2[feature_names3]
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(dfTrain2.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
X = valid_X

y = valid_y



# Logistic Regression

logreg = LogisticRegression(solver='lbfgs',max_iter=10000)

logreg.fit(train_X, train_y)

preds_val = logreg.predict(valid_X)

cm_log = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_log

hit_log=(tp+tn)/(tn+fp+fn+tp)

acc_log = round(logreg.score(X, y) * 100, 2)



# Support Vector Machines

#svc = SVC(gamma='auto')

svc = SVC(gamma='scale',probability=True)

svc.fit(train_X, train_y)

preds_val = svc.predict(valid_X)

cm_svc = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_svc

hit_svc = (tp+tn)/(tn+fp+fn+tp)

acc_svc = round(svc.score(X, y) * 100, 2)



# K nearest neighbours

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(train_X, train_y)

preds_val = knn.predict(valid_X)

cm_knn = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_knn

hit_knn = (tp+tn)/(tn+fp+fn+tp)

acc_knn = round(knn.score(X, y) * 100, 2)



# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(train_X, train_y)

preds_val = gaussian.predict(valid_X)

cm_gaussian =  confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_gaussian

hit_gaussian = (tp+tn)/(tn+fp+fn+tp)

acc_gaussian = round(gaussian.score(X, y) * 100, 2)



# Perceptron

perceptron = Perceptron(max_iter=1000,tol=1e-4)

perceptron.fit(train_X, train_y)

preds_val = perceptron.predict(valid_X)

cm_perceptron = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_perceptron

hit_perceptron = (tp+tn)/(tn+fp+fn+tp)

acc_perceptron = round(perceptron.score(X, y) * 100, 2)



# Stochastic Gradient Descent

sgd = SGDClassifier(max_iter=10000,

                    tol=1e-4,

                    random_state=10)

sgd.fit(train_X, train_y)

preds_val = sgd.predict(valid_X)

cm_sgd = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_sgd

hit_sgd = (tp+tn)/(tn+fp+fn+tp)

acc_sgd = round(sgd.score(X, y) * 100, 2)



# Linear SVC

linear_svc = LinearSVC(max_iter=100000,dual=False)

linear_svc.fit(train_X, train_y)

preds_val = linear_svc.predict(valid_X)

cm_linear_svc = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_linear_svc

hit_linear_svc = (tp+tn)/(tn+fp+fn+tp)

acc_linear_svc = round(linear_svc.score(X, y) * 100, 2)



# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(train_X, train_y)

preds_val = decision_tree.predict(valid_X)

cm_decision_tree = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_decision_tree

hit_decision_tree = (tp+tn)/(tn+fp+fn+tp)

acc_decision_tree = round(decision_tree.score(X, y) * 100, 2)



# Random Forest

random_forest = RandomForestClassifier(n_estimators=1100, 

                                       #criterion='entropy',

                                       #max_features='sqrt'

                                       criterion='gini',

                                       max_features='auto',

                                       oob_score=True,

                                       random_state=0,

                                       min_samples_split=5,

                                       min_samples_leaf=5,

                                       max_depth=6,

                                       n_jobs=-1

                                      )



random_forest.fit(train_X, train_y)

preds_val = random_forest.predict(valid_X)

cm_random_forest = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_random_forest

hit_random_forest = (tp+tn)/(tn+fp+fn+tp)

random_forest.score(X, y)

acc_random_forest = round(random_forest.score(X, y) * 100, 2)



# ADAboost

adaBoost = AdaBoostClassifier(n_estimators=300, 

                                random_state=0,

                                learning_rate=0.1

                              )

adaBoost.fit(train_X, train_y)

preds_val = adaBoost.predict(valid_X)

cm_ada = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_ada

hit_ada = (tp+tn)/(tn+fp+fn+tp)

adaBoost.score(X, y)

acc_ada = round(adaBoost.score(X, y) * 100, 2)



# ExtraTrees

extraTrees = ExtraTreesClassifier(n_jobs=-1,

                                n_estimators=500,

                                #max_features=0.5,

                                max_depth=8,

                                min_samples_leaf=2)

extraTrees.fit(train_X, train_y)

preds_val = extraTrees.predict(valid_X)

cm_et = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_et

hit_et = (tp+tn)/(tn+fp+fn+tp)

extraTrees.score(X, y)

acc_et = round(extraTrees.score(X, y) * 100, 2)



# GradientBoosting

gb = GradientBoostingClassifier(n_estimators= 150,

                             max_depth= 5,

                             learning_rate=0.05,

                             min_samples_leaf= 2)

gb.fit(train_X, train_y)

preds_val = gb.predict(valid_X)

cm_gb = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_gb

hit_gb = (tp+tn)/(tn+fp+fn+tp)

gb.score(X, y)

acc_gb = round(gb.score(X, y) * 100, 2)



# XGBoost

xgboostModel = XGBClassifier(learning_rate = 0.03,

                             n_estimators= 2000,

                             max_depth= 5,

                             min_child_weight= 2,

                             #gamma=1,

                             gamma=0.9,                        

                             subsample=0.8,

                             colsample_bytree=0.8,

                             #objective= 'binary:logistic',

                             #objective ='reg:squarederror',

                             nthread= -1,

                             scale_pos_weight=1)

xgboostModel.fit(train_X, train_y)

preds_val = xgboostModel.predict(valid_X)

cm_xgb = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_xgb

hit_xgb = (tp+tn)/(tn+fp+fn+tp)

xgboostModel.score(X, y)

acc_xgb = round(xgboostModel.score(X, y) * 100, 2)



# lightgbm

gbm = lgb.LGBMClassifier(boosting_type='gbdt',

                        objective='binary',

                        num_leaves=31,

                        learning_rate=0.05,

                        #num_boost_round=100000,

                        n_estimators=200,

                        silent=-1,

                        verbose=-1)

gbm.fit(train_X, train_y,

        eval_set=[(valid_X, valid_y)],

        eval_metric='l1',

        early_stopping_rounds=5,

        verbose=100)

preds_val = gbm.predict(valid_X)

cm_gbm = confusion_matrix(valid_y, preds_val).ravel()

tn, fp, fn, tp = cm_gbm

hit_gbm = (tp+tn)/(tn+fp+fn+tp)

gbm.score(X, y)

acc_gbm = round(gbm.score(X, y) * 100, 2)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC','Decision Tree',

              'XGBoost','lightgbm','AdaBoost','extraTrees','GradientBoost'],

    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, 

              acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree, 

              acc_xgb, acc_gbm, acc_ada, acc_et, acc_gb],

    'hitScore': [cm_svc, cm_knn, cm_log, cm_random_forest, cm_gaussian, 

                 cm_perceptron, cm_sgd, cm_linear_svc, cm_decision_tree,

                 cm_xgb, cm_gbm, cm_ada, cm_et, cm_gb]})

pd.set_option("display.max_rows", None)

models.sort_values(by='Score', ascending=False)
print(train_X.columns.values)

print(decision_tree.feature_importances_)

print(extraTrees.feature_importances_)

print(random_forest.feature_importances_)

print(xgboostModel.feature_importances_)

print(gb.feature_importances_)

print(gbm.feature_importances_/sum(gbm.feature_importances_))

print(adaBoost.feature_importances_)
#estimate = np.around(gbm.predict(XTest)).astype(int)

estimate = np.around(logreg.predict(XTest)).astype(int)

df=pd.DataFrame(dfTest['PassengerId'])

df.insert(1,'Survived',estimate)

df.to_csv('submission.csv', index=False)