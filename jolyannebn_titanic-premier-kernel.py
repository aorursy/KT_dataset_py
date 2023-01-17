import numpy as np 

import pandas as pd 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

from matplotlib import pyplot

from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV

from sklearn import  metrics

from sklearn.metrics import log_loss

import xgboost as xgb

# On lit nos données d'entrée

train = pd.read_csv("../input/train.csv")

taille_train = len(train)

labels = train['Survived']



test = pd.read_csv("../input/test.csv")



# On concatène les deux sets de données pour faire le préprocessing plus rapidement

all_data = pd.concat([train, test],ignore_index=True, sort='True')
train.head()
# On regarde les valeurs nulles

all_data.isnull().sum()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
def get_name(name):

    lastName = name.split(',')

    title = lastName[1].split('.')

    if (lastName and title):

        return lastName[0], title[0], title[1]

    return ""



r = all_data['Name'].apply(get_name)



LastNames, titles, Firstames = [], [], []

for i in range(len(r)):

    LastNames.append(r[i][0])

    titles.append(r[i][1])

    Firstames.append(r[i][2])

    

all_data['LastName'] = LastNames

all_data['Title'] = titles

all_data['FirstName'] = Firstames
all_data['Title'].unique()
pd.crosstab(all_data['Title'], all_data['Sex'])
def combine_title(title):

    title = title.replace(" ","")

    title = title.replace('Mlle', 'Miss')

    title = title.replace('Ms', 'Miss')

    title = title.replace('Mme', 'Mrs')

    

    if title in ['Lady', 'theCountess', 'Don', 'Sir', 'Jonkheer', 'Dona']:

        return "Royal"

    elif title in ['Capt', 'Col', 'Dr', 'Major', 'Rev'] :

        return 'Other'

    else :

        return title



all_data['Title'] = all_data['Title'].apply(combine_title)
all_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = all_data[['FamilySize']].applymap(lambda x: 0 if x>1 else 1)
all_data['Age'].fillna(all_data['Age'].median(), inplace = True)
fare_moyen = all_data[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean()

fare_moyen
all_data.loc[(all_data['Fare'].isnull()) | (all_data['Fare']==0)].Fare
for i in range(3):

    all_data.loc[(all_data['Fare']==0) & (all_data['Pclass']==(i+1)),'Fare']  = fare_moyen.Fare[i]



#La valeur non définie a pour classe 3

all_data['Fare'] = all_data['Fare'].fillna(fare_moyen.Fare[2])
all_data.loc[(all_data['Embarked'].isnull())]
all_data.loc[(all_data['Cabin']=='B28')]
pd.crosstab(all_data['Pclass'], all_data['Embarked'])
# Le port principal d'embarquement des personnes de première classe est le port S

all_data['Embarked'] = all_data['Embarked'].fillna('S')
all_data['HasCabin'] = all_data[['Cabin']].applymap(lambda x: 0 if pd.isnull(x) else 1)
all_data['Title'] = all_data['Title'].map( {'Master': 1, 'Miss': 2, 'Mr': 3, 'Mrs':4, 'Other': 5, 'Royal':6} ).astype(int)

all_data['Embarked'] = all_data['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
all_data = all_data.drop(columns=['Name','LastName', 'FirstName','Ticket','Cabin', 'Parch', 'SibSp', 'PassengerId'])

all_data.head()
## On reforment nos 2 datasets: trainig / Testing



train = all_data.iloc[0:taille_train]

test = all_data.iloc[taille_train:]

#train['Survived'] = labels
# Matrice de corrélation pour voir quelles variables sont corrélées

colormap = plt.cm.viridis

plt.figure(figsize=(14,14))

plt.title('Matrice de corrélation', y=1.05, size=18)

sns.heatmap(train.astype(float).corr(),linewidths=0.08,vmax=1, square=True, cmap=colormap, linecolor='white', annot=True)
train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean().sort_values(by='Survived', ascending=False)
train = train.drop(columns=['Survived'])

test = test.drop(columns=['Survived'])

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

score_rf_train = random_forest.score(X_train, y_train)

score_rf_test = random_forest.score(X_test, y_test)

print("score de training : "+ str(score_rf_train) + "    score de testing : " + str(score_rf_test))
def logloss(true_label, predicted, eps=1e-15):

  p = np.clip(predicted, eps, 1 - eps)

  if true_label == 1:

    return -log(p)

  else:

    return -log(1 - p)
rf = RandomForestClassifier(n_estimators = 200)

rf.fit(X_train, y_train)



y_proba_test = rf.predict_proba(X_test)

y_proba_train = rf.predict_proba(X_train)



test_loss = log_loss(y_test, y_proba_test, eps = 1e-15, normalize = True, sample_weight = None, labels = None)

print("test cross entropy = " + str(test_loss))

train_loss = log_loss(y_train, y_proba_train, eps = 1e-15, normalize = True, sample_weight = None, labels = None)

print("train cross entropy = " + str(train_loss))
dtrain = xgb.DMatrix(X_train, label = y_train)

dtest = xgb.DMatrix(X_test, label = y_test)

watchlist = [(dtrain, 'train'), (dtest, 'valid')]
param = {'max_depth': 2, 'eta':0.02, 'subsample' : 0.6,'base_score' : 0.2, "objective" : "binary:logistic", "eval_metric": "logloss"} 



bst = xgb.train(param, dtrain, 10000, watchlist, early_stopping_rounds=150, verbose_eval=50)



y_pred = bst.predict(dtest)

loss = log_loss(y_test, y_pred, eps = 1e-15, normalize = True, sample_weight = None, labels = None)

print("cross entropy = " + str(loss)) # the mean loss per sample.
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3)

X_train['Survived']  = y_train

X_test['Survived']  = y_test



def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['Survived'].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='logloss', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    # On run l'algo sur es données

    alg.fit(dtrain[predictors], dtrain['Survived'],eval_metric='logloss')

        

    # On prédit le set d'entrainement

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    

    # On prédit le set de test

    dtest_predictions = alg.predict(dtest[predictors])

    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]

        

    # On affiche les infos:

    print ("\n Rapport d'informations")

    print ("Accuracy  sur le set d'entrainement: %.4g" % metrics.accuracy_score(dtrain['Survived'].values, dtrain_predictions))

    print ("Accuracy  sur le set de test: %.4g" % metrics.accuracy_score(dtest['Survived'].values, dtest_predictions))

    loss = log_loss(dtrain['Survived'], dtrain_predprob, eps = 1e-15, normalize = True, sample_weight = None, labels = None)

    print("cross entropy (train) = " + str(loss))

    loss = log_loss(dtest['Survived'], dtest_predprob, eps = 1e-15, normalize = True, sample_weight = None, labels = None)

    print("cross entropy (test) = " + str(loss)) 

    

    plot_importance(alg)

    pyplot.show()

    return alg
predictors = [x for x in X_train.columns if x not in ['Survived']]



xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=2,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)
m1 = modelfit(xgb1, X_train, X_test, predictors)
param_test1 = {

'n_estimators':[300,500],

 'max_depth':[4,5,6], 

 'min_child_weight':[4,5,6],

}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate = 0.01, n_estimators=500, max_depth=5,

 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='neg_log_loss',n_jobs=4,iid=False, cv=None)

gsearch1.fit(X_train[predictors], X_train['Survived'])

gsearch1.best_params_, gsearch1.best_score_
param_test1 = {

'learning_rate':[0.01,0.05,0.025],

'n_estimators':[500,520,540],

'max_depth':[5], 

'min_child_weight': [4]

}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate = 0.01, n_estimators=500, max_depth=5,

 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='neg_log_loss',n_jobs=4,iid=False, cv=None)

gsearch1.fit(X_train[predictors], X_train['Survived'])

gsearch1.best_params_, gsearch1.best_score_
param_test1 = {

'learning_rate':[0.01],

'n_estimators':[520],

'max_depth':[5], 

'min_child_weight': [4],

'gamma':[0.0, 0.05, 0.1],

'reg_alpha':[1e-5, 1e-2, 0.1, 1]

}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate = 0.01, n_estimators=520, max_depth=5,

 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='neg_log_loss',n_jobs=4,iid=False, cv=None)

gsearch1.fit(X_train[predictors], X_train['Survived'])

gsearch1.best_params_, gsearch1.best_score_
xgbfinal = XGBClassifier(

 learning_rate =0.01,

 n_estimators=520,

 max_depth=5,

 min_child_weight=4,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 reg_alpha=0.01,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)
mfinal = modelfit(xgbfinal, X_train, X_test, predictors)
y_pred = mfinal.predict(test)
testid = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({

        "PassengerId": testid["PassengerId"],

        "Survived": y_pred

    })
submission.head()
submission.to_csv("Titanic.csv", index = False)
# with open("submission_file.csv", 'w') as f:

#     f.write("PassengerId,,Survived\n")

#     for i in range(len(y_pred)):

#         f.write(str(testid[i])

#                 +','

#                 +str(y_pred[i])+'\n')