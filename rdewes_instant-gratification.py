# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# About the data

train.shape, test.shape
train.info()
test.info()
train.head(3).T
# Distribution of the data

train.describe()
# Verifying for NaN

sum(train.isna().any())
df = pd.concat([train, test], sort=False)
df[df['wheezy-copper-turtle-magic']==1].describe()
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import StratifiedKFold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import time



def get_best_param(x, y, modelo, params):

    grid = GridSearchCV(estimator=modelo, param_grid=params, cv=3, n_jobs=-1)

    grid_result = grid.fit(x, y)        

    return (grid_result.best_score_, grid_result.best_params_)



# Treina tudo e acha o melhor

def testa_tudo(train, alvo):

    '''

    Função que tenta um monte de algoritmo de classificação e ve qual que presta. Demora um pouco.

    '''

    start_time = time.time()

    melhores = dict()

    # modelos

    models = {

        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),

    }

    # parametros

    params = {

        'QuadraticDiscriminantAnalysis': {'tol': [1,0.1,0.01,0.001,0.0001,0.00001]}

    }

    for nome, modelo in models.items():

        param = params[nome]

        melhores[nome] = get_best_param(train, alvo, modelo, param)

        print("Modelo: %s, melhor classificação: %f usando %s" % (nome, melhores[nome][0], melhores[nome][1]))

    print("Tempo rodando: " + str((time.time() - start_time)) + ' ms')

    return melhores
def qual_melhor(melhores):

    melhor = {'score': 0, 'nome': '', 'params': None}

    for nome, res in melhores.items():

        if res[0] > melhor['score']:

            melhor['score'] = res[0]

            melhor['nome'] = nome

            melhor['params'] = res[1]

    return melhor



def roda_melhor(melhor, train, target, test):

    models = {

        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),

    }

    modelo = models[melhor['nome']].set_params(**melhor['params'])

    modelo.fit(train, target)

    return modelo.predict_proba(test)

    
size_x = train.shape[0]

size_y = test.shape[0]



#melhores = testa_tudo(novo[:size_x], train['target'])
#melhor = qual_melhor(melhores)

#pred = roda_melhor(melhor, novo[:size_x], train['target'], novo[size_x:])
#test['target'] = pred

#test[['id', 'target']].to_csv('submission.csv', index=False)
from sklearn.metrics import roc_auc_score



# wheezy-turtle is a cat variable

cols = [c for c in df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



# Train by wheezy-copper-turtle-magic group

oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in range(df['wheezy-copper-turtle-magic'].max()+1):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index

    idx2 = test2.index

    train2.reset_index(drop=True, inplace=True)

    

    # Normalizing the data

    df2 = pd.concat([train2[cols], test2[cols]])

    df2 = VarianceThreshold(threshold=1.5).fit_transform(df2[cols])

    

    # Getting the vectors

    train3 = df2[:train2.shape[0]]

    test3 = df2[train2.shape[0]:]

    

    # CrossValidation using StratifiedKFold

    skf = StratifiedKFold(n_splits=10, random_state=666)

    for train_index, test_index in skf.split(train2, train2['target']):

        clf = QuadraticDiscriminantAnalysis(tol=1)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits





auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
#test['target'] = preds

#test[['id', 'target']].to_csv('submission.csv', index=False)
# Improving the model with pseudo labels



# INITIALIZE VARIABLES

test['target'] = preds

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABELED DATA

    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        

        # MODEL AND PREDICT WITH QDA

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

        oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3,:])[:,1]

        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

       

    #if k%64==0: print(k)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('Pseudo Labeled QDA scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)