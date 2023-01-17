# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.interpolate

import scipy.integrate

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Carregando os csvs de treino e teste respectivamente

df  = pd.read_csv('../input/train.csv', header=0)

dftest = pd.read_csv('../input/test.csv')
#Visualizando estatisticas descritivas do dataset de treino

df.describe()

#Visualizando estatisticas descritivas do dataset de teste

dftest.describe()
#Removendo dados duplicados do dataset de treino

print('Antes:', df.shape)

df.drop_duplicates()

print('Depois:', df.shape)
#Removendo dados duplicados do dataset de test

print('Antes:', dftest.shape)

dftest.drop_duplicates()

print('Depois:', dftest.shape)
#copiando os datasets carregados para as variaveis de treino e teste que serão utilizadas no experimento

train = df

test  = dftest


data = []

for f in train.columns:

    # definindo o uso (entre rótulo, id e atributos)

    if f == 'target':

        role = 'target' # rótulo

    elif f == 'id':

        role = 'id'

    else:

        role = 'input' # atributos

         

    # definindo o tipo do dado

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif train[f].dtype == float:

        level = 'interval'

    elif train[f].dtype == int:

        level = 'ordinal'

        

    # mantem keep como verdadeiro pra tudo, exceto id

    keep = True

    if f == 'id':

        keep = False

    

    # cria o tipo de dado

    dtype = train[f].dtype

    

    # cria dicionário de metadados

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(f_dict)

    

meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta.set_index('varname', inplace=True)
meta
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
atributos_missing = []



for f in train.columns:

    missings = train[train[f] == -1][f].count()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/df.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))

        

print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
atributos_missing = []



for f in test.columns:

    missings = test[test[f] == -1][f].count()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/dftest.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))

        

print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
# removendo ps_car_03_cat e ps_car_05_cat que tem muitos valores faltantes

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

train = train.drop(vars_to_drop, axis=1)

test = test.drop(vars_to_drop, axis=1)

meta.loc[(vars_to_drop),'keep'] = False  # atualiza os metadados para ter como referência (processar o test depois)
from sklearn.preprocessing import Imputer



media_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

moda_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = media_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_12'] = media_imp.fit_transform(train[['ps_car_12']]).ravel()

train['ps_car_14'] = media_imp.fit_transform(train[['ps_car_14']]).ravel()

train['ps_car_11'] = moda_imp.fit_transform(train[['ps_car_11']]).ravel()



test['ps_reg_03'] = media_imp.fit_transform(test[['ps_reg_03']]).ravel()

test['ps_car_12'] = media_imp.fit_transform(test[['ps_car_12']]).ravel()

test['ps_car_14'] = media_imp.fit_transform(test[['ps_car_14']]).ravel()

test['ps_car_11'] = moda_imp.fit_transform(test[['ps_car_11']]).ravel()
v = meta[(meta.level == 'nominal') & (meta.keep)].index



for f in v:

    dist_values = train[f].value_counts().shape[0]

    print('Atributo {} tem {} valores distintos'.format(f, dist_values))
v = meta[(meta.level == 'nominal') & (meta.keep)].index

print('Antes do one-hot encoding tinha-se {} atributos'.format(train.shape[1]))

train = pd.get_dummies(train, columns=v, drop_first=True)

print('Depois do one-hot encoding tem-se {} atributos'.format(train.shape[1]))



test = pd.get_dummies(test, columns=v, drop_first=True)

missing_cols = set( train.columns ) - set( test.columns )

for c in missing_cols:

    test[c] = 0

    

train, test = train.align(test, axis=1)
print(train.shape)

print(test.shape)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']



X_test  = test.drop(['id', 'target'], axis=1)

y_test  = test['target']
def gini(actual, pred, cmpcol = 0, sortcol = 1):  

       assert( len(actual) == len(pred) )  

       all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)  

       all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]  

       totalLosses = all[:,0].sum()  

       giniSum = all[:,0].cumsum().sum() / totalLosses  

  

       giniSum -= (len(actual) + 1) / 2.  

       return giniSum / len(actual)  

  

def gini_normalized(a, p):  

   return gini(a, p) / gini(a, a)  
model = LogisticRegression()

model.fit(X_train, y_train)

y_predl = model.predict(X_test)

acc_logistic = round(accuracy_score(y_predl, y_test) * 100, 2)

print(acc_logistic)





y_predlp = model.predict_proba(X_test)[:,1]

ginil = gini(y_predl, y_predlp)

giniln = gini_normalized(y_predl, y_predlp)

print(ginil)

print(giniln)
decisiontree = DecisionTreeClassifier()

decisiontree.fit(X_train, y_train)

y_predd = decisiontree.predict(X_test)

acc_decisiontree = round(accuracy_score(y_predd, y_test) * 100, 2)

print(acc_decisiontree)



y_preddp = decisiontree.predict_proba(X_test)[:,1]

ginid = gini(y_predd, y_preddp)

ginidn = gini_normalized(y_predd, y_preddp)

print(ginid)

print(ginidn)
gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

y_predg = gbk.predict(X_test)

acc_gbk = round(accuracy_score(y_predg, y_test) * 100, 2)

print(acc_gbk)



y_predgp = gbk.predict_proba(X_test)[:,1]

ginigp = gini(y_predg, y_predgp)

ginigpn = gini_normalized(y_predg, y_predgp)

print(ginigp)

print(ginigpn)
randomforest = RandomForestClassifier()

randomforest.fit(X_train, y_train)

y_predr = randomforest.predict(X_test)

acc_randomforest = round(accuracy_score(y_predr, y_test) * 100, 2)

print(acc_randomforest)



y_predrp = gbk.predict_proba(X_test)[:,1]

ginir = gini(y_predr, y_predrp)

ginirn = gini_normalized(y_predr, y_predrp)

print(ginir)

print(ginirn)
clf = MultinomialNB()

clf.fit(X_train, y_train)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

y_pred_nb = clf.predict(X_test)

acc_nb = round(accuracy_score(y_predl, y_test) * 100, 2)

print(acc_nb)





y_pred_nbp = clf.predict_proba(X_test)[:,1]

gininb = gini(y_pred_nb, y_pred_nbp)

ginidnb = gini_normalized(y_pred_nb, y_pred_nbp)

print(gininb)

print(ginidnb)
def lorenz(arr):

    # this divides the prefix sum by the total sum

    # this ensures all the values are between 0 and 1.0

    scaled_prefix_sum = arr.cumsum() / arr.sum()

    # this prepends the 0 value (because 0% of all people have 0% of all wealth)

    return np.insert(scaled_prefix_sum, 0, 0)

lorenz_curve = lorenz(np.sort(y_predrp, axis=None))

# we need the X values to be between 0.0 to 1.0

plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)

# plot the straight line perfect equality curve

plt.plot([0,1], [0,1])

plt.show()
# Create submission file

submission = pd.DataFrame()

submission['id'] = dftest['id']

submission['target'] = y_predrp

submission.to_csv('submit.csv', float_format='%.6f', index=False)