from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as graphic

from math import sqrt, factorial, log, ceil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target_name = 'Target'
print ("Tamanho train: ", train.shape)
print ("Tamanho test: ", test.shape)
train.describe()
from collections import OrderedDict

plt.figure(figsize = (15, 12))
plt.style.use('fivethirtyeight')

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'yellow', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

for i, col in enumerate(train.select_dtypes('float')):
    ax = plt.subplot(4, 2, i + 1)
    for poverty_level, color in colors.items():
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)
train.select_dtypes('object').head()
mapping = {"yes": 1, "no": 0}

for df in [train, test]:
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

train[['dependency', 'edjefa', 'edjefe']].describe()
plt.figure(figsize = (8, 8))

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):
    ax = plt.subplot(3, 1, i + 1)
 
    for poverty_level, color in colors.items():
        
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 
                    ax = ax, color = color, label = poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top = 2)
test['Target'] = np.nan
data = train.append(test, ignore_index = True)
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

not_equal = all_equal[all_equal != True]
households_leader = train.groupby('idhogar')['parentesco1'].sum()
households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
for household in not_equal.index:
    true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])
    train.loc[train['idhogar'] == household, 'Target'] = true_target
    
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print('\n Missing Values')
missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})
missing['percent'] = missing['total'] / len(data)
missing.sort_values('percent', ascending = False).head(10).drop('Target')
def plot_value_counts(df, col, heads_only = False):
    if heads_only:
        df = df.loc[df['parentesco1'] == 1].copy()
        
    plt.figure(figsize = (8, 6))
    df[col].value_counts().sort_index().plot.bar(color = 'blue',
                                                 edgecolor = 'k',
                                                 linewidth = 2)
    plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
    plt.show();
data['v18q1'] = data['v18q1'].fillna(0)
own_variables = [x for x in data if x.startswith('tipo')]
data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (8, 6),
                                                                        color = 'blue',
                                                              edgecolor = 'k', linewidth = 2);
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 60)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0
data['v2a1-missing'] = data['v2a1'].isnull()
data['v2a1-missing'].value_counts()
data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0
data['rez_esc-missing'] = data['rez_esc'].isnull()
data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc-missing']
ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']
hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']
hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
data = data.drop(columns = sqr_)
heads = data.loc[data['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])
elec = []
for i, row in heads.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)
        
heads['elec'] = elec
heads['elec-missing'] = heads['elec'].isnull()
heads = heads.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])
water = []
for i, row in heads.iterrows():
    if row['abastaguano'] == 1:
        water.append(0)
    elif row['abastaguafuera'] == 1:
        water.append(1)
    elif row['abastaguadentro'] == 1:
        water.append(2)
    else:
        water.append(np.nan)
        
heads['water'] = water
heads['water-missing'] = heads['water'].isnull()
heads = heads.drop(columns = ['abastaguano', 'abastaguafuera', 'abastaguadentro'])
sanit = []
for i, row in heads.iterrows():
    if row['sanitario1'] == 1:
        sanit.append(0)
    elif row['v14a'] == 0:
        sanit.append(0)
    elif row['sanitario2'] == 1:
        sanit.append(3)
    elif row['sanitario3'] == 1:
        sanit.append(2)
    elif row['sanitario5'] == 1:
        sanit.append(1)
    elif row['sanitario6'] == 1:
        sanit.append(2)
    else:
        sanit.append(np.nan)
        
heads['sanit'] = sanit
heads['sanit-missing'] = heads['sanit'].isnull()
heads = heads.drop(columns = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'v14a'])
cozinhar = []
for i, row in heads.iterrows():
    if row['energcocinar1'] == 1:
        cozinhar.append(0)
    elif row['energcocinar2'] == 1:
        cozinhar.append(2)
    elif row['energcocinar3'] == 1:
        cozinhar.append(2)
    elif row['energcocinar4'] == 1:
        cozinhar.append(1)
    else:
        cozinhar.append(np.nan)
        
heads['cozinhar'] = cozinhar
heads['cozinhar-missing'] = heads['cozinhar'].isnull()
heads = heads.drop(columns = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4'])
heads = heads.drop(columns = ['area2', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6'])
heads['walls'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]), axis = 1)
heads = heads.drop(columns = ['epared1', 'epared2', 'epared3'])

heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]), axis = 1)
heads = heads.drop(columns = ['etecho1', 'etecho2', 'etecho3'])

heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]), axis = 1)
heads = heads.drop(columns = ['eviv1', 'eviv2', 'eviv3'])

heads['walls+roof+floor'] = heads['walls'] + heads['roof'] + heads['floor']
counts = pd.DataFrame(heads.groupby(['walls+roof+floor'])['Target'].value_counts(normalize = True)).rename(columns = {'Target': 'Normalized Count'}).reset_index()
counts.head()
heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']
heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
heads['electronics-per-capita'] = (heads['v18q1'] + heads['qmobilephone']) / heads['tamviv']
ind = data[id_ + ind_bool + ind_ordered]
ind[[c for c in ind if c.startswith('instl')]].head()
ind['inst'] = np.argmax(np.array(ind[[c for c in ind if c.startswith('instl')]]), axis = 1)
ind = ind.drop(columns = [c for c in ind if c.startswith('instlevel')])
ind['inst/age'] = ind['inst'] / ind['age']
ind['escolari/age'] = ind['escolari'] / ind['age']
ind['tech'] = ind['v18q'] + ind['mobilephone']
ind['tech'].describe()
def combine_features(data, cols=[], name=''):
    df = data.copy()
    for i, col in enumerate(cols):
        print(i + 1, col)
    df[cols] = df[cols].multiply([i for i in range(1, len(cols) + 1)], axis=1)
    df[name] = df[cols].sum(axis=1)
    df.drop(cols, axis=1, inplace=True)
    return df
heads = combine_features(heads, cols=[col for col in heads.columns if col.startswith('lugar')], name='region')
print('Region count by target.');
sns.factorplot("region", col="Target", col_wrap=4, data=heads, kind="count");
heads = combine_features(heads, cols=[col for col in heads.columns if col.startswith('tipovivi')], name='home_own')
print('Home ownership type count by target.');
sns.factorplot("home_own", col="Target", col_wrap=4, data=heads, kind="count");
heads = heads.drop(columns = ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras',
                             'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisomadera', 'techozinc',
                             'techoentrepiso', 'techocane', 'techootro', 'hogar_nin', 'hogar_mayor', 'hogar_adul',
                             'dependency', 'edjefe', 'edjefa', 'meaneduc', 'r4h1', 'r4h2', 'r4h3', 'r4m1',
                             'r4m2', 'r4m3', 'r4t1', 'r4t2', 'walls', 'roof', 'floor', 'rooms', 'v2a1', 'qmobilephone',
                             'v18q1', 'hacapo', 'cielorazo', 'bedrooms', 'pisonotiene', 'hacdor'])
ind = ind.drop(columns = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6',
                              'estadocivil7', 'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
                              'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 'parentesco11',
                              'parentesco12', 'mobilephone', 'escolari', 'inst', 'age', 'v18q', 'male'])
final = heads.merge(ind, on = id_ , how = 'left')
print('Final features shape: ', final.shape)
print('\n Missing Values')
missing = pd.DataFrame(final.isnull().sum()).rename(columns = {0: 'total'})
missing['percent'] = missing['total'] / len(final)
missing.sort_values('percent', ascending = False).head(10).drop('Target')
final = final[final['elec-missing'] != True]
final = final[final['rez_esc-missing'] != True]
final = final[final['cozinhar-missing'] != True]
final = final[final['water-missing'] != True]
final = final[final['sanit-missing'] != True]
final = final[final['v2a1-missing'] != True]
print('Final features shape: ', final.shape)
final = final.drop(columns = ['sanit-missing', 'cozinhar-missing', 'water-missing', 'elec-missing', 'v2a1-missing',
                              'rez_esc-missing'])
print('Final features shape: ', final.shape)
train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))

SPLIT = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar'])
FINALtest = final[final['Target'].isnull()]
submission_base = FINALtest[['Id', 'idhogar']].copy()
FINALtest = FINALtest.drop(columns = ['Id', 'idhogar', 'Target' ])
train, test = train_test_split(SPLIT, test_size=0.2)
neighbors = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]
Xtrain = train
Ytrain = train.Target
Xtest = test
Ytest = test.Target
print('With CV = 3:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 5:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 10:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 15:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=15, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 20:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(Xtrain,Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
YtestPred = knn.predict(Xtest)
from sklearn.metrics import accuracy_score
accuracy_score(Ytest,YtestPred)
FINALtest.insert(0,'Target',0.0)
FINALtest.dtypes
XFINALtest = FINALtest
YFINALtestPred = knn.predict(XFINALtest)
YFINALtestPred
pred = pd.DataFrame(submission_base)
pred["Target"] = YFINALtestPred
pred.to_csv("prediction.csv", index=False)
