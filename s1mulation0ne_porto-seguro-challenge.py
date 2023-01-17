import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(10)
train.describe()
data = []

for f in train.columns:

    # Defining the role

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

         

    # Defining the level

    # A comparação com 'int' não estava funcionando, portanto foi utilizado np.int64

    # O código foi extraído do kaggle e está sendo executado localmente no jupyter

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif train[f].dtype == float:

        level = 'interval'

    elif train[f].dtype == np.int64:

        level = 'ordinal'

        

    # Initialize keep to True for all variables except for id

    keep = True

    if f == 'id':

        keep = False

    

    # Defining the data type 

    dtype = train[f].dtype

    

    # Creating a Dict that contains all the metadata for the variable

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
missing = []

for f in train:

    miss = (train[train[f]==-1][f]).count()

    if (miss>0):

        print('{}:\t{:.2%}\t{}'.format(f, miss/train[f].count(), miss))

        missing.append(f)
meta.at['ps_reg_03', 'keep'] = False

meta.at['ps_car_03_cat', 'keep'] = False

meta.at['ps_car_05_cat', 'keep'] = False
## Seleciona os dados contínuos e ordinais ara normalização

#interval = meta[(meta['level']=='interval')|(meta['level']=='ordinal')].index



#Seleciona apenas os dados contínuos para realizar normalização

interval = meta[(meta['level']=='interval') & (meta['keep']==True)].index



scaler = StandardScaler()

scaler.fit(train[interval])

data_s = scaler.transform(train[interval])



df_aux = pd.DataFrame(data_s, columns=interval)

train[interval] = df_aux
#not_cat = [f for f in train if 'cat' not in f]

train_replace = train.replace(-1, np.nan)

not_cat = [m for m in missing if 'cat' not in m]

train_keep_cat = train.replace({nc:-1 for nc in not_cat}, np.nan)

#train_keep_cat = train.replace(-1, np.nan)



for f in missing:

    if meta.loc[f]['level'] == 'nominal':

        train_replace[f].fillna(method='ffill', inplace=True)

        train_keep_cat[f].fillna(method='ffill', inplace=True)

    elif meta.loc[f]['level'] == 'interval':

        train_replace[f].fillna(method='ffill', inplace=True)

        train_keep_cat[f].fillna(method='ffill', inplace=True)

    elif meta.loc[f]['level'] == 'ordinal':

        train_replace[f].fillna(method='ffill', inplace=True)

        train_keep_cat[f].fillna(method='ffill', inplace=True)

    elif meta.loc[f]['level'] == 'binary':

        train_replace[f].fillna(method='ffill', inplace=True)

        train_keep_cat[f].fillna(method='ffill', inplace=True)

    
for f in meta[(meta['level']=='nominal') & (meta['keep']==True)].index:

    print('{} possui {} categorias'.format(f, len(train_replace[f].unique())))
for f in meta[(meta['level']=='nominal') & (meta['keep']==True)].index:

    print('{} possui {} categorias'.format(f, len(train_keep_cat[f].unique())))
meta.at['ps_car_11_cat', 'keep'] = False
test_keep_cat = test.copy()

v = meta[(meta.level == 'nominal') & (meta.keep)].index

print('Antes do one-hot encoding tinha-se {} atributos'.format(train_keep_cat.shape[1]))

train_keep_cat = pd.get_dummies(train_keep_cat, columns=v, drop_first=True)

print('Depois do one-hot encoding tem-se {} atributos'.format(train_keep_cat.shape[1]))



test_keep_cat = pd.get_dummies(test_keep_cat, columns=v, drop_first=True)

missing_cols = set( train_keep_cat.columns ) - set( test_keep_cat.columns )

for c in missing_cols:

    test_keep_cat[c] = 0

    

train_keep_cat, test_keep_cat = train_keep_cat.align(test_keep_cat, axis=1)
test_replace = test.copy()

v = meta[(meta.level == 'nominal') & (meta.keep)].index

print('Antes do one-hot encoding tinha-se {} atributos'.format(train_replace.shape[1]))

train_replace = pd.get_dummies(train_replace, columns=v, drop_first=True)

print('Depois do one-hot encoding tem-se {} atributos'.format(train_replace.shape[1]))



test_replace = pd.get_dummies(test_replace, columns=v, drop_first=True)

missing_cols = set( train_replace.columns ) - set( test_replace.columns )

for c in missing_cols:

    test_replace[c] = 0

    

train_replace, test_replace = train_replace.align(test_replace, axis=1)
for c in test_replace.columns:

    print(c)
meta.at['target', 'keep'] = False
x=train_keep_cat.drop(meta[meta['keep']==False].index, axis=1)

y=train['target']

X_test=pd.DataFrame(test_keep_cat.drop(meta[meta['keep']==False].index, axis=1))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
clf=LogisticRegression()

clf.fit(x_train,y_train)

Y_pred=clf.predict(X_test)

acc = round(clf.score(x_test, y_test) * 100, 2)

acc
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn.fit(x_train, y_train)
#y_pred = knn.predict(X_test)



# comparando com gabarito

#accuracy_score(y_test, y_pred)