# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd                     

import matplotlib.pyplot as plt          

import numpy as np                      

from scipy.sparse import csr_matrix      

from scipy import stats

import seaborn as sns

import missingno as msno

import string

from pandas.api.types import CategoricalDtype
sample_submission = pd.read_csv("../input/categorical-feature-encoding-challenge-ii/sample_submission.csv")

df_test = pd.read_csv("../input/categorical-feature-encoding-challenge-ii/test.csv")

df_train = pd.read_csv("../input/categorical-feature-encoding-challenge-ii/train.csv")
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
summary_train = resumetable(df_train)

summary_train
summary_test = resumetable(df_test)

summary_test
bin_cols = ["bin_0" , "bin_1" , "bin_2"]
bin_cols
df_train[bin_cols] = df_train[bin_cols].replace(np.nan, 0)

df_test[bin_cols] = df_test[bin_cols].replace(np.nan, 0)
other_cols = ["bin_3" , "bin_4" , "nom_0" , "nom_1" , "nom_2" , "nom_3" , "nom_4" , "nom_5" , "nom_6" , "nom_7" , "nom_8" , "nom_9" , "ord_0" , "ord_1" , "ord_2" , "ord_3" , "ord_4" , "ord_5" , "day" , "month"

]
def replace_nan(data):

    for column in data.columns:

        if data[column].isna().sum() > 0:

            data[column] = data[column].fillna(data[column].mode()[0])





replace_nan(df_train)

replace_nan(df_test)
df_train.head()
bin_val = {'T':1, 'F':0, 'Y':1, 'N':0}



df_train['bin_3'] = df_train['bin_3'].map(bin_val)

df_train['bin_4'] = df_train['bin_4'].map(bin_val)

df_test['bin_3'] = df_test['bin_3'].map(bin_val)

df_test['bin_4'] = df_test['bin_4'].map(bin_val)
df_train.head()
df_test['target'] = 'test'

df = pd.concat([df_train, df_test], axis=0, sort=False )
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
df_train, df_test = df[df['target'] != 'test'], df[df['target'] == 'test'].drop('target', axis=1)

del df
df_train.head()
ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert','Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Lava Hot','Boiling Hot','Hot','Warm','Cold','Freezing'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',

                                     'W','X','Y','Z'], ordered=True)
# Transforming ordinal features



df_train.ord_1 = df_train.ord_1.astype(ord_1)

df_train.ord_2 = df_train.ord_2.astype(ord_2)

df_train.ord_3 = df_train.ord_3.astype(ord_3)

df_train.ord_4 = df_train.ord_4.astype(ord_4)
df_train.ord_3.head()
# Transforming ordinal features



df_test.ord_1 = df_test.ord_1.astype(ord_1)

df_test.ord_2 = df_test.ord_2.astype(ord_2)

df_test.ord_3 = df_test.ord_3.astype(ord_3)

df_test.ord_4 = df_test.ord_4.astype(ord_4)
df_test.ord_3.head()
# Geting the codes of the ordinal categoy



df_train.ord_1 = df_train.ord_1.cat.codes

df_train.ord_2 = df_train.ord_2.cat.codes

df_train.ord_3 = df_train.ord_3.cat.codes

df_train.ord_4 = df_train.ord_4.cat.codes



df_test.ord_1 = df_test.ord_1.cat.codes

df_test.ord_2 = df_test.ord_2.cat.codes

df_test.ord_3 = df_test.ord_3.cat.codes

df_test.ord_4 = df_test.ord_4.cat.codes
df_train['ord_5'] = df_train['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))
df_train.ord_5
df_test['ord_5'] = df_test['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))
df_test.ord_5
df_train.bin_3
df_train[['ord_0', 'ord_1', 'ord_2', 'ord_3']].head()
df_train_CE=df_train.copy()

columns=['day','month']

for col in columns:

    df_train_CE[col+'_sin']=np.sin((2*np.pi*df_train_CE[col])/max(df_train_CE[col]))

    df_train_CE[col+'_cos']=np.cos((2*np.pi*df_train_CE[col])/max(df_train_CE[col]))

df_train_CE=df_train_CE.drop(columns,axis=1)

df_train=df_train_CE

df_train.head()
df_test_CE=df_test.copy()

columns=['day','month']

for col in columns:

    df_test_CE[col+'_sin']=np.sin((2*np.pi*df_test_CE[col])/max(df_test_CE[col]))

    df_test_CE[col+'_cos']=np.cos((2*np.pi*df_test_CE[col])/max(df_test_CE[col]))

df_test_CE=df_test_CE.drop(columns,axis=1)

df_test=df_train_CE

df_test.head()
nom_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
resumetable(df_train[nom_cols])
for col in nom_cols:

    df_train[f'hash_{col}'] = df_train[col].apply( lambda x: hash(str(x)) % 5000)

    df_test[f'hash_{col}'] = df_test[col].apply( lambda x: hash(str(x)) % 5000)
new_feat = ['hash_nom_5', 'hash_nom_6', 'hash_nom_7', 'hash_nom_8',

            'hash_nom_9']



resumetable(df_train[nom_cols + new_feat])
df_train[['nom_5', 'hash_nom_5']].head()
df_test[['nom_5', 'hash_nom_5']].head()
# Dropping the original columns



df_train = df_train.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1)

df_test = df_test.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1)
resumetable(df_train)
resumetable(df_test)
df_train['target']= df_train.astype('int64')

df_test['target']= df_test.astype('int64')
from sklearn.model_selection import train_test_split
X = df_train.drop(["target"],axis=1)

y = df_train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
params = {#1

        'learning_rate': 0.05,

        'feature_fraction': 0.15,

        'min_data_in_leaf' : 80,

        'max_depth': 6,

        'objective': 'binary',

        'num_leaves':25,

        'metric': 'auc',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': True,

        'boost_from_average': False}
trn_data = lgb.Dataset(X_train,

                           label=y_train,

                           )

val_data = lgb.Dataset(X_test,

                           label=y_test,

                           )
num_round = 1000000

clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=250,

                    early_stopping_rounds = 3000)