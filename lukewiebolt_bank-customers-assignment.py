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
df = pd.read_sas('/kaggle/input/bank-customers/bank_customers.sas7bdat', encoding = 'iso-8859-1')

df.head()
df.shape
df.columns
df.columns[df.isnull().any()]
df['HMVAL'].fillna(0, inplace = True)

df['INV'].fillna(0, inplace = True)

df['INVBAL'].fillna(0, inplace = True)

df['CC'].fillna(0, inplace = True)

df['CCBAL'].fillna(0, inplace = True)

df['POSAMT'].fillna(0, inplace = True)

df['POSAMT'].fillna(0, inplace = True)

df['CCPURC'].fillna(0, inplace = True)

df['HMOWN'].fillna(0, inplace = True)

df['PHONE'].fillna(0, inplace = True)

df['ACCTAGE'].fillna(df['ACCTAGE'].mean(), inplace = True)

df['POS'].fillna(df['POS'].mean(), inplace = True)

df['INCOME'].fillna(df['INCOME'].mean(), inplace = True)

df['LORES'].fillna(df['LORES'].mean(), inplace = True)

df['CRSCORE'].fillna(df['CRSCORE'].mean(), inplace = True)

df['AGE'].fillna(df['AGE'].mean(), inplace = True)
df['BRANCH'].unique()
df['BRANCH'].replace(['B12','B7', 'B5', 'B1', 'B9', 'B2', 'B3', 'B8','B15','B4','B18', 'B14', 'B6', 'B16', 'B19', 'B17', 'B13', 'B11', 'B10'],[12,7,5,1,9,2,3,8,15,4,18,14,6,16,19,17,13,11,10],inplace=True)
res_dummies = pd.get_dummies(df['RES'],prefix = 'RES', prefix_sep = '_', drop_first = True)

df = df.drop('RES',axis = 1)

df = df.join(res_dummies)

df 
df['LOC'].unique()
y = df['LOC']
x = df.drop('LOC', axis = 1)
from sklearn import tree

import graphviz
clf = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=5, max_depth=5, max_features=None)

clf_expanded = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

clf_expanded = clf_expanded.fit(x, y)
fn = ['ACCTAGE', 'DDA', 'DDABAL', 'DEP', 'DEPAMT', 'CASHBK', 'CHECKS',

       'DIRDEP', 'NSF', 'NSFAMT', 'PHONE', 'TELLER', 'SAV', 'SAVBAL', 'ATM',

       'ATMAMT', 'POS', 'POSAMT', 'CD', 'CDBAL', 'IRA', 'IRABAL', 'LOCBAL', 

       'INV', 'INVBAL', 'ILS', 'ILSBAL', 'MM', 'MMBAL', 'MMCRED',

       'MTG', 'MTGBAL', 'CC', 'CCBAL', 'CCPURC', 'SDB', 'INCOME', 'HMOWN',

       'LORES', 'HMVAL', 'AGE', 'CRSCORE', 'MOVED', 'INAREA', 'BRANCH',

       'RES_S', 'RES_U']

cn = ['0', '1']



dot_data = tree.export_graphviz(clf, out_file=None,

                    feature_names= fn,

                    class_names= cn,

                    filled=True, rounded=True,  

                    special_characters=True)  

graph = graphviz.Source(dot_data)  

graph.render("Tree 1 Prunned") 
fn = ['ACCTAGE', 'DDA', 'DDABAL', 'DEP', 'DEPAMT', 'CASHBK', 'CHECKS',

       'DIRDEP', 'NSF', 'NSFAMT', 'PHONE', 'TELLER', 'SAV', 'SAVBAL', 'ATM',

       'ATMAMT', 'POS', 'POSAMT', 'CD', 'CDBAL', 'IRA', 'IRABAL', 'LOCBAL', 

       'INV', 'INVBAL', 'ILS', 'ILSBAL', 'MM', 'MMBAL', 'MMCRED',

       'MTG', 'MTGBAL', 'CC', 'CCBAL', 'CCPURC', 'SDB', 'INCOME', 'HMOWN',

       'LORES', 'HMVAL', 'AGE', 'CRSCORE', 'MOVED', 'INAREA', 'BRANCH',

       'RES_S', 'RES_U']

cn = ['0', '1']



dot_data = tree.export_graphviz(clf_expanded, out_file=None,

                    feature_names= fn,

                    class_names= cn,

                    filled=True, rounded=True,  

                    special_characters=True)  

graph = graphviz.Source(dot_data)  

graph.render("Tree_Expanded") 