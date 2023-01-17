# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas.tools.plotting import scatter_matrix
from sklearn.feature_extraction import DictVectorizer
sb = pd.read_csv('../input/sample_submission.csv')

print(sb.ix[0])

print(sb.ix[23])

print(sb.ix[2664])

print(sb.ix[235234])

sb.head()
nRows = 10000
data_train = pd.read_csv('../input/train_ver2.csv', nrows=nRows)

data_train.head()
data_train.shape
data_test = pd.read_csv('../input/test_ver2.csv', nrows=nRows)

data_test.head()
data_train['sexo'] = data_train['sexo'].fillna(0)

data_train['sexo'] = data_train['sexo'].replace('H', 1)

data_train['sexo'] = data_train['sexo'].replace('V', 2)
data_train['age'] = data_train['age'].replace(' NA', -100)

data_train['age'] = data_train['age'].astype(np.int)

data_train['age'] = data_train['age'].replace(-100, np.median(data_train['age']))
data_train['renta'].unique()
# Select categorial and number sign

categorical_columns = [c for c in data_train.columns if data_train[c].dtype.name == 'object']

numerical_columns   = [c for c in data_train.columns if data_train[c].dtype.name != 'object']

print(categorical_columns)

print(numerical_columns)
corr_mass = data_train[numerical_columns].corr()

corr_mass
columsCutted = corr_mass.fillna(0)

s = columsCutted.unstack()

so = s.sort_values()

print(so)
columsCutted.ix['age','indrel']
scatter_matrix(data_train[['ind_nomina_ult1','ind_nom_pens_ult1','ind_nom_pens_ult1','ind_cco_fin_ult1','ind_cno_fin_ult1']], alpha=0.2, figsize=(6, 6), diagonal='kde')