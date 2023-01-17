import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize

import numpy as np

import pandas as pd

import gc

import random

random.seed(2018)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize



import lightgbm as lgb

import xgboost as xgb

import os



import warnings

warnings.filterwarnings('ignore')

import random



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

X_test = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_test/ib_base_inicial_test.csv", parse_dates=["codmes"])

campanias = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_campanias/ib_base_campanias.csv", parse_dates=["codmes"])

digital = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_digital/ib_base_digital.csv", parse_dates=["codday"])

rcc = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_rcc/ib_base_rcc.csv", parse_dates=["codmes"])

reniec = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_reniec/ib_base_reniec.csv")

sunat = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_sunat/ib_base_sunat.csv")

vehicular = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_vehicular/ib_base_vehicular.csv")

train = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_train/ib_base_inicial_train.csv" , parse_dates=["codmes"])

#Chequeo la forma de los dataset complementarios y hago un pequeño apunte para saber en que consiste cada variable.

#Sentirse libre de mirar uno por uno con df.head()

print('Forma del csv campanias', campanias.shape)

print('Forma del csv digital', digital.shape)

print('Forma del csv rcc', rcc.shape)

print('Forma del csv reniec', reniec.shape)

print('Forma del csv sunat', sunat.shape)

print('Forma del csv vehicular', vehicular.shape)
rcc.producto.value_counts()     
# Chequeo nulos en rcc

total = rcc.isnull().sum().sort_values(ascending = False)

percent = (rcc.isnull().sum()/rcc.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(6)
train.shape, X_test.shape
list(train.columns.values)
train.head()
X_test.head()
# Chequeo nulos

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(20)
# Chequeo nulos

total = X_test.isnull().sum().sort_values(ascending = False)

percent = (X_test.isnull().sum()/X_test.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(20)
# Chequeo nulos en campañas

total = campanias.isnull().sum().sort_values(ascending = False)

percent = (campanias.isnull().sum()/campanias.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(20)
# Chequeo nulos en digital

total = digital.isnull().sum().sort_values(ascending = False)

percent = (digital.isnull().sum()/digital.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(6)
# Chequeo nulos en reniec

total = reniec.isnull().sum().sort_values(ascending = False)

percent = (reniec.isnull().sum()/reniec.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(3)
# Chequeo nulos en sunat

total = sunat.isnull().sum().sort_values(ascending = False)

percent = (sunat.isnull().sum()/sunat.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(6)
#paso a formato fecha la columna codmes

train['codmes'] =  pd.to_datetime(train['codmes'], format='%Y%m')
#saco el dia que esta por defecto en el 1ero de cada mes

train['codmes'] = pd.to_datetime(train['codmes'] ).dt.to_period('M')
train.head()
sns.countplot(train['codtarget'], palette='Set3')
print("Existen", round(100*train["codtarget"].value_counts()[1]/train.shape[0],2), "% de registros del target buscado")
train.codmes.value_counts()
X_test['codmes'] =  pd.to_datetime(X_test['codmes'], format='%Y%m')

X_test['codmes'] = pd.to_datetime(X_test['codmes']).dt.to_period('M')

X_test.codmes.value_counts()
campanias['codmes'] =  pd.to_datetime(campanias['codmes'], format='%Y%m')

campanias['codmes'] = pd.to_datetime(campanias['codmes']).dt.to_period('M')

campanias.codmes.value_counts()
digital['codday'] =  pd.to_datetime(digital['codday'], format='%Y%m%d')

digital['codday'] = pd.to_datetime(digital['codday']).dt.to_period('M')

digital.codday.value_counts()
rcc['codmes'] =  pd.to_datetime(rcc['codmes'], format='%Y%m')

rcc['codmes']= pd.to_datetime(rcc['codmes']).dt.to_period('M')

rcc.codmes.value_counts()