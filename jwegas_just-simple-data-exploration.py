# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_ver2.csv', nrows=1000)

train.head()
# calculate shape of train data

nrow_train = 0

ncol_train = 0

train_chunk = pd.read_csv('../input/train_ver2.csv', chunksize=1000000, low_memory=False)

for chunk in train_chunk:

    nrow_train += chunk.shape[0]

    ncol_train = chunk.shape[1]

    

print('Train set has %d rows and %d columns'%(nrow_train, ncol_train))
def getColumnInfo(colName, dataSetName):

    tmp = pd.read_csv('../input/'+dataSetName, usecols=(colName,))

    NaNs = 100.0 * tmp[pd.isnull(tmp[colName])].shape[0] / tmp.shape[0]

    example = tmp[~pd.isnull(tmp[colName])][colName].values[0]

    print('ColName: %s;\t type: %s;\t NaNs rate: %0.4f perc.;\t example: %s'%(colName, tmp[colName].values.dtype, NaNs, example))
columns = pd.read_csv('../input/train_ver2.csv', nrows=1).columns

for col in columns:

    getColumnInfo(col, 'train_ver2.csv')
columns = pd.read_csv('../input/train_ver2.csv', nrows=1).columns

columns
fecha_dato = pd.read_csv('../input/train_ver2.csv', usecols=('fecha_dato',))
fecha_dato['fecha_dato'] = fecha_dato['fecha_dato'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')\

                                                          if pd.isnull(x) is False\

                                                          else x)
fecha_dato['year'] = fecha_dato['fecha_dato'].apply(lambda x: x.year\

                                                          if pd.isnull(x) is False\

                                                          else x)

fecha_dato['month'] = fecha_dato['fecha_dato'].apply(lambda x: x.month\

                                                          if pd.isnull(x) is False\

                                                          else x)

fecha_dato['day'] = fecha_dato['fecha_dato'].apply(lambda x: x.day\

                                                          if pd.isnull(x) is False\

                                                          else x)
fecha_dato.year.value_counts()
fecha_dato.groupby('year').month.value_counts()
fecha_dato.day.value_counts()
# clean memory

del fecha_dato

gc.collect()
ncodpers = pd.read_csv('../input/train_ver2.csv', usecols=('ncodpers',), dtype=dtypes)
ncodpers.head()
ncodpers['ncodpers'].value_counts().value_counts()
ncodpers.max()
ncodpers.min()
ncodpers.info()
# uint32
# clean memory

del ncodpers

gc.collect()
ind_empleado = pd.read_csv('../input/train_ver2.csv', usecols=('ncodpers','ind_empleado',), dtype=dtypes)
ind_empleado.head()
ind_empleado['ind_empleado'].value_counts()
ind_empleado.drop_duplicates()['ind_empleado'].value_counts()
del ind_empleado

gc.collect()
pais_residencia = pd.read_csv('../input/train_ver2.csv', usecols=('ncodpers','pais_residencia',), dtype=dtypes)
pais_residencia.head()
pais_residencia.drop_duplicates()['pais_residencia'].value_counts()
pais_residencia['IsES'] = pais_residencia['pais_residencia'].apply(lambda x: 1 if x == 'ES' and pd.isnull(x) == False\

                                                                   else 0)
pais_residencia.drop_duplicates()['IsES'].value_counts()
pais_residencia.drop_duplicates()['IsES'].value_counts() / pais_residencia.drop_duplicates().shape[0] * 100.0
del pais_residencia

gc.collect()
sexo = pd.read_csv('../input/train_ver2.csv', usecols=('ncodpers','sexo',), dtype=dtypes)
sexo.head()
sexo['sexo'].value_counts() / sexo.shape[0] * 100.
sexo.drop_duplicates()['sexo'].value_counts() / sexo.drop_duplicates().shape[0] * 100.
sexo.info()
del sexo

gc.collect()
age = pd.read_csv('../input/train_ver2.csv', usecols=('ncodpers','sexo', 'age'), dtype=dtypes)
age.head()
age[pd.isnull(age['age'])]
age['age'].mean()
age.dtypes
age['age'].value_counts()
dtypes = {

   'ncodpers': np.uint32, 

}