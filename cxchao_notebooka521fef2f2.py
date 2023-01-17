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
chunksize=10000;

train = pd.read_csv('../input/train_ver2.csv',nrows = chunksize,dtype = 'object')

test = pd.read_csv('../input/test_ver2.csv',dtype = 'object')

sample = pd.read_csv('../input/sample_submission.csv',dtype = 'object')
train.shape,test.shape,

train = pd.read_csv('../input/train_ver2.csv',nrows = 1000000,dtype = 'object')
train.tail()

test.head()

test.conyuemp[test.conyuemp.notnull()]

(set(train.segmento.fillna(2)))

test.describe()

#train.conyuemp.dtype
train_pre=train.loc[train.index.difference(train[train[test.columns].T.isnull().sum()>10].index)]

train_fea = train_pre[test.columns]

nan = pd.DataFrame(train_fea.isnull().sum())

nan['test'] = test.isnull().sum()

nan.columns=['train','test']

nan['desc']=list(['Date','Id_customer','Employee_index','Residence','Sex','Age','Date_hold','New_customer',

            'Seniority','Primary_customer','Lastdata_primary','Type_beginmouth','Relationtype_bm(active)',

            'Residence_withbank','Foreigner_birthwithbank','isSpouse','Channel','Deceased','Address_primary',

            'Address_customer/province','Province_name','Activity','Income_household','Type_customer'])

nan
#train_pre[train_pre['cod_prov'].fillna(0).astype('float')==8][test.columns]

#(train_pre['cod_prov']).value_counts()

train_pre['cod_prov']=train_pre['cod_prov'].fillna('8')

#(train_pre['nomprov']).value_counts()

train_pre['nomprov']=train_pre['nomprov'].fillna('BARCELONA')

#(train_pre.renta.astype('float')).dropna().describe()

train_pre['renta']=train_pre['renta'].fillna('87839')

#train[train['indrel']=='99'].iloc[:,2:24]  idrel==1,ult_fec_cli_1t 缺失

#set(train_pre['tipodom'].astype('int'))

train_pre['tipodom']=train_pre['tipodom'].fillna('1')

#train_pre['sexo'].astype('str').value_counts().plot(kind='bar')

train_pre['sexo']=train_pre['sexo'].fillna('N')

train_pre['indrel_1mes']=train_pre['indrel_1mes'].fillna('N')

train_pre['tiprel_1mes']=train_pre['tiprel_1mes'].fillna('N')
train_pre[train.columns.difference(test.columns)].fillna(0).astype('int').sum()

print((set(train.ncodpers.tolist()+test.ncodpers.tolist())))

print((set(test.ncodpers.values)))

print((set(train.fecha_dato.values)))
train.fecha_dato.value_counts()
test.ncodpers
train_pre[train_pre['nomprov'].isnull()][test.columns]
train_fea.sexo.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=0.6)
#import matplotlib.pyplot as plt

pd.DataFrame(train_fea.age.value_counts()).sort_index().plot()
plt.rc_params()
range(0,100)