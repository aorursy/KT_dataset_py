import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
pd.set_option('display.max_columns', 143)
HPtrain=pd.read_csv('../input/train.csv')

HPtest=pd.read_csv('../input/test.csv')

HPsample=pd.read_csv('../input/sample_submission.csv')

HPtrain.head()
HPtrain.shape
HPtrain.describe().append(HPtrain.isnull().sum().rename('isnull'))
HPtest.head()
HPtest.shape
HPtest.describe().append(HPtest.isnull().sum().rename('isnull'))
HPsample.head()
from collections import Counter
a=Counter(HPtrain['r4t3'])
b=Counter(HPtrain['tamhog'])
a-b
from collections import Counter
a0=Counter(HPtrain['tamviv'])
b0=Counter(HPtrain['tamhog'])
a0-b0
from collections import Counter
a1=Counter(HPtrain['r4t3'])
b1=Counter(HPtrain['tamviv'])
a1-b1
from collections import Counter
a2=Counter(HPtrain['r4t3'])
b2=Counter(HPtrain['hhsize'])
a2-b2
from collections import Counter
a3=Counter(HPtrain['tamviv'])
b3=Counter(HPtrain['hhsize'])
a3-b3
from collections import Counter
a4=Counter(HPtrain['tamhog'])
b4=Counter(HPtrain['hhsize'])
a4-b4
from collections import Counter
a3=Counter(HPtrain['hhsize'])
b3=Counter(HPtrain['hogar_total'])
print(a3-b3)
from collections import Counter
a5=Counter(HPtrain['tamhog'])
b5=Counter(HPtrain['hogar_total'])
print(a5-b5)
from collections import Counter
a6=Counter(HPtrain['SQBage'])
b6=Counter(HPtrain['agesq'])
print(a6-b6)

from collections import Counter
at6=Counter(HPtest['SQBage'])
bt6=Counter(HPtest['agesq'])
print(at6-bt6)
HPtrain.drop(['hhsize', 'hogar_total' and 'agesq'],axis=1,inplace=True)

HPtest.drop(['hhsize', 'hogar_total' and 'agesq'],axis=1,inplace=True)
HPtrain['r4t3'].unique()
HPtrain['tamhog'].unique()
HPtrain['tamviv'].unique()
HPtrain.loc[HPtrain['tamviv']==15]
del HPtrain['tamhog']

del HPtest['tamhog']
del HPtrain['tamviv']

del HPtest['tamviv']
HPtrain.shape
HPtrain.head()
HPtrain.describe().append(HPtrain.isnull().sum().rename('isnull'))
HPtest.describe().append(HPtest.isnull().sum().rename('isnull'))
HPtrain.loc[((HPtrain['v2a1'].isnull())|(HPtrain['tipovivi3']==0)),['v2a1','tipovivi3','tipovivi2']]
HPtrain.loc[((HPtrain['v2a1'].notnull()) & (HPtrain['tipovivi3']==0)),['v2a1','tipovivi3','tipovivi2']]
HPtrain['v2a1']= HPtrain['v2a1'].fillna(value=0)

HPtest['v2a1']= HPtest['v2a1'].fillna(value=0)
HPtrain.loc[(HPtrain['v18q1'].isnull()) & (HPtrain['v18q']==0),['Id', 'v18q1', 'v18q', 'idhogar', 'age','Target']]
HPtrain.loc[HPtrain['v18q1']!=HPtrain['v18q'],['Id', 'v18q1', 'v18q',  'idhogar','age']]
HPtrain['v18q1']= HPtrain['v18q1'].fillna(value=0.0)

HPtest['v18q1']= HPtest['v18q1'].fillna(value=0.0)
HPtrain.loc[((HPtrain['rez_esc'].isnull()) & (HPtrain['age']>=18) | (HPtrain['age']<=6)),['rez_esc','age']]
HPtrain.loc[((HPtrain['rez_esc'].isnull()) & (HPtrain['age']>6) & (HPtrain['age']<18)),['Id','rez_esc','age','dis']]
HPtrain['rez_esc'].unique()
HPtrain['rez_esc']= HPtrain['rez_esc'].fillna(value=0.0)

HPtest['rez_esc']= HPtest['rez_esc'].fillna(value=0.0)
HPtrain.loc[HPtrain['meaneduc'].isnull(), ['Id', 'meaneduc', 'age', 'rez_esc','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3']]
HPtrain['meaneduc']= HPtrain['meaneduc'].fillna(value=0.0)

HPtest['meaneduc']= HPtest['meaneduc'].fillna(value=0.0)
HPtrain['SQBmeaned']= HPtrain['SQBmeaned'].fillna(value=0.0)

HPtest['SQBmeaned']= HPtest['SQBmeaned'].fillna(value=0.0)
print(HPtrain.shape)

print(HPtest.shape)
print(HPtrain.isnull().sum().sum())

print(HPtest.isnull().sum().sum())
HPtrainC=HPtrain.copy()
HPtestC=HPtest.copy()
HPtrainC.dtypes
HPtrainC.head()
print(HPtrainC.shape)

print(HPtestC.shape)

cols=list(HPtrainC.columns)
print(cols)
HPtrainC=HPtrainC[['idhogar','Id', 'male', 'female', 'v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6','parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12','escolari', 'rez_esc', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis',  'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'Target']].copy()

HPtestC=HPtestC[['idhogar','Id', 'male', 'female', 'v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6','parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12','escolari', 'rez_esc', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis',  'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']].copy()
print(HPtrainC.shape)

print(HPtestC.shape)
HPtrainC.head(10)
HPtestC.head(10)
print(list(HPtrainC.columns))
HPtrainN=HPtrainC[['idhogar','v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'Target']].copy()

HPtestN=HPtestC[['idhogar','v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']].copy()

test_predict= HPtestC[['idhogar', 'Id']].copy()
print(HPtrainN.shape)

print(HPtestN.shape)

print(test_predict.shape)
HPtrainN.head()
# This give us an estimate of the number of households in the train/test dataset
print(HPtrainN['idhogar'].nunique())

print(HPtestN['idhogar'].nunique())
HPtrainN.drop_duplicates(subset='idhogar', keep='first', inplace=True)

HPtestN.drop_duplicates(subset='idhogar', keep='first', inplace=True)
print(HPtrainN.shape)

print(HPtestN.shape)


HPtrainN['area2'].unique()
HPtrainN['area1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['area2'].value_counts().plot(kind='bar')
sns.despine
del HPtrainN['area2']

del HPtestN['area2']

HPtrainN['lugar1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['lugar2'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['lugar3'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['lugar4'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['lugar5'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['lugar6'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['lugar2'] = HPtrainN['lugar2'].map({0: 0, 1: 2})

HPtestN['lugar2'] = HPtestN['lugar2'].map({0: 0, 1: 2})
HPtrainN['lugar3'] = HPtrainN['lugar3'].map({0: 0, 1: 3})

HPtestN['lugar3'] = HPtestN['lugar3'].map({0: 0, 1: 3})
HPtrainN['lugar4'] = HPtrainN['lugar4'].map({0: 0, 1: 4})

HPtestN['lugar4'] = HPtestN['lugar4'].map({0: 0, 1: 4})
HPtrainN['lugar5'] = HPtrainN['lugar5'].map({0: 0, 1: 5})

HPtestN['lugar5'] = HPtestN['lugar5'].map({0: 0, 1: 5})
HPtrainN['lugar6'] = HPtrainN['lugar6'].map({0: 0, 1: 6})

HPtestN['lugar6'] = HPtestN['lugar6'].map({0: 0, 1: 6})
HPtrainN['lugar1'] = HPtrainN['lugar1']+ HPtrainN['lugar2']+ HPtrainN['lugar3']+ HPtrainN['lugar4']+ HPtrainN['lugar5']+ HPtrainN['lugar6']

HPtestN['lugar1'] = HPtestN['lugar1']+ HPtestN['lugar2']+ HPtestN['lugar3']+ HPtestN['lugar4']+ HPtestN['lugar5']+ HPtestN['lugar6']
HPtrainN['lugar1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN.drop(['lugar2','lugar3','lugar4','lugar5','lugar6'], inplace=True, axis=1)

HPtestN.drop(['lugar2','lugar3','lugar4','lugar5','lugar6'], inplace=True, axis=1)
HPtrainN.shape

HPtrainN['tipovivi1'].unique()
HPtrainN['tipovivi2'].unique()
HPtrainN['tipovivi2'].unique()
HPtrainN['tipovivi4'].unique()
HPtrainN['tipovivi5'].unique()
HPtrainN['tipovivi2'] = HPtrainN['tipovivi2'].map({0: 0, 1: 2})

HPtestN['tipovivi2'] = HPtestN['tipovivi2'].map({0: 0, 1: 2})
HPtrainN['tipovivi3'] = HPtrainN['tipovivi3'].map({0: 0, 1: 3})

HPtestN['tipovivi3'] = HPtestN['tipovivi3'].map({0: 0, 1: 3})
HPtrainN['tipovivi4'] = HPtrainN['tipovivi4'].map({0: 0, 1: 4})

HPtestN['tipovivi4'] = HPtestN['tipovivi4'].map({0: 0, 1: 4})
HPtrainN['tipovivi5'] = HPtrainN['tipovivi5'].map({0: 0, 1: 5})

HPtestN['tipovivi5'] = HPtestN['tipovivi5'].map({0: 0, 1: 5})
HPtrainN['tipovivi1'] = HPtrainN['tipovivi1'] + HPtrainN['tipovivi2'] + HPtrainN['tipovivi3'] + HPtrainN['tipovivi4'] + HPtrainN['tipovivi5']

HPtestN['tipovivi1'] = HPtestN['tipovivi1'] + HPtestN['tipovivi2'] + HPtestN['tipovivi3'] + HPtestN['tipovivi4'] + HPtestN['tipovivi5']
HPtrainN.drop(['tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',], inplace=True, axis=1)

HPtestN.drop(['tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',], inplace=True, axis=1)
HPtrainN['tipovivi1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN.shape

HPtrainN['eviv2'].unique()
HPtrainN['eviv3'].unique()
HPtrainN['eviv2'] = HPtrainN['eviv2'].map({0: 0, 1: 2})

HPtestN['eviv2'] = HPtestN['eviv2'].map({0: 0, 1: 2})
HPtrainN['eviv3'] = HPtrainN['eviv3'].map({0: 0, 1: 3})

HPtestN['eviv3'] = HPtestN['eviv3'].map({0: 0, 1: 3})
HPtrainN['eviv1'] = HPtrainN['eviv1'] + HPtrainN['eviv2'] + HPtrainN['eviv3'] 

HPtestN['eviv1'] = HPtestN['eviv1'] + HPtestN['eviv2'] + HPtestN['eviv3'] 
HPtrainN.drop(['eviv2', 'eviv3'], inplace=True, axis=1)

HPtestN.drop(['eviv2', 'eviv3'], inplace=True, axis=1)
HPtrainN['eviv1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN.shape

HPtrainN['etecho1'].unique()
HPtrainN['etecho2'].unique()
HPtrainN['etecho3'].unique()
HPtrainN['etecho2'] = HPtrainN['etecho2'].map({0: 0, 1: 2})

HPtestN['etecho2'] = HPtestN['etecho2'].map({0: 0, 1: 2})
HPtrainN['etecho3'] = HPtrainN['etecho3'].map({0: 0, 1: 3})

HPtestN['etecho3'] = HPtestN['etecho3'].map({0: 0, 1: 3})
HPtrainN['etecho1'] = HPtrainN['etecho1'] + HPtrainN['etecho2'] + HPtrainN['etecho3'] 

HPtestN['etecho1'] = HPtestN['etecho1'] + HPtestN['etecho2'] + HPtestN['etecho3'] 
HPtrainN.drop(['etecho2', 'etecho3'], inplace=True, axis=1)

HPtestN.drop(['etecho2', 'etecho3'], inplace=True, axis=1)
HPtrainN['etecho1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN.shape

HPtrainN['epared1'].unique()
HPtrainN['epared2'].unique()
HPtrainN['epared2'].nunique()
HPtrainN['epared3'].unique()
HPtrainN['epared2'] = HPtrainN['epared2'].map({0: 0, 1: 2})

HPtestN['epared2'] = HPtestN['epared2'].map({0: 0, 1: 2})
HPtrainN['epared3'] = HPtrainN['epared3'].map({0: 0, 1: 3})

HPtestN['epared3'] = HPtestN['epared3'].map({0: 0, 1: 3})
HPtrainN['epared1'] = HPtrainN['epared1'] + HPtrainN['epared2'] + HPtrainN['epared3'] 

HPtestN['epared1'] = HPtestN['epared1'] + HPtestN['epared2'] + HPtestN['epared3'] 
HPtrainN.drop(['epared2', 'epared3'], inplace=True, axis=1)

HPtestN.drop(['epared2', 'epared3'], inplace=True, axis=1)
HPtrainN['epared1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['epared1'].value_counts()
HPtrainN.shape

HPtrainN['elimbasu1'].unique()
HPtrainN['elimbasu2'].unique()
HPtrainN['elimbasu3'].unique()
HPtrainN['elimbasu4'].unique()
HPtrainN['elimbasu5'].unique()
HPtrainN['elimbasu2'] = HPtrainN['elimbasu2'].map({0: 0, 1: 2})

HPtestN['elimbasu2'] = HPtestN['elimbasu2'].map({0: 0, 1: 2})
HPtrainN['elimbasu3'] = HPtrainN['elimbasu3'].map({0: 0, 1: 3})

HPtestN['elimbasu3'] = HPtestN['elimbasu3'].map({0: 0, 1: 3})
HPtrainN['elimbasu4'] = HPtrainN['elimbasu4'].map({0: 0, 1: 4})

HPtestN['elimbasu4'] = HPtestN['elimbasu4'].map({0: 0, 1: 4})
HPtrainN['elimbasu5'] = HPtrainN['elimbasu5'].map({0: 0, 1: 5})

HPtestN['elimbasu5'] = HPtestN['elimbasu5'].map({0: 0, 1: 5})
HPtrainN['elimbasu6'] = HPtrainN['elimbasu6'].map({0: 0, 1: 6})

HPtestN['elimbasu6'] = HPtestN['elimbasu6'].map({0: 0, 1: 6})
HPtrainN['elimbasu1'] = HPtrainN['elimbasu1'] +   HPtrainN['elimbasu2'] +  HPtrainN['elimbasu3'] +  HPtrainN['elimbasu4'] +  HPtrainN['elimbasu5'] +  HPtrainN['elimbasu6'] 

HPtestN['elimbasu1'] = HPtestN['elimbasu1'] +   HPtestN['elimbasu2'] +  HPtestN['elimbasu3'] +  HPtestN['elimbasu4'] +  HPtestN['elimbasu5'] +  HPtestN['elimbasu6'] 
HPtrainN.drop(['elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'], inplace=True, axis=1)

HPtestN.drop(['elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'], inplace=True, axis=1)
HPtrainN['elimbasu1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['elimbasu1'].value_counts()
HPtrainN.shape

HPtrainN['energcocinar1'].unique()
HPtrainN['energcocinar2'].unique()
HPtrainN['energcocinar3'].unique()
HPtrainN['energcocinar4'].unique()
HPtrainN['energcocinar2'] = HPtrainN['energcocinar2'].map({0: 0, 1: 2})

HPtestN['energcocinar2'] = HPtestN['energcocinar2'].map({0: 0, 1: 2})
HPtrainN['energcocinar3'] = HPtrainN['energcocinar3'].map({0: 0, 1: 3})

HPtestN['energcocinar3'] = HPtestN['energcocinar3'].map({0: 0, 1: 3})
HPtrainN['energcocinar4'] = HPtrainN['energcocinar4'].map({0: 0, 1: 4})

HPtestN['energcocinar4'] = HPtestN['energcocinar4'].map({0: 0, 1: 4})
HPtrainN['energcocinar1'] = HPtrainN['energcocinar1'] +   HPtrainN['energcocinar2'] +  HPtrainN['energcocinar3'] +  HPtrainN['energcocinar4'] 

HPtestN['energcocinar1'] = HPtestN['energcocinar1'] +   HPtestN['energcocinar2'] +  HPtestN['energcocinar3'] +  HPtestN['energcocinar4'] 
HPtrainN.drop(['energcocinar2','energcocinar3','energcocinar4'], inplace=True, axis=1)

HPtestN.drop(['energcocinar2','energcocinar3','energcocinar4'], inplace=True, axis=1)
HPtrainN['energcocinar1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['energcocinar1'].value_counts()
HPtrainN.shape

HPtrainN['v14a'].value_counts()
del HPtrainN['v14a']

del HPtestN['v14a']
HPtrainN['sanitario1'].value_counts()
HPtrainN['sanitario2'] = HPtrainN['sanitario2'].map({0: 0, 1: 2})

HPtestN['sanitario2'] = HPtestN['sanitario2'].map({0: 0, 1: 2})
HPtrainN['sanitario3'] = HPtrainN['sanitario3'].map({0: 0, 1: 3})

HPtestN['sanitario3'] = HPtestN['sanitario3'].map({0: 0, 1: 3})
HPtrainN['sanitario5'] = HPtrainN['sanitario5'].map({0: 0, 1: 5})

HPtestN['sanitario5'] = HPtestN['sanitario5'].map({0: 0, 1: 5})
HPtrainN['sanitario6'] = HPtrainN['sanitario6'].map({0: 0, 1: 6})

HPtestN['sanitario6'] = HPtestN['sanitario6'].map({0: 0, 1: 6})
HPtrainN['sanitario1'] = HPtrainN['sanitario1'] +   HPtrainN['sanitario2'] +  HPtrainN['sanitario3'] +  HPtrainN['sanitario5'] +  HPtrainN['sanitario6']

HPtestN['sanitario1'] = HPtestN['sanitario1'] +   HPtestN['sanitario2'] +  HPtestN['sanitario3'] +  HPtestN['sanitario5'] +  HPtestN['sanitario6']
HPtrainN.drop(['sanitario2','sanitario3','sanitario5','sanitario6'], inplace=True, axis=1)

HPtestN.drop(['sanitario2','sanitario3','sanitario5','sanitario6'], inplace=True, axis=1)
HPtrainN['sanitario1'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['sanitario1'].value_counts()
HPtrainN.shape

HPtrainN['public'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['planpri'] = HPtrainN['planpri'].map({0: 0, 1: 2})

HPtestN['planpri'] = HPtestN['planpri'].map({0: 0, 1: 2})
HPtrainN['noelec'] = HPtrainN['noelec'].map({0: 0, 1: 3})

HPtestN['noelec'] = HPtestN['noelec'].map({0: 0, 1: 3})
HPtrainN['coopele'] = HPtrainN['coopele'].map({0: 0, 1: 4})

HPtestN['coopele'] = HPtestN['coopele'].map({0: 0, 1: 4})
HPtrainN['public'] = HPtrainN['public'] + HPtrainN['planpri'] + HPtrainN['noelec'] + HPtrainN['coopele']

HPtestN['public'] = HPtestN['public'] + HPtestN['planpri'] + HPtestN['noelec'] + HPtestN['coopele']
HPtrainN.drop(['planpri', 'noelec', 'coopele'], inplace=True, axis=1)

HPtestN.drop(['planpri', 'noelec', 'coopele'], inplace=True, axis=1)
HPtrainN['public'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['public'] = HPtrainN['public'].map({0: 3, 1:1,2:2,3:3,4:4})
HPtrainN['public'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['public'].value_counts()
HPtrainN.shape

HPtrainN['paredblolad'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['paredzocalo'] = HPtrainN['paredzocalo'].map({0: 0, 1: 2})

HPtestN['paredzocalo'] = HPtestN['paredzocalo'].map({0: 0, 1: 2})
HPtrainN['paredpreb'] = HPtrainN['paredpreb'].map({0: 0, 1: 3})

HPtestN['paredpreb'] = HPtestN['paredpreb'].map({0: 0, 1: 3})
HPtrainN['pareddes'] = HPtrainN['pareddes'].map({0: 0, 1: 4})

HPtestN['pareddes'] = HPtestN['pareddes'].map({0: 0, 1: 4})
HPtrainN['paredmad'] = HPtrainN['paredmad'].map({0: 0, 1: 5})

HPtestN['paredmad'] = HPtestN['paredmad'].map({0: 0, 1: 5})
HPtrainN['paredzinc'] = HPtrainN['paredzinc'].map({0: 0, 1: 6})

HPtestN['paredzinc'] = HPtestN['paredzinc'].map({0: 0, 1: 6})
HPtrainN['paredfibras'] = HPtrainN['paredfibras'].map({0: 0, 1: 7})

HPtestN['paredfibras'] = HPtestN['paredfibras'].map({0: 0, 1: 7})
HPtrainN['paredother'] = HPtrainN['paredother'].map({0: 0, 1: 8})

HPtestN['paredother'] = HPtestN['paredother'].map({0: 0, 1: 8})
HPtrainN['paredblolad'] = HPtrainN['paredblolad'] + HPtrainN['paredzocalo'] + HPtrainN['paredpreb'] + HPtrainN['pareddes'] + HPtrainN['paredmad'] + HPtrainN['paredzinc'] + HPtrainN['paredfibras'] + HPtrainN['paredother']

HPtestN['paredblolad'] = HPtestN['paredblolad'] + HPtestN['paredzocalo'] + HPtestN['paredpreb'] + HPtestN['pareddes'] + HPtestN['paredmad'] + HPtestN['paredzinc'] + HPtestN['paredfibras'] + HPtestN['paredother']
HPtrainN.drop(['paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc','paredfibras','paredother'], inplace=True, axis=1)

HPtestN.drop(['paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc','paredfibras','paredother'], inplace=True, axis=1)
HPtrainN['paredblolad'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['paredblolad'].value_counts()
HPtrainN.shape

HPtrainN['pisomoscer'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['pisocemento'] = HPtrainN['pisocemento'].map({0: 0, 1: 2})

HPtestN['pisocemento'] = HPtestN['pisocemento'].map({0: 0, 1: 2})
HPtrainN['pisoother'] = HPtrainN['pisoother'].map({0: 0, 1: 3})

HPtestN['pisoother'] = HPtestN['pisoother'].map({0: 0, 1: 3})
HPtrainN['pisonatur'] = HPtrainN['pisonatur'].map({0: 0, 1: 4})

HPtestN['pisonatur'] = HPtestN['pisonatur'].map({0: 0, 1: 4})
HPtrainN['pisonotiene'] = HPtrainN['pisonotiene'].map({0: 0, 1: 5})

HPtestN['pisonotiene'] = HPtestN['pisonotiene'].map({0: 0, 1: 5})
HPtrainN['pisomadera'] = HPtrainN['pisomadera'].map({0: 0, 1: 6})

HPtestN['pisomadera'] = HPtestN['pisomadera'].map({0: 0, 1: 6})
HPtrainN['pisomoscer'] = HPtrainN['pisomoscer'] + HPtrainN['pisocemento'] + HPtrainN['pisoother'] + HPtrainN['pisonatur'] + HPtrainN['pisonotiene'] + HPtrainN['pisomadera']

HPtestN['pisomoscer'] = HPtestN['pisomoscer'] + HPtestN['pisocemento'] + HPtestN['pisoother'] + HPtestN['pisonatur'] + HPtestN['pisonotiene'] + HPtestN['pisomadera']
HPtrainN.drop(['pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera'], inplace=True, axis=1)

HPtestN.drop(['pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera'], inplace=True, axis=1)
HPtrainN['pisomoscer'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['pisomoscer'].value_counts()
HPtrainN.shape

HPtrainN['techozinc'].value_counts()
HPtrainN['techoentrepiso'].value_counts()
HPtrainN['techocane'].value_counts()
HPtrainN['techootro'].value_counts()
HPtrainN['techoentrepiso'] = HPtrainN['techoentrepiso'].map({0: 0, 1: 2})

HPtestN['techoentrepiso'] = HPtestN['techoentrepiso'].map({0: 0, 1: 2})
HPtrainN['techocane'] = HPtrainN['techocane'].map({0: 0, 1: 3})

HPtestN['techocane'] = HPtestN['techocane'].map({0: 0, 1: 3})
HPtrainN['techootro'] = HPtrainN['techootro'].map({0: 0, 1: 4})

HPtestN['techootro'] = HPtestN['techootro'].map({0: 0, 1: 4})
HPtrainN['techozinc'] = HPtrainN['techozinc'] + HPtrainN['techoentrepiso'] + HPtrainN['techocane'] + HPtrainN['techootro']

HPtestN['techozinc'] = HPtestN['techozinc'] + HPtestN['techoentrepiso'] + HPtestN['techocane'] + HPtestN['techootro']
HPtrainN.drop(['techoentrepiso', 'techocane', 'techootro'], inplace=True, axis=1)

HPtestN.drop(['techoentrepiso', 'techocane', 'techootro'], inplace=True, axis=1)
HPtrainN['techozinc'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['techozinc'].value_counts()
HPtrainN['techozinc'] = HPtrainN['techozinc'].map({0: 4, 1: 1, 2:2, 3:3, 4:4})
HPtrainN['techozinc'].value_counts()
HPtrainN.shape

HPtrainN['abastaguadentro'].value_counts()
HPtrainN['abastaguafuera'] = HPtrainN['abastaguafuera'].map({0: 0, 1: 2})

HPtestN['abastaguafuera'] = HPtestN['abastaguafuera'].map({0: 0, 1: 2})
HPtrainN['abastaguano'] = HPtrainN['abastaguano'].map({0: 0, 1: 3})

HPtestN['abastaguano'] = HPtestN['abastaguano'].map({0: 0, 1: 3})
HPtrainN['abastaguadentro'] = HPtrainN['abastaguadentro'] + HPtrainN['abastaguafuera'] + HPtrainN['abastaguano']

HPtestN['abastaguadentro'] = HPtestN['abastaguadentro'] + HPtestN['abastaguafuera'] + HPtestN['abastaguano']
HPtrainN.drop(['abastaguafuera', 'abastaguano'], inplace=True, axis=1)

HPtestN.drop(['abastaguafuera', 'abastaguano'], inplace=True, axis=1)
HPtrainN['abastaguadentro'].value_counts()
HPtrainN.shape

HPtrainN.drop(['v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2'], inplace=True, axis=1)

HPtestN.drop(['v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2'], inplace=True, axis=1)
print(HPtrainN.shape)

print(HPtestN.shape)

HPtrainN['hacapo'].value_counts()
HPtrainN.loc[HPtrainN['hacapo']==1,['idhogar', 'r4t3','rooms', 'bedrooms', 'overcrowding']]

HPtrainN.dtypes
HPtrainN['dependency'].value_counts()
HPtrainN.loc[HPtrainN['dependency']=='no',['idhogar', 'dependency','r4t3', 'hogar_nin', 'hogar_adul', 'hogar_mayor']]
HPtrainN.loc[HPtrainN['dependency']=='yes',['idhogar', 'dependency','r4t3', 'hogar_nin', 'hogar_adul', 'hogar_mayor']]
HPtrainN['dependency'] = HPtrainN['dependency'].replace({'no': 0, 'yes': 1})

HPtestN['dependency'] = HPtestN['dependency'].replace({'no': 0, 'yes': 1})
HPtrainN['dependency'].value_counts()
HPtrainN['dependency'].dtype
# Finally we convert it to numeric 
HPtrainN['dependency'] = pd.to_numeric(HPtrainN['dependency'])

HPtestN['dependency'] = pd.to_numeric(HPtestN['dependency'])

HPtrainN['edjefe'].value_counts()
HPtrainN['edjefe'].isnull().sum()
HPtrainN['edjefe'].dtype
HPtrainN['edjefe'] = HPtrainN['edjefe'].replace({'no': 0, 'yes': 1})

HPtestN['edjefe'] = HPtestN['edjefe'].replace({'no': 0, 'yes': 1})
HPtrainN['edjefe'].dtype
HPtrainN['edjefe'] = pd.to_numeric(HPtrainN['edjefe'])

HPtestN['edjefe'] = pd.to_numeric(HPtestN['edjefe'])
HPtrainN['edjefe'].value_counts().plot(kind='bar')
sns.despine

HPtrainN['edjefa'].value_counts()
HPtrainN['edjefa'] = HPtrainN['edjefa'].replace({'no': 0, 'yes': 1})

HPtestN['edjefa'] = HPtestN['edjefa'].replace({'no': 0, 'yes': 1})
HPtrainN['edjefa'].dtype
HPtrainN['edjefa'] = pd.to_numeric(HPtrainN['edjefa'])
HPtestN['edjefa'] = pd.to_numeric(HPtestN['edjefa'])
HPtrainN['edjefa'].dtype
HPtrainN['edjefa'].value_counts().plot(kind='bar')
sns.despine

HPtrainN['idhogar']= HPtrainN['idhogar'].astype('category')
HPtrainN['idhogar']= HPtrainN['idhogar'].cat.codes

HPtestN['idhogar']= HPtestN['idhogar'].astype('category')
HPtestN['idhogar']= HPtestN['idhogar'].cat.codes
HPtrainN['Target'].value_counts().plot(kind='bar')
sns.despine
HPtrainN['Target'].value_counts()

HPtrainN.dtypes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn import linear_model
X = HPtrainN.drop(['Target','idhogar'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['r4t3']

del HPtestN['r4t3']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['bedrooms']

del HPtestN['bedrooms']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['epared1']

del HPtestN['epared1']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['eviv1']

del HPtestN['eviv1']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['sanitario1']

del HPtestN['sanitario1']
X = HPtrainN.drop(['Target','idhogar'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['rooms']

del HPtestN['rooms']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['abastaguadentro']

del HPtestN['abastaguadentro']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['refrig']

del HPtestN['refrig']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['energcocinar1']

del HPtestN['energcocinar1']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['meaneduc']

del HPtestN['meaneduc']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['hogar_adul']

del HPtestN['hogar_adul']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['etecho1']

del HPtestN['etecho1']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['overcrowding']

del HPtestN['overcrowding']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['techozinc']

del HPtestN['techozinc']
X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
del HPtrainN['idhogar']

del HPtestN['idhogar']
print(HPtrainN.shape)

print(HPtestN.shape)
HPtrainN.columns.values
HPtestN.columns.values

HPtrainN['Target'].value_counts()
print('Target1', round(HPtrainN['Target'].value_counts()[1]/len(HPtrainN) * 100,2), '% of the dataset')
print('TArget2', round(HPtrainN['Target'].value_counts()[2]/len(HPtrainN) * 100,2), '% of the dataset')
print('Target3', round(HPtrainN['Target'].value_counts()[3]/len(HPtrainN) * 100,2), '% of the dataset')
print('TArget4', round(HPtrainN['Target'].value_counts()[4]/len(HPtrainN) * 100,2), '% of the dataset')
HPtrainN = HPtrainN.sample(frac=1)

Target1_df = HPtrainN.loc[HPtrainN['Target'] == 1]
Target2_df = HPtrainN.loc[HPtrainN['Target'] == 2]
Target3_df = HPtrainN.loc[HPtrainN['Target'] == 3]
Target4_df = HPtrainN.loc[HPtrainN['Target'] == 4][:1034]

normal_distributed_df = pd.concat([Target1_df,Target2_df,Target3_df,Target4_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

print(new_df.head())
print(new_df.shape)
from sklearn.model_selection import train_test_split
X = new_df.drop('Target',axis=1)
y = new_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
# Classifier Libraries
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
# Let's implement simple classifiers

classifiers = {
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}
from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)


# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_


# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf=grid_tree.best_estimator_

# Random Forest Classifier
rfcl = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
#rfcl = RandomForestClassifier()
param_grid = { 'n_estimators': [600,700,800],  'max_features': ['auto','sqrt','log2']}
rfc_grid = GridSearchCV(estimator=rfcl, param_grid=param_grid)
rfc_grid.fit(X_train, y_train)
rfc=rfc_grid.best_estimator_
print(rfc_grid.best_params_)
# Overfitting Case

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

randomforest_score = cross_val_score(rfc, X_train, y_train, cv=5)
print('RandomForest Classifier Cross Validation Score', round(randomforest_score.mean() * 100, 2).astype(str) + '%')
rfc_pred = rfc.predict(X_test)
from sklearn.feature_selection import RFE
names=list(new_df.columns)
#rank all features, i.e continue the elimination until the last one
rfe = RFE(rfc, n_features_to_select=10, step=1)
rfe.fit(X,y)
print('Features sorted by their rank:')
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

# Plot feature importance
feature_importance = rfc.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
#def plot_confusion_matrix(cm,title='Confusion matrix',cmap=plt.cm.Blues):
def plot_confusion_matrix(cm, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
import itertools
from itertools import product
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,rfc_pred)
plot_confusion_matrix(cm, normalize=False, title='Confusion matrix',cmap=plt.cm.Purples)
print(classification_report(y_test,rfc_pred))
X_whole = HPtrainN.drop('Target',axis=1)
y_whole = HPtrainN['Target']
X_whole.shape
rfc_pred_whole = rfc.predict(X_whole)
cm_whole=confusion_matrix(y_whole,rfc_pred_whole)
plot_confusion_matrix(cm_whole, normalize=False, title='Confusion matrix',cmap=plt.cm.Purples)
print(classification_report(y_whole,rfc_pred_whole))
rfc_pred_test = rfc.predict(HPtestN)
dftest = pd.DataFrame({'Target': rfc_pred_test})
dftest.head()
dftest.shape
test_predictN=test_predict[['idhogar']].copy()
test_predictN.head()
test_predictN.drop_duplicates(subset='idhogar', inplace=True)
test_predictN.head()
test_predictN.shape
test_predictN.index = range(len(test_predictN))
test_predictN.head()
test_pred_concat= pd.concat([test_predictN,dftest], axis=1)
test_pred_concat.shape
test_pred_concat.head()
test_predict= pd.merge(test_predict, test_pred_concat, how='left', on=['idhogar'])
test_predict.head()
test_predict=test_predict.drop(['idhogar'],axis=1)
test_predict.head()
test_predict.shape
test_predict.to_csv('test_predict.csv',index=False)