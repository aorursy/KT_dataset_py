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
# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
limit_row = 900000

train = pd.read_csv('../input/train_ver2.csv', nrows=limit_row)

test = pd.read_csv('../input/test_ver2.csv')
#test.columns

#train['fecha_dato'] = pd.to_datetime(train['fecha_dato'])

#train['fecha_alta'] = pd.to_datetime(train['fecha_alta'])



#train['customer_asso'] = train.fecha_dato - train.fecha_alta



#test['fecha_dato'] = pd.to_datetime(test['fecha_dato'])

#test['fecha_alta'] = pd.to_datetime(test['fecha_alta'])



#test['customer_asso'] = test.fecha_dato - test.fecha_alta
del train['fecha_dato']

del train['fecha_alta']

del train['conyuemp']

del train['ult_fec_cli_1t']

del train['ind_nuevo']

del train['indrel']

del train['indrel_1mes']

del train['indfall']

del train['tipodom']

del train['nomprov']



del test['fecha_dato']

del test['fecha_alta']

del test['conyuemp']

del test['ult_fec_cli_1t']

del test['ind_nuevo']

del test['indrel']

del test['indrel_1mes']

del test['indfall']

del test['tipodom']

del test['nomprov']
train.pais_residencia = train.pais_residencia.fillna(train.pais_residencia.mode()[0])

train.sexo = train.sexo.fillna(train.sexo.mode()[0])

train.tiprel_1mes = train.tiprel_1mes.fillna(train.tiprel_1mes.mode()[0])

train.indresi = train.indresi.fillna(train.indresi.mode()[0])

train.indext = train.indext.fillna(train.indext.mode()[0])

train.cod_prov = train.cod_prov.fillna(train.cod_prov.mode()[0])

train.ind_actividad_cliente = train.ind_actividad_cliente.fillna(train.ind_actividad_cliente.mode()[0])

train.renta = train.renta.fillna(train.renta.mean())

train.segmento = train.segmento.fillna(train.segmento.mode()[0])

train.ind_nomina_ult1 = train.ind_nomina_ult1.fillna(train.ind_nomina_ult1.mode()[0])

train.ind_nom_pens_ult1 = train.ind_nom_pens_ult1.fillna(train.ind_nom_pens_ult1.mode()[0])

train.age = train.age.astype("str")

train.age = list(map (lambda x: x.strip(), train.age))

train.age = train.age.replace("NA",train.age.mode()[0])

train.age = train.age.astype('int')

train.ind_empleado = train.ind_empleado.fillna(train.ind_empleado.mode()[0])

train.canal_entrada = train.canal_entrada.fillna(train.canal_entrada.mode()[0])

#train.customer_asso = train.customer_asso.fillna(train.customer_asso.mean())
# define training and testing sets



X_train = train.drop(['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',

       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',

       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',

       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'], axis=1)



Y_train = train.drop(['ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age',

       'antiguedad', 'tiprel_1mes', 'indresi', 'indext', 'canal_entrada',

       'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento',], axis=1)



X_test  = test.copy()
#X_test['customer_asso'] = pd.to_numeric(X_test['customer_asso'])

#X_train['customer_asso'] = pd.to_numeric(X_test['customer_asso'])



#X_test.age = X_test.age.astype("str")

#X_test.age = list(map (lambda x: x.strip(), X_test.age))

#X_test.age = X_test.age.astype('int')
#X_train.segmento.isnull().sum()

X_train.segmento=list(map (lambda x: x.strip(), X_train.segmento))
#X_test = X_test.dropna()
X_train['type'] = 'train'

X_test['type'] = 'test'
frame = [X_train,X_test]
C = pd.concat(frame)
le = preprocessing.LabelEncoder()
le.fit(C.ind_empleado.unique())

C.ind_empleado = le.transform(C.ind_empleado)



le.fit(C.pais_residencia.unique())

C.pais_residencia = le.transform(C.pais_residencia)



le.fit(C.sexo.unique())

C.sexo = le.transform(C.sexo)



le.fit(C.tiprel_1mes.unique())

C.tiprel_1mes = le.transform(C.tiprel_1mes)



le.fit(C.indresi.unique())

C.indresi = le.transform(C.indresi)



le.fit(C.indext.unique())

C.indext = le.transform(C.indext)



le.fit(C.canal_entrada.unique())

C.canal_entrada = le.transform(C.canal_entrada)



le.fit(C.segmento.unique())

C.segmento = le.transform(C.segmento)
X_train_1 = C[C.type == 'train']

X_test_1 = C[C.type == 'test']
X_train_1 = X_train_1.drop('type', axis=1)

X_test_1 = X_test_1.drop('type', axis=1)
X_train_1.antiguedad = X_train_1.antiguedad.astype('str')

X_train_1.antiguedad=list(map (lambda x: x.strip(), X_train_1.antiguedad))

X_train_1.antiguedad = X_train_1.antiguedad.replace("NA",X_train_1.antiguedad.mode()[0])
X_test_1.renta = X_test_1.renta.astype('str')

X_test_1.renta = list(map(lambda x: x.strip(), X_test_1.renta))
X_test_1.renta = X_test_1.renta.replace("NA", "0")
# Random Forests



random_forest = RandomForestClassifier(n_estimators=10)



random_forest.fit(X_train_1,Y_train)



Y_pred = random_forest.predict(X_test_1)



# Random Forests got the best score

# From now on Random Forests is the chosen one

random_forest.score(X_train_1,Y_train)
subfile = []



for j in Y_pred:

    #print(j)

    tmp = []

    for i in range(len(j)):

        if j[i] == 1.0:

            tmp.append(Y_train.columns[i])

    subfile.append(tmp)
ncods=[]

for cd in X_test_1["ncodpers"]:

    ncods.append(cd)
finsub = []

finsub.append("ncodpers,added_products\n")

for i in range(len(ncods)):

    finsub.append(str(ncods[i])+","+",".join(subfile[i])+"\n")
fh = open("mysub.csv",'w')

for _ in finsub:

    fh.write(_)

fh.close()