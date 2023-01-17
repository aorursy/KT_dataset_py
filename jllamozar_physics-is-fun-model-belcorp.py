# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing] Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def simple_mem_reduce(df):

    for col in df.columns:

        if df[col].dtype == int:

            m = df[col].max()

            if m > np.iinfo(np.uint32).max:

                df[col] = df[col].astype(np.uint64)

            elif m > np.iinfo(np.uint16).max:

                df[col] = df[col].astype(np.uint32)

            elif m > np.iinfo(np.uint8).max:

                df[col] = df[col].astype(np.uint16)

            elif m < np.iinfo(np.uint8).max:

                df[col] = df[col].astype(np.uint8)

                

        elif df[col].dtype == float:

            m = df[col].max()

            if m > np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float64)

            elif m > np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float32)

            elif m < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float16)

        

    return df



from pylab import *

import pandas as pd

from pandas.plotting import scatter_matrix

import seaborn as sns

from scipy import stats

import random

import sys 

import gc

import time



from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold

from sklearn.metrics import roc_curve, auc, precision_recall_curve

from sklearn.preprocessing import minmax_scale, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

from sklearn.metrics import recall_score, precision_score, r2_score, roc_auc_score

from sklearn.svm import SVC

from pandas.plotting import scatter_matrix

from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier ,  RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor

from scipy import spatial

from sklearn.neural_network import MLPClassifier

from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso

from sklearn.feature_selection import RFE, RFECV

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.cluster import KMeans

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA 

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from scipy.optimize import curve_fit



def KVS_TEST(dfX, y, pv0 = .005):

    """Kolmogorov-Smirnov TEST """

    names = []

    PV = []

    ST = []

    #Z = dfX.copy()

    for n in dfX.columns:

        X0 = dfX[n] ;

        z0 = X0[y==0].values

        z1 = X0[y==1].values

        st, pv = stats.ks_2samp(z0, z1)

        if pv <pv0:

            names.append(n)

            PV.append(pv)

            ST.append(st)

    gc.collect()

    return names, array(PV), array(ST)



def order_camp(x):

    x= array(x)

    nc, y = mod(x, 100), (x//100)#.astype(np.uint64)

    z = (y -2018)*18 + nc

    #print(nc, y)

    return z



def find_corr(df, cc = .7, CMAP = 'viridis', N =1, method = 'pearson'):

    """df : DataFrame """

    CORR = df.corr(method = method)

    CORR[abs(CORR)< cc ] =0

    CC = CORR.copy()

    IC = []

    #CC.fillna(1, inplace = True)

    for i, II in enumerate(CC.index):

    #    print( (CC.iloc[i, i:]>0).sum() )

    #    print(CC.iloc[i, i:]>0 )

        if (  abs( CC.iloc[i, i:] ) >0 ).sum()>1:

            IC.append(II)

    IC = IC +    CORR.columns[CORR.isna().sum() == df.columns.size].tolist()  

    INC = list(CC.columns)

    [ INC.remove(c) for c in IC]

    figure(N, figsize = (12,6))

    subplot(1,2,1),  sns.heatmap(CORR, cmap = CMAP)

    CC = df[INC].corr()

    #CC[abs(CC)<cc] = 0

    subplot(1,2,2), sns.heatmap(CC, cmap = CMAP)

    tight_layout()

    show()

    return INC
pwd
maestro_consultora = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/maestro_consultora.csv')

campanha_consultora  = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/campana_consultora.csv')

submission = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/predict_submission.csv')

maestro_producto = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/maestro_producto.csv')

#df = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/dtt_fvta_cl.csv')

#maestro_producto = maestro_producto[maestro_producto.columns[1:]]

#print(maestro_producto.shape, df.shape)

maestro_producto.head()
print(campanha_consultora.shape)

campanha_consultora = campanha_consultora[campanha_consultora.columns[1:]]

campanha_consultora.head()
import matplotlib.pyplot as plt

import seaborn as sns

ax = sns.countplot(x="campana", hue="Flagpasopedido", data=campanha_consultora)

plt.xticks(rotation = 70);
print(maestro_consultora.shape)

maestro_consultora = maestro_consultora[maestro_consultora.columns[1:]]

maestro_consultora.head()
print(campanha_consultora.shape)

COLC = []

# no se considerara los elementos 'flag'

for c in campanha_consultora.columns:

    print(c, campanha_consultora[c].unique().size, campanha_consultora[c].dtypes)

    if campanha_consultora[c].unique().size <4:

        print(campanha_consultora[c].unique())

    if campanha_consultora[c].unique().size >2:

        COLC.append(c)

print(campanha_consultora.columns)

print(COLC)
campanha_consultora.columns
maestro_consultora.head()
#funcion que cuenta los numeros de valores por cada indice

def data_count(data, c = ['idproducto'], index_  = 'idconsultora'):

    

    if len(c)==1:

        c = c[0]

        X_train_pi = pd.crosstab(data[index_], data[c]).astype(np.uint16)

        X_train_pi.columns = [c + "_" + str(v) for v in X_train_pi.columns]

        #X_train_pi = X_train_pi/X_train_pi.sum()

        gc.collect()

    else :

        k = 0

        X_train_pi = []

        print(data[c].shape)

        for cc in c:

            print("haciendo", cc, k)

            temp = pd.crosstab(data[index_], data[cc])

            temp.columns = [cc + "_" + str(v) for v in temp.columns]

            #temp = temp/temp.sum()

            X_train_pi.append(temp.astype(np.uint16) )

            k = k +1

        X_train_pi = pd.concat(X_train_pi, axis=1)

        gc.collect()

    return X_train_pi
%%time

X = campanha_consultora[campanha_consultora.campana <201906][COLC].copy()

X = X.set_index('IdConsultora').copy()

X['campana'] = order_camp(X.campana) 

X['max_campana'] = X.groupby('IdConsultora')['campana'].max()

X['tiempo_campana'] = X.max_campana - X.campana
%%time

# la data de campanha_consultora se dividira en ventanas de los ultimos 6 meses y los 3 siguientes 

###################################################

def X_TRAIN(X, NP= 4):

    X = X[X.tiempo_campana<NP]

    X1_ = X.groupby('IdConsultora')['cantidadlogueos'].mean()

    X2_ = X['geografia']

    X.drop(columns = ['cantidadlogueos', 'geografia'], inplace = True)





    X_train= data_count(X.fillna(-999).reset_index(), c = X.columns[1:-2], index_ = 'IdConsultora')

    data_train = X.reset_index()[['IdConsultora', 'campana']].groupby(['IdConsultora']).campana.count()

    for c in X_train.columns:

        X_train[c] = X_train[c]/data_train





    X2_ = data_count(X2_.reset_index(), c = ['geografia'], index_ = 'IdConsultora')

    for c in X2_.columns:

        X2_[c] = X2_[c]/data_train





    y_prev = campanha_consultora[['campana', 'IdConsultora','Flagpasopedido']]

    y_train = pd.Series(0, index = X_train.index)

    idx = np.intersect1d(y_prev[(y_prev.campana == 201906) & (y_prev.Flagpasopedido == 1)].IdConsultora, X_train.index)

    y_train.loc[idx] = 1

    print(y_train.sum(), y_train.shape)



    names, p, s = KVS_TEST(X_train, y_train, pv0 = .05)

    print(len(names))

    X_train = X_train[names]

    NN = find_corr(X_train, cc = .95)

    print(len(NN))

    X_train = X_train[NN]

    X_train  = pd.concat([X_train, X1_, X2_], axis = 1)

    return X_train, y_train





def X_TRAIN_BACK(X, NP= 4):

    X = X[ (X.tiempo_campana<2*NP) & (X.tiempo_campana>=NP)]

    

    X1_ = X.groupby('IdConsultora')['cantidadlogueos'].mean()

    X2_ = X['geografia']

    X.drop(columns = ['cantidadlogueos', 'geografia'], inplace = True)





    X_train= data_count(X.fillna(-999).reset_index(), c = X.columns[1:-2], index_ = 'IdConsultora')

    data_train = X.reset_index()[['IdConsultora', 'campana']].groupby(['IdConsultora']).campana.count()

    for c in X_train.columns:

        X_train[c] = X_train[c]/data_train





    X2_ = data_count(X2_.reset_index(), c = ['geografia'], index_ = 'IdConsultora')

    for c in X2_.columns:

        X2_[c] = X2_[c]/data_train





    y_prev = campanha_consultora[['campana', 'IdConsultora','Flagpasopedido']]

    y_train = pd.Series(0, index = X_train.index)

    idx = np.intersect1d(y_prev[(y_prev.campana == 201906) & (y_prev.Flagpasopedido == 1)].IdConsultora, X_train.index)

    y_train.loc[idx] = 1

    print(y_train.sum(), y_train.shape)



    names, p, s = KVS_TEST(X_train, y_train, pv0 = .05)

    print(len(names))

    X_train = X_train[names]

    NN = find_corr(X_train, cc = .95)

    print(len(NN))

    X_train = X_train[NN]

    X_train  = pd.concat([X_train, X1_, X2_], axis = 1)

    X_train.columns = [c + '_back' for c in X_train.columns]

    return X_train
%%time



###################################################

def X_TEST(X, NP= 4):

    X = X[X.tiempo_campana<NP]

    X1_ = X.groupby('IdConsultora')['cantidadlogueos'].mean()

    X2_ = X['geografia']

    X.drop(columns = ['cantidadlogueos', 'geografia'], inplace = True)





    X_train= data_count(X.fillna(-999).reset_index(), c = X.columns[1:-2], index_ = 'IdConsultora')

    data_train = X.reset_index()[['IdConsultora', 'campana']].groupby(['IdConsultora']).campana.count()

    for c in X_train.columns:

        X_train[c] = X_train[c]/data_train





    X2_ = data_count(X2_.reset_index(), c = ['geografia'], index_ = 'IdConsultora')

    for c in X2_.columns:

        X2_[c] = X2_[c]/data_train





    X_train  = pd.concat([X_train, X1_, X2_], axis = 1)

    return X_train





def X_TEST_BACK(X, NP= 4):

    X = X[ (X.tiempo_campana<2*NP) & (X.tiempo_campana>=NP)]

    

    X1_ = X.groupby('IdConsultora')['cantidadlogueos'].mean()

    X2_ = X['geografia']

    X.drop(columns = ['cantidadlogueos', 'geografia'], inplace = True)





    X_train= data_count(X.fillna(-999).reset_index(), c = X.columns[1:-2], index_ = 'IdConsultora')

    data_train = X.reset_index()[['IdConsultora', 'campana']].groupby(['IdConsultora']).campana.count()

    for c in X_train.columns:

        X_train[c] = X_train[c]/data_train





    X2_ = data_count(X2_.reset_index(), c = ['geografia'], index_ = 'IdConsultora')

    for c in X2_.columns:

        X2_[c] = X2_[c]/data_train



    X_train  = pd.concat([X_train, X1_, X2_], axis = 1)

    X_train.columns = [c + '_back' for c in X_train.columns]

    return X_train
X_train, y_train  = X_TRAIN(X, 6)
X_trainB  = X_TRAIN_BACK(X, 3)
X_trainB.head()
X_train = pd.concat([X_train, X_trainB], axis = 1)

print(X_train.shape)

NN = find_corr(X_train, cc = .95)

X_train = X_train[NN]

print(X_train.shape)
%%time

X = campanha_consultora[COLC].copy()

X = X.set_index('IdConsultora').copy()

X['campana'] = order_camp(X.campana) 

X['max_campana'] = X.groupby('IdConsultora')['campana'].max()

X['tiempo_campana'] = X.max_campana - X.campana
X_test  = X_TEST(X, 6)

X_testB  = X_TEST_BACK(X, 3)
X_test = pd.concat([X_test, X_testB], axis =1)

X_test = X_test[X_train.columns]

print(X_test.shape)

X_test.head()
IX = intersect1d(X_train.index, X_test.index)

X1, X2 = X_train.loc[IX]['cantidadlogueos'] , X_test.loc[IX][ 'cantidadlogueos']

scatter(X1, X2, alpha = .5, s = 3);

x1 = linspace(X1.min(), X1.max())

plot(x1, x1, 'k--');
#maestro_consultora.fillna(-999, inplace = True)

# no se considerara estas columns ['flagconsultoradigital', 'flagsupervisor', 'flagcorreovalidad', 'flagcelularvalidado',

                                #'campanaprimerpedido']

maestro_consultora.drop(columns = ['flagconsultoradigital', 'flagsupervisor', 'flagcorreovalidad', 'flagcelularvalidado',

                                'campanaprimerpedido'], inplace = True)

#maestro_consultora.campanaingreso = (maestro_consultora.campanaingreso//100)

maestro_consultora.head()

print(X_train.shape, X_test.shape)



maestro_consultora.head()
maestro_consultora.estadocivil.replace(np.nan, 'Otros' , inplace = True)

maestro_consultora.estadocivil.value_counts()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

maestro_consultora.estadocivil = le.fit_transform(maestro_consultora.estadocivil)

maestro_consultora.campanaultimopedido =order_camp(maestro_consultora.campanaultimopedido)

maestro_consultora.head()
# CAMPAÑA DE ULTIMO PEDIDO, ESTE FEAUTURE SE AÑADIRA AL FINAL. 

z  = maestro_consultora[['campanaultimopedido','IdConsultora']].set_index('IdConsultora').copy()

IZ = intersect1d( z.index, y_train.index)

scatter(IZ, z.loc[IZ], c = y_train.loc[IZ] , alpha = .5, s = 4)

grid()
# como se observa para valores menores a el pedido no se realiza.

#para evitar el data leakage reeplazaremos los valores menores  a 24

z[z.campanaultimopedido<24] = 24

print(z.min())

z.head()
Z = maestro_consultora.set_index('IdConsultora')[['estadocivil', 'edad']]

Z.head()
print(X_train.shape, X_test.shape)

X_train = pd.merge(X_train, Z.reset_index(), how="left", on=["IdConsultora"]).set_index('IdConsultora')

X_test = pd.merge(X_test, Z.reset_index(), how="left", on=["IdConsultora"]).set_index('IdConsultora')

print(X_train.shape, X_test.shape)

X_train.head()
!ls  /kaggle/input/physics-is-fun-data2/
%%time

X_testID = pd.read_csv('/kaggle/input/physics-is-fun-data2/test_prodID_F.csv', index_col = 'idconsultora').astype(np.float32)

X_train_count = pd.read_csv('/kaggle/input/physics-is-fun-data2/train_count_F.csv', index_col = 'idconsultora').astype(np.float32)

X_test_count = pd.read_csv('/kaggle/input/physics-is-fun-data2/test_count_F.csv', index_col = 'idconsultora').astype(np.float32)

X_trainID = pd.read_csv('/kaggle/input/physics-is-fun-data2/train_prodID_F.csv', index_col = 'idconsultora').astype(np.float32)

print(X_trainID.shape, X_train_count.shape, X_testID.shape, X_test_count.shape)

X_trainID.head()
#BEST 500 FEATURES BY X_trainID and X_train_count

COL500 = ['palancapersonalizacion_-999', 'codigotipooferta_7', 'tipo_MAQUILLAJE/PARA OJOS/SOMBRAS', 'codigotipooferta_106', 'tipo_MAQUILLAJE/PARA LABIOS/DELINEADOR LABIOS', 'codigotipooferta_40', 'tipo_TRATAMIENTO FACIAL/PARA ROSTRO/DESMANCHADORAS-ACLARADORAS', 'tipo_MAQUILLAJE/PARA ROSTRO/CORRECTORES', 'idproducto_200092902', 'idproducto_200094487', 'tipo_TRATAMIENTO FACIAL/PARA ROSTRO/NUTRITIVAS REVITALIZADORA', 'codigotipooferta_24', 'idproducto_200056638', 'idproducto_200029898', 'codigotipooferta_3', 'idproducto_210090591', 'idproducto_200094901', 'tipo_TRATAMIENTO CORPORAL/MANOS/HUMECTANTES-NUTRITIVAS', 'codigotipooferta_127', 'tipo_VARIOS/VARIOS/VARIOS', 'idproducto_200086411', 'idproducto_200090386', 'codigotipooferta_4', 'idproducto_210085083', 'idproducto_210089715', 'idproducto_200096723', 'idproducto_200095632', 'idproducto_200089499', 'tipo_MAQUILLAJE/PARA OJOS/MASCARA', 'codigotipooferta_34', 'unidadnegocio_APOYO', 'idproducto_210090992', 'idproducto_200086432', 'palancapersonalizacion_Ofertas Para tí', 'codigopalancapersonalizacion_42.0', 'codigotipooferta_48', 'idproducto_200095771', 'tipo_PROMOCION USUARIOS/COMPLEMENTOS/BLOQUE FRAGANCIAS', 'codigotipooferta_9', 'idproducto_200095903', 'tipo_CUIDADO PERSONAL/TALCOS/ESPECIFICOS', 'codigotipooferta_16', 'palancapersonalizacion_App Consultora Contenedor Showroom Ficha', 'tipo_CUIDADO PERSONAL/TALCOS/DAMAS', 'codigotipooferta_36', 'idproducto_210089819', 'codigotipooferta_18', 'codigotipooferta_14', 'idproducto_200096885', 'idproducto_200095791', 'idproducto_210089662', 'palancapersonalizacion_Ofertas Para ti', 'idproducto_200056623', 'idproducto_210091231', 'idproducto_200095931', 'idproducto_200091731', 'idproducto_210086323', 'codigotipooferta_49', 'idproducto_200085692', 'palancapersonalizacion_Desktop Pedido Digitado', 'grupooferta_-999', 'idproducto_200094924', 'idproducto_210089618', 'palancapersonalizacion_App Consultora Pedido Digitado', 'idproducto_200091288', 'codigotipooferta_26', 'idproducto_210089603', 'tipo_PROMOCION USUARIOS/HOGAR/BLOQUE TF', 'palancapersonalizacion_Mobile Pedido Digitado', 'idproducto_200094909', 'tipo_TRATAMIENTO CORPORAL/AREAS ESPECIFICAS/HUMECTANTES-NUTRITIVAS', 'codigotipooferta_123', 'idproducto_210088204', 'idproducto_200091898', 'idproducto_200091845', 'idproducto_200094066', 'codigotipooferta_128', 'idproducto_200094927', 'codigotipooferta_1', 'tipo_PROMOCION USUARIOS/COMPLEMENTOS/FRAGANCIA PUNTUAL', 'idproducto_200084882', 'codigotipooferta_202', 'idproducto_200094907', 'idproducto_200085129', 'idproducto_210090104', 'idproducto_200094929', 'idproducto_200091891', 'idproducto_210087419', 'idproducto_210090122', 'idproducto_200087725', 'tipo_TRATAMIENTO FACIAL/PARA OJOS/NUTRITIVAS REVITALIZADORA', 'tipo_ACCESORIOS COSMETICOS/MAQUILLAJE/VARIOS', 'idproducto_200094926', 'grupooferta_PROMOCION USUARIO', 'idproducto_200095342', 'idproducto_200093884', 'idproducto_210090586', 'idproducto_200091888', 'idproducto_200092933', 'idproducto_210089596', 'idproducto_200092901', 'codigotipooferta_225', 'idproducto_200094910', 'codigotipooferta_114', 'tipo_FRAGANCIAS/UNISEX/COLONIA Y O EAU DE TOILETE', 'idproducto_200089488', 'categoria_FINART/BIJOUTERIE', 'idproducto_210088845', 'idproducto_200088259', 'idproducto_200095425', 'idproducto_210090139', 'idproducto_210085609', 'codigotipooferta_35', 'idproducto_210089449', 'idproducto_200082554', 'idproducto_200090953', 'codigotipo_554', 'idproducto_210090765', 'idproducto_210089724', 'idproducto_210089631', 'palancapersonalizacion_Showroom', 'idproducto_200042737', 'idproducto_210090395', 'codigotipooferta_51', 'idproducto_200087009', 'codigotipo_041', 'idproducto_210089810', 'idproducto_200091530', 'idproducto_210089446', 'idproducto_210090723', 'idproducto_200092568', 'idproducto_200095895', 'idproducto_200091230', 'codigotipo_094', 'canalingresoproducto_APP', 'idproducto_200050018', 'codigotipooferta_213', 'idproducto_200086396', 'tipo_MAQUILLAJE/PARA ROSTRO/BRONCEADOR', 'idproducto_200094738', 'canalingresoproducto_WEB', 'idproducto_200092313', 'idproducto_210085583', 'codigotipooferta_205', 'idproducto_210090584', 'codigotipooferta_117', 'tipo_DAMAS/SPORT ELEGANTE/PULS.CUERO', 'codigotipooferta_50', 'tipo_HOGAR/TIPO/ORGANIZADORES', 'idproducto_200094720', 'idproducto_200089972', 'idproducto_200088595', 'codigopalancapersonalizacion_4812.0', 'idproducto_200094149', 'idproducto_200090505', 'idproducto_210088926', 'idproducto_210087919', 'idproducto_200085127', 'idproducto_210089630', 'idproducto_200096887', 'idproducto_200039855', 'idproducto_200088516', 'idproducto_200083943', 'idproducto_210048754', 'idproducto_200082639', 'codigopalancapersonalizacion_4020401.0', 'idproducto_210089665', 'idproducto_200089627', 'idproducto_200094552', 'tipo_HOGAR/TIPO/ACCESORIOS', 'idproducto_200094964', 'idproducto_200096883', 'idproducto_200085151', 'codigopalancapersonalizacion_-999.0', 'tipo_HOGAR/TIPO/BELLEZA Y SALUD', 'idproducto_200090367', 'tipo_PROMOCION USUARIOS/HOGAR/BLOQUE BEBE', 'idproducto_200090178', 'tipo_COMPLEMENTOS/TIPO/BILLETERA', 'idproducto_200091859', 'tipo_TRATAMIENTO CORPORAL/CUERPO/HUMECTANTES-NUTRITIVAS', 'tipo_TRATAMIENTO FACIAL/PARA ROSTRO/MASCARILLA FACIAL', 'idproducto_210086322', 'idproducto_200076945', 'idproducto_200096703', 'idproducto_210089451', 'idproducto_200094715', 'tipo_CUIDADO PERSONAL/DESODORANTES/ROLL-ON', 'idproducto_200091532', 'idproducto_200094719', 'idproducto_200085777', 'idproducto_200094717', 'idproducto_210086794', 'idproducto_210085584', 'idproducto_200085733', 'idproducto_200091781', 'codigotipooferta_33', 'tipo_COMPLEMENTOS/TIPO/MOCHILA', 'idproducto_200094903', 'idproducto_210089820', 'idproducto_200090647', 'idproducto_200072404', 'codigotipooferta_212', 'idproducto_210088530', 'idproducto_200089168', 'tipo_MAQUILLAJE/PARA ROSTRO/ILUMINADOR', 'idproducto_200091344', 'codigotipooferta_208', 'idproducto_200085010', 'idproducto_210090402', 'tipo_CUIDADO PERSONAL/TALCOS/NI#OS-BEBES', 'idproducto_200091981', 'idproducto_210089734', 'idproducto_210090560', 'idproducto_200064286', 'idproducto_200084551', 'codigotipo_190', 'idproducto_200085649', 'tipo_PROMOCION USUARIOS/COMPLEMENTOS/BLOQUE TF', 'idproducto_200079531', 'codigotipooferta_64', 'idproducto_210090589', 'codigotipooferta_203', 'idproducto_200086833', 'idproducto_210089586', 'idproducto_210085436', 'idproducto_200085656', 'idproducto_200089487', 'idproducto_210090528', 'idproducto_200085133', 'categoria_HOGAR', 'idproducto_200069929', 'idproducto_200086399', 'idproducto_200085006', 'codigotipooferta_6', 'idproducto_200093964', 'idproducto_200095346', 'idproducto_210085408', 'palancapersonalizacion_App Consultora Landing Ganadoras Ganadoras Ficha', 'idproducto_200088574', 'idproducto_200036444', 'idproducto_200091328', 'codigotipo_526', 'palancapersonalizacion_Mobile Landing Ofertas Para Ti Ofertas Para Ti Ficha', 'idproducto_200093364', 'idproducto_200088568', 'idproducto_200087581', 'marca_LBEL', 'idproducto_210091362', 'idproducto_200091900', 'tipo_COMPLEMENTOS/TENDENCIA/BILLETERA', 'idproducto_200093330', 'idproducto_200083913', 'idproducto_200082525', 'idproducto_200084504', 'tipo_MAQUILLAJE/PARA ROSTRO/RUBOR', 'idproducto_200087252', 'idproducto_200091960', 'palancapersonalizacion_App Consultora Pedido Ofertas Para Ti Carrusel', 'idproducto_200094173', 'idproducto_200094178', 'idproducto_200082524', 'idproducto_200085752', 'tipo_HOGAR/TIPO/ACCESORIO ELECTRONICO', 'idproducto_200084550', 'idproducto_200093239', 'idproducto_200090429', 'idproducto_200085428', 'codigotipo_098', 'idproducto_200088602', 'codigotipo_999', 'idproducto_200085609', 'codigotipo_040', 'idproducto_200091269', 'idproducto_200087327', 'idproducto_200094904', 'idproducto_210049364', 'idproducto_200093872', 'idproducto_200094713', 'idproducto_200087965', 'codigotipooferta_224', 'codigotipooferta_107', 'idproducto_200086172', 'idproducto_200085139', 'idproducto_200085008', 'tipo_COMPLEMENTOS/TEMATICOS/CARTERA', 'idproducto_200082514', 'tipo_PROMOCION USUARIOS/COMPLEMENTOS/BLOQUE BEBE', 'idproducto_200090424', 'idproducto_200090991', 'idproducto_200085203', 'idproducto_200084501', 'tipo_MAQUILLAJE/PARA U#AS/BASE DE UÑAS', 'idproducto_200088615', 'idproducto_200091892', 'codigotipooferta_15', 'idproducto_200085914', 'codigotipooferta_25', 'idproducto_200094716', 'idproducto_210056393', 'idproducto_210087767', 'idproducto_200084116', 'marca_CYZONE', 'idproducto_200086414', 'idproducto_200082678', 'tipo_FRAGANCIAS/NI#OS-NI#AS/COLONIA Y O EAU DE TOILETE', 'idproducto_200060753', 'palancapersonalizacion_App Consultora Landing Showroom Showroom Ficha', 'codigotipooferta_115', 'tipo_FRAGANCIAS/FAMILIAR BA#O/COLONIA DE BA#O', 'idproducto_210089660', 'codigotipooferta_60', 'idproducto_200082691', 'idproducto_200088336', 'codigotipo_508', 'idproducto_200089508', 'idproducto_200095404', 'idproducto_210085429', 'idproducto_200094172', 'palancapersonalizacion_Oferta Final', 'codigotipooferta_46', 'idproducto_200091357', 'idproducto_200085091', 'codigotipooferta_29', 'idproducto_200088606', 'idproducto_200069856', 'idproducto_200085181', 'idproducto_210088979', 'idproducto_200083615', 'codigotipooferta_116', 'idproducto_200090372', 'idproducto_210088145', 'idproducto_200069684', 'idproducto_200085608', 'idproducto_200091343', 'idproducto_200058539', 'idproducto_210090500', 'idproducto_200086037', 'idproducto_200092163', 'idproducto_210083869', 'palancapersonalizacion_App Consultora Contenedor Oferta Del D�a Ficha', 'idproducto_200091709', 'idproducto_200092164', 'idproducto_200084554', 'idproducto_200082560', 'tipo_TRATAMIENTO FACIAL/PARA ROSTRO/LIMPIADORAS DEMAQUILLADORE', 'idproducto_200084840', 'tipo_MAQUILLAJE/PARA ROSTRO/POLVOS', 'idproducto_200087962', 'idproducto_200056624', 'idproducto_200087955', 'idproducto_210089616', 'codigotipo_366', 'idproducto_200090349', 'idproducto_200090373', 'canalingresoproducto_-999', 'codigotipo_025', 'idproducto_200084102', 'idproducto_200085180', 'tipo_CUIDADO PERSONAL/SHAMPOO/ADULTOS', 'idproducto_200092744', 'codigopalancapersonalizacion_22.0', 'idproducto_210089821', 'tipo_TRATAMIENTO FACIAL/PARA ROSTRO/HUMECTANTES', 'idproducto_200086385', 'idproducto_200091094', 'idproducto_210085428', 'idproducto_200089290', 'idproducto_200085007', 'idproducto_200089293', 'idproducto_200094878', 'idproducto_200089498', 'idproducto_200088610', 'idproducto_200078930', 'idproducto_200087985', 'tipo_CUIDADO PERSONAL/DESODORANTES/AEROSOL', 'codigotipo_497', 'idproducto_200073557', 'idproducto_200076877', 'idproducto_200092333', 'idproducto_200086106', 'idproducto_210088522', 'tipo_CUIDADO PERSONAL/JABONES/JABONES INTIMOS', 'codigopalancapersonalizacion_12.0', 'idproducto_210089850', 'idproducto_200088472', 'idproducto_210091305', 'palancapersonalizacion_Mobile Landing Showroom Showroom Ficha', 'idproducto_200092165', 'idproducto_200085153', 'idproducto_210085643', 'idproducto_210088764', 'tipo_HOGAR/SET/ORGANIZADORES', 'idproducto_200086397', 'idproducto_200088474', 'idproducto_200091964', 'idproducto_200090085', 'idproducto_210085438', 'idproducto_200059089', 'tipo_COMPLEMENTOS/TEMATICOS/MOCHILA', 'tipo_LENTES/DAMAS/MODERNO', 'idproducto_200088384', 'idproducto_200094711', 'codigotipooferta_220', 'idproducto_200082533', 'idproducto_200085175', 'idproducto_200093149', 'codigotipooferta_44', 'idproducto_200088132', 'categoria_LENTES', 'codigotipooferta_209', 'idproducto_210088777', 'idproducto_210085413', 'idproducto_200088613', 'idproducto_200087010', 'tipo_BIJOUTERIE/NI#AS/TWEENS', 'idproducto_200090369', 'marca_ESIKA', 'idproducto_200063985', 'categoria_MAQUILLAJE', 'codigopalancapersonalizacion_4020002.0', 'idproducto_200085137', 'idproducto_200084298', 'tipo_CUIDADO PERSONAL/SHAMPOO/BEBES', 'idproducto_200091886', 'idproducto_200050939', 'tipo_MAQUILLAJE/PARA OJOS/DELINEADOR OJOS', 'idproducto_210085582', 'idproducto_200034758', 'codigotipooferta_201', 'idproducto_210089268', 'idproducto_200092742', 'idproducto_200093834', 'tipo_CUIDADO PERSONAL/TRATAMIENTO CAPILAR/ADULTOS', 'idproducto_200095343', 'idproducto_200090385', 'codigotipo_016', 'codigotipo_128', 'categoria_PROMOCION USUARIOS', 'categoria_TRATAMIENTO FACIAL', 'idproducto_200084602', 'codigotipooferta_8', 'idproducto_200092280', 'idproducto_210085407', 'codigotipo_101', 'tipo_BIJOUTERIE/DAMAS/ESTUCHE DE ARETES X4', 'tipo_MAQUILLAJE/PARA OJOS/EMBELLECEDOR', 'idproducto_210085410', 'grupooferta_DEMO + GVTAS', 'tipo_MAQUILLAJE/PARA OJOS/DELINEADOR CEJAS', 'idproducto_210088527', 'grupooferta_ARRASTRE', 'palancapersonalizacion_App Consultora Landing Ofertas Para Ti Ofertas Para Ti Ficha', 'idproducto_200095935', 'codigotipo_238', 'idproducto_200091762', 'codigotipooferta_11', 'codigotipo_398', 'idproducto_200093488', 'grupooferta_NUEVA COLECCION', 'codigotipooferta_30', 'codigotipo_243', 'idproducto_200088517', 'idproducto_200097376', 'codigotipo_001', 'idproducto_200093146', 'idproducto_200091553', 'codigopalancapersonalizacion_4712.0', 'idproducto_200061181', 'categoria_MUESTRAS COSMETICOS', 'codigotipooferta_13', 'tipo_MAQUILLAJE/PARA U#AS/ESMALTE', 'idproducto_200093298', 'palancapersonalizacion_Desktop Landing Ofertas Para Ti Ofertas Para Ti Carrusel', 'idproducto_210086542', 'idproducto_200093960', 'idproducto_200088570', 'idproducto_200095102', 'unidadnegocio_ACCESORIOS', 'idproducto_200091340', 'idproducto_210083868', 'codigotipo_488', 'tipo_MAQUILLAJE/MIXTO/MULTIFUNCIONAL', 'idproducto_200095151', 'idproducto_200087964', 'idproducto_200059780', 'idproducto_200056813', 'idproducto_200087694', 'tipo_TRATAMIENTO FACIAL/PARA ROSTRO/DEMAQUILLADORES', 'codigotipooferta_214', 'idproducto_200036215', 'tipo_FRAGANCIAS/CABALLEROS/COLONIA Y O EAU DE TOILETE', 'idproducto_200089222', 'idproducto_200091133', 'categoria_CUIDADO PERSONAL', 'tipo_MAQUILLAJE/PARA OJOS/PRODUCTO PARA CEJAS', 'idproducto_200091092', 'idproducto_200090455', 'idproducto_200088739', 'idproducto_200090384']

print(len(COL500))
X_train1 = pd.concat([X_trainID, X_train_count], axis = 1)

X_test1 = pd.concat([X_testID, X_test_count], axis = 1)

del X_trainID, X_train_count, X_testID, X_test_count

X_train1 = X_train1[COL500]

X_test1 = X_test1[COL500]

gc.collect()
X_train = pd.concat([X_train, X_train1], axis = 1)

X_test = pd.concat([X_test, X_test1], axis = 1)

del X_train1, X_test1

gc.collect()

print(X_train.shape, X_test.shape)

X_train.head()
X_train.head()
c = 'codigofactura_APP'

scatter( X_train[c].loc[X_train.index] , X_test[c].loc[X_train.index] ,alpha = .5, s = 3)


gc.collect()

print(X_train.shape, X_test.shape)

#/kaggle/input/times-series-numerical/X_train_ts_binnig.csv

!ls  /kaggle/input/physics-is-fun-data1/
# DATA 1

X_train_ts = pd.read_csv('/kaggle/input/physics-is-fun-data1/X_train_ts_binnig.csv', index_col='idconsultora').astype(np.float32)

X_test_ts = pd.read_csv('/kaggle/input/physics-is-fun-data1/X_test_ts_binnig.csv', index_col='idconsultora').astype(np.float32)

print(X_train_ts.shape, X_test_ts.shape)
X_train = pd.concat([X_train, X_train_ts], axis = 1)

X_test = pd.concat([X_test, X_test_ts], axis = 1)

del X_train_ts, X_test_ts

print(X_train.shape, X_test.shape)

gc.collect()


print(X_train.shape, X_test.shape)

X_train.fillna(0, inplace = True)

X_test.fillna(0, inplace = True)
print(X_train.shape, X_test.shape, y_train.shape)
### best features

COLF = ['codigofactura_-999', 'evaluacion_nuevas_C_2d2', 'codigofactura_-999_back', 'segmentacion_Nuevas_back', 'evaluacion_nuevas_C_4d4', 'descuento_log_sh_6', 'evaluacion_nuevas_C_5d5', 'segmentacion_Nivel2', 'realvtamncatalogo_log_sh_6', 'evaluacion_nuevas_I_3d5', 'palancapersonalizacion_-999', 'evaluacion_nuevas_I_1d2', 'evaluacion_nuevas_I_3d4', 'codigotipooferta_106', 'evaluacion_nuevas_I_1d3', 'segmentacion_Nivel2_back', 'segmentacion_Nivel5', 'evaluacion_nuevas_I_4d5', 'realuuvendidas_log_sh_6', 'segmentacion_Nivel4', 'idproducto_200090386', 'evaluacion_nuevas_C_3d3', 'cantidadlogueos', 'idproducto_200094901', 'evaluacion_nuevas_I_5d6', 'idproducto_210089715', 'idproducto_210091231', 'ahorro_log_diff_2', 'idproducto_200096885', 'idproducto_210090591', 'evaluacion_nuevas_C_6d6', 'idproducto_210089662', 'palancapersonalizacion_App Consultora Contenedor Showroom Ficha', 'idproducto_200095903', 'descuento_log_diff_9', 'cantidadlogueos_back', 'idproducto_200096723', 'realvtamncatalogo_log_sh_8', 'idproducto_200095791', 'preciocatalogo_log_sh_2', 'idproducto_200092902', 'idproducto_200091288', 'idproducto_200094924', 'codigopalancapersonalizacion_42.0', 'evaluacion_nuevas_C_3d3_back', 'idproducto_200094927', 'idproducto_210089819', 'ahorro_log_sh_5', 'idproducto_200094487', 'realuuvendidas_log_sh_7', 'codigotipooferta_225', 'idproducto_200095425', 'palancapersonalizacion_Ofertas Para tí', 'idproducto_210090992', 'palancapersonalizacion_Desktop Pedido Digitado', 'idproducto_210089449', 'descuento_log_diff_1', 'palancapersonalizacion_Mobile Pedido Digitado', 'idproducto_210089810', 'idproducto_210090139', 'idproducto_200092933', 'codigofactura_APP_back', 'codigopalancapersonalizacion_4812.0', 'codigotipooferta_24', 'idproducto_210086542', 'tipo_HOGAR/TIPO/ACCESORIOS', 'palancapersonalizacion_App Consultora Pedido Digitado', 'idproducto_210088204', 'edad', 'segmentacion_Nivel5_back', 'evaluacion_nuevas_C_1d1_back', 'tipo_PROMOCION USUARIOS/COMPLEMENTOS/FRAGANCIA PUNTUAL', 'idproducto_200094738', 'idproducto_200094552', 'tipo_PROMOCION USUARIOS/HOGAR/BLOQUE TF', 'idproducto_200095931', 'idproducto_210088530', 'unidadnegocio_APOYO', 'preciocatalogo_log_diff_5', 'idproducto_210090586', 'tipo_COMPLEMENTOS/TIPO/BILLETERA', 'idproducto_200095895', 'idproducto_200095771', 'preciocatalogo_log_sh_10', 'idproducto_200092165', 'idproducto_210089820', 'idproducto_200091981', 'idproducto_200085127', 'idproducto_200091133', 'idproducto_200097376', 'idproducto_200082524', 'idproducto_200085733', 'palancapersonalizacion_Showroom', 'idproducto_200089488', 'palancapersonalizacion_App Consultora Pedido Ofertas Para Ti Carrusel', 'codigotipo_101', 'idproducto_200088474', 'tipo_HOGAR/TIPO/ORGANIZADORES', 'idproducto_200094716', 'idproducto_200093364', 'codigotipo_526', 'idproducto_210090402', 'idproducto_210085413', 'idproducto_210085583', 'idproducto_200094715', 'codigotipo_094', 'idproducto_200085175', 'idproducto_200082554', 'tipo_MAQUILLAJE/PARA ROSTRO/BRONCEADOR', 'idproducto_200092568', 'idproducto_200092163', 'tipo_MAQUILLAJE/PARA ROSTRO/ILUMINADOR', 'idproducto_200091532', 'idproducto_210088522', 'idproducto_200093239', 'codigotipooferta_1', 'palancapersonalizacion_Mobile Landing Ofertas Para Ti Ofertas Para Ti Ficha', 'idproducto_200094173', 'idproducto_200091891', 'idproducto_200085181', 'idproducto_200092901', 'idproducto_200094066', 'idproducto_210056393', 'idproducto_200091357', 'codigotipooferta_201', 'idproducto_200090429', 'idproducto_200090953', 'idproducto_200091845', 'preciocatalogo_log_10', 'codigopalancapersonalizacion_4020401.0', 'idproducto_200087725', 'idproducto_200088574', 'idproducto_200094713', 'idproducto_200085151', 'tipo_HOGAR/TIPO/BELLEZA Y SALUD', 'tipo_CUIDADO PERSONAL/TALCOS/DAMAS', 'idproducto_210088926', 'idproducto_200086106', 'idproducto_200091230', 'idproducto_200093298', 'idproducto_210083869', 'idproducto_200084504', 'codigotipooferta_208', 'descuento_log_1', 'idproducto_200087252', 'codigotipooferta_212', 'idproducto_200094717', 'codigotipooferta_220', 'idproducto_200084501', 'idproducto_200088613', 'codigotipooferta_128', 'idproducto_200084550', 'idproducto_200088739', 'idproducto_200088606', 'idproducto_200090424', 'canalingresoproducto_WEB', 'codigotipooferta_205', 'tipo_MAQUILLAJE/MIXTO/MULTIFUNCIONAL', 'idproducto_200056813', 'idproducto_200090647', 'idproducto_200085692', 'idproducto_200094904', 'tipo_COMPLEMENTOS/TIPO/MOCHILA', 'ahorro_log_diff_10', 'idproducto_200042737', 'idproducto_200091892', 'ahorro_log_5', 'idproducto_210090528', 'idproducto_200085428', 'idproducto_200090385', 'idproducto_200090367', 'idproducto_210085407', 'tipo_CUIDADO PERSONAL/JABONES/JABONES INTIMOS', 'palancapersonalizacion_Desktop Landing Ofertas Para Ti Ofertas Para Ti Carrusel', 'idproducto_200085777', 'idproducto_200093872', 'idproducto_200085609', 'idproducto_200090369', 'idproducto_200088384', 'idproducto_200095151', 'idproducto_200094711', 'idproducto_200085137', 'tipo_COMPLEMENTOS/TEMATICOS/MOCHILA', 'idproducto_200082678', 'idproducto_200091886', 'idproducto_200084882', 'idproducto_200093330', 'idproducto_200088517', 'idproducto_200085203', 'idproducto_200087009', 'idproducto_200083913', 'idproducto_200036444', 'codigotipo_128', 'palancapersonalizacion_App Consultora Landing Showroom Showroom Ficha', 'tipo_MAQUILLAJE/PARA U#AS/BASE DE UÑAS', 'idproducto_200088132', 'idproducto_200084840', 'codigotipooferta_107', 'idproducto_200093488', 'idproducto_200093964', 'idproducto_200092280', 'idproducto_200091530', 'idproducto_200094720', 'idproducto_200085006', 'idproducto_200083943', 'idproducto_200085133', 'idproducto_200087965', 'idproducto_200091269', 'idproducto_200091731', 'idproducto_200082525', 'idproducto_200082533', 'idproducto_200089290', 'idproducto_200088472', 'idproducto_200088336', 'tipo_MAQUILLAJE/PARA OJOS/PRODUCTO PARA CEJAS', 'idproducto_200084554', 'idproducto_200088570', 'idproducto_210089821', 'palancapersonalizacion_Oferta Final', 'idproducto_200058539', 'tipo_HOGAR/SET/ORGANIZADORES', 'realvtamncatalogo_log_diff_5', 'idproducto_200082639', 'idproducto_200096703', 'idproducto_210085408', 'tipo_CUIDADO PERSONAL/TALCOS/NI#OS-BEBES', 'tipo_COMPLEMENTOS/TEMATICOS/CARTERA', 'codigotipo_497', 'idproducto_200094964', 'idproducto_200092742', 'idproducto_210049364', 'idproducto_200091094', 'idproducto_200092313', 'codigotipooferta_46', 'idproducto_210087419', 'idproducto_200069929', 'idproducto_200087962', 'codigotipo_238', 'idproducto_200093834', 'idproducto_200088602', 'codigotipo_190', 'idproducto_200089498', 'tipo_COMPLEMENTOS/TENDENCIA/BILLETERA', 'codigotipooferta_209', 'realuuvendidas_log_1', 'tipo_TRATAMIENTO CORPORAL/AREAS ESPECIFICAS/HUMECTANTES-NUTRITIVAS', 'codigotipo_508', 'idproducto_210091305', 'idproducto_200086399', 'idproducto_200093884', 'categoria_MUESTRAS COSMETICOS', 'idproducto_200094149', 'idproducto_200085007', 'tipo_HOGAR/TIPO/ACCESORIO ELECTRONICO', 'idproducto_210089665', 'idproducto_200087985', 'tipo_MAQUILLAJE/PARA ROSTRO/RUBOR', 'codigotipooferta_16', 'idproducto_210090589', 'idproducto_200086397', 'idproducto_200076877', 'idproducto_200034758', 'idproducto_200056624', 'idproducto_210085438', 'codigotipo_040', 'idproducto_200061181', 'tipo_LENTES/DAMAS/MODERNO', 'codigotipooferta_214', 'idproducto_200089627', 'idproducto_200091328', 'idproducto_200088615', 'idproducto_200086414', 'idproducto_200091709', 'realuuvendidas_log_3', 'idproducto_200085608', 'idproducto_200094903', 'idproducto_200084602', 'idproducto_200092333', 'idproducto_200091340', 'idproducto_200083615', 'idproducto_210085609', 'idproducto_200086385', 'idproducto_200060753', 'idproducto_200093146', 'idproducto_200091343', 'ahorro_log_10', 'idproducto_200088259', 'idproducto_200090384', 'codigotipooferta_64', 'codigotipo_398', 'idproducto_210087767', 'canalingresoproducto_APP', 'codigotipooferta_26', 'descuento_log_diff_5', 'tipo_MAQUILLAJE/PARA U#AS/ESMALTE', 'categoria_FINART/BIJOUTERIE', 'codigotipooferta_35', 'codigopalancapersonalizacion_12.0', 'codigopalancapersonalizacion_22.0', 'codigotipooferta_50']

print(len(COLF))
# agrupamos con los datos de la ultima campaña

IIXX = intersect1d(X_train.index, z.index)

X_train = pd.concat([X_train[COLF], z.loc[IIXX]], axis = 1)

X_test = pd.concat([X_test[COLF], z], axis = 1)

print(X_train.shape, X_test.shape)
y_prev = campanha_consultora[['campana', 'IdConsultora','Flagpasopedido']]



y_train = pd.Series(0, index = X_train.index)

idx = np.intersect1d(y_prev[(y_prev.campana == 201906) & (y_prev.Flagpasopedido == 1)].IdConsultora, X_train.index)



y_train.loc[idx] = 1

print(y_train.sum(), y_train.shape)

y_train.head()



from sklearn.utils import shuffle





#X_train, y_train =  shuffle(X_train, y_train , random_state=999)

%%time



from sklearn.model_selection import TimeSeriesSplit

NFOLDS = 10

#folds = StratifiedKFold(n_splits=NFOLDS )

folds = KFold(n_splits=NFOLDS )



#folds = TimeSeriesSplit(n_splits=NFOLDS )

columns = X_train.columns #COLF #X_train.columns# #X_train.columns #COL785[0:200] #X_train.columns #COL785[0:200] #X_train.columns #['flagpasopedidoweb_1.0_21'] + COL220F  #X_train.columns #COL600[0:300] # X_train.columns # X_train.columns #COL500_ORD[0:200] #

splits = folds.split(X_train, y_train)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X_train.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



YP = []

 

model = XGBClassifier(base_score=.194624935923356592, booster='gbtree',

       colsample_bylevel=1, colsample_bytree=1, gamma=0,

       #learning_rate=0.15625549474872114, max_delta_step=0, max_depth= 9,

       learning_rate= 0.2, max_delta_step=0, max_depth= 7, 

       min_child_weight=1, missing=None, n_estimators=400, n_jobs=-1,

       nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=10, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1 , tree_method='gpu_hist', predictor='gpu_predictor' )



for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train1, X_valid = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]

    y_train1, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(y_train1.sum(), y_valid.sum(), X_train1.shape, X_valid.shape)



    model.fit(X_train1, y_train1,

        eval_set=[(X_train1, y_train1), (X_valid, y_valid)],

        eval_metric='auc',

        verbose=False,

        early_stopping_rounds=60)

    feature_importances[f'fold_{fold_n + 1}'] = model.feature_importances_

    

    y_pred_valid = model.predict_proba(X_valid)[:,1]

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += model.predict_proba(X_test[columns])[:,1] / NFOLDS

    YP.append(model.predict_proba(X_test[columns])[:,1])

    del X_train1, X_valid, y_train1, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y_train, y_oof)}")

gc.collect()

plt.figure(figsize=(16, 16))

feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(5));
feature_importances.sort_values(by='average', ascending=False).head()
scatter(YP[0], YP[-1], s = 1, alpha = .5)
yp = pd.Series(y_preds, index = X_test.index)

#yp = pd.Series(YP[-1], index = X_test.index)



yp.head()
print(yp.shape, submission.shape, X_train.shape, X_test.shape)
submission = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/predict_submission.csv')

submission.head()


print(submission.shape, yp.shape)

IX = np.intersect1d(submission.idconsultora, yp.index)

submission.set_index('idconsultora', inplace = True)

submission.flagpasopedido = yp.loc[IX]

submission.reset_index(inplace = True)

submission.head()
submission.to_csv('submit_belcorp.csv', index= False)

from IPython.display import HTML





def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



create_download_link(filename='submit_belcorp.csv')
print(X_train.shape, X_test.shape, submission.shape)