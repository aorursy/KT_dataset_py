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
from pylab import *

import pandas as pd

from pandas.plotting import scatter_matrix

import seaborn as sns

from scipy import stats

import random

import sys 

import gc



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

from scipy.optimize import curve_fit

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def year_month_2_dt(MES):

    y = trunc(MES/100)

    m = mod(MES,100)

    m = array(m)

    m_ = zeros(m.size)

    for i, mm in enumerate(m):

        if mm<3:

            m_[i] = mm-2 + 12

            y.iloc[i] = y.iloc[i]-1

        else:

            m_[i] = mm-2

    date = pd.to_datetime(pd.DataFrame({'year':y,'month':m_, 'day':y*0 +1})) 

    return date



def year_month_dt(MES):

    y = trunc(MES/100)

    m = mod(MES,100)

    date = pd.to_datetime(pd.DataFrame({'year':y,'month':m, 'day':y*0 +1})) 

    return date









def year_month_2(MES):

    y = trunc(MES/100)

    m = mod(MES,100)

    m = array(m)

    m_ = zeros(m.size)

    for i, mm in enumerate(m):

        if mm<3:

            m_[i] = mm-2 + 12

            y.iloc[i] = y.iloc[i]-1

        else:

            m_[i] = mm-2

    date = y*100 + m_

    return date.astype('int')



def MES_to_X(MES):

    y = trunc(MES/100)

    m = mod(MES,100)

    x = (y-2017)*12 + m

    return x















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

    return names, PV, ST

df_producto = pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_PRODUCTOS_BCO_F.txt', index_col = 'ID_CLIENTE')

df_stock_train= pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_STOCK_VARS_TRAIN_F.txt', index_col = 'ID_CLIENTE')

df_stock_test= pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_STOCK_VARS_TEST_F.txt', index_col = 'ID_CLIENTE')

print(df_producto.shape, df_stock_train.shape , df_stock_test.shape)

df_producto.head()


MES_T0 = pd.concat([df_stock_train.MES_T0, df_stock_test.MES_T0], axis = 0)

MES_T0.head()



df_producto.head()
df_producto = pd.concat([df_producto, MES_T0.loc[df_producto.index] ], axis = 1)

df_producto.head()

print(df_producto.shape)

# works with months minor a month to evaluate



df_producto = df_producto[df_producto.MES<df_producto.MES_T0]

print(df_producto.shape)

df_producto.head()


df = pd.DataFrame()

df['name'] = df_producto.columns

df['counts'] = 0

CC = df_producto.columns

for k, c in enumerate(CC):

    #print(c, df_producto[c].unique().size)

    df['counts'].loc[k] = df_producto[c].unique().size

df.head()



names = df[df.counts>4].name.to_list()

namesD = df[(df.counts>1) &(df.counts<=4)].name.to_list()

df_producto_D = df_producto[namesD]

df_producto = df_producto[names]



a = find_corr(df_producto, cc = .95)

print(len(a))

df_producto = df_producto[a]



A = pd.get_dummies(df_producto_D.astype('str') )

A.head()





A = A.groupby('ID_CLIENTE').sum()

print(A.shape, df_producto_D.shape)

A.head()



df_producto_D = A

del A
df_producto.MES = MES_to_X(df_producto.MES)

df_producto.MES_T0 = MES_to_X(df_producto.MES_T0)



IX = intersect1d(df_stock_train.index , df_producto.index)

y_train = df_stock_train.loc[IX].FUGA_3M

print(len(IX), len(y_train))
def mean_dates_4(df, namecol = '_'):

    df0 = df.groupby('ID_CLIENTE').mean()

    df0.drop(columns = ['MES', 'MES_T0'], inplace = True)

    CC = df0.columns

    CC = [c + namecol  for c in CC]

    df0.columns = CC

    #######################3333

    A = df.loc[(df.MES > df.MES_T0-5) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '41' for c in CC]

    A.columns = CC

    df1 = A

    #####################3

    A = df.loc[(df.MES<=df.MES_T0-5) & (df.MES>df.MES_T0-9) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '42' for c in CC]

    A.columns = CC

    df2 = A

    #######################################################

    A = df.loc[(df.MES>df.MES_T0-13) & (df.MES<=df.MES_T0-9) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '43' for c in CC]

    A.columns = CC

    df3 = A

    DF = pd.concat([df0, df1, df2, df3], axis =  1)

    del df0, df1, df2, df3, A

    return DF



def mean_test(df0, IX, y_train, namecol = '_', pv0 = 1e-5 ):



    df = mean_dates_4(df0, namecol = namecol)

    

    names, pv, st = KVS_TEST(df.loc[IX], y_train , pv0 = pv0)

    u = find_corr(df[names], cc= .9)

    print( len(u) )

    df = df[u]

    return df 

dz = df_producto.reset_index().set_index(['ID_CLIENTE','MES', 'MES_T0'])

dz.head()
####### diff



df = dz.copy()

dff1 = reduce_mem_usage( (df -df.groupby(['ID_CLIENTE']).shift(1)) )

dff_sh = reduce_mem_usage( (df.groupby(['ID_CLIENTE']).shift(1)/df ) )

dflog = reduce_mem_usage( df.apply(np.log) )





dff_sh.replace([np.inf, -np.inf], np.nan, inplace = True)

dff_sh.fillna(1, inplace = True)

dff1.fillna(0, inplace = True)

dflog.replace([np.inf, -np.inf], np.nan, inplace = True)

dflog.fillna(1, inplace = True)
df_producto.head()
dz = mean_dates_4(df_producto)

print(dz.shape, df_producto.shape)
names, PV, ST = KVS_TEST(dz.loc[IX], y_train, pv0 = 1e-5)

print(len(names), dz.shape)

scatter(log10(array(PV)) , ( array(ST)) , alpha = .5, s = 10), grid()

sns.jointplot(x = log10( array(PV)+1e-300), y = ( array(ST)   ), kind="hex")
dfA = mean_test(df_producto, IX, y_train, namecol = '_', pv0 = 1e-3 )

dfD = mean_test(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_DIFF_', pv0 = 1e-3)

dfSH = mean_test(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_SH_', pv0 = 1e-3 )

dfLOG = mean_test( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_LOG_', pv0 = 1e-3 )



del  dff1, dff_sh, dflog, dz, df

DF = pd.concat([dfA, dfD, dfSH, dfLOG], axis = 1)

del dfA, dfD, dfSH, dfLOG

DF.head()
df = df_producto.reset_index().set_index(['ID_CLIENTE','MES', 'MES_T0'])



dfSTD =  df.groupby('ID_CLIENTE').std()

dfK = df.groupby(['ID_CLIENTE']).apply( pd.DataFrame.kurt )

dfSK = df.groupby(['ID_CLIENTE']).apply( pd.DataFrame.skew )



print(dfSTD.shape, dfK.shape, dfSK.shape)



gc.collect()

names, pv, st = KVS_TEST(dfSTD.loc[IX], y_train,  pv0 = 1e-3 )

dfSTD = dfSTD[names]

print(len(names))



names, pv, st = KVS_TEST(dfK.loc[IX], y_train,  pv0 = 1e-3 )

dfK = dfK[names] 

print(len(names))



names, pv, st = KVS_TEST(dfSK.loc[IX], y_train,  pv0 = 1e-3 )

dfSK = dfSK[names]

print(len(names))
cc = [ c + '_STD' for c in dfSTD.columns]

dfSTD.columns = cc



cc = [ c + '_KURT' for c in dfK.columns]

dfK.columns = cc



cc = [ c + '_SKEW' for c in dfSK.columns]

dfSK.columns = cc



dfST = pd.concat([dfSTD, dfK, dfSK], axis = 1)

del dfSTD, dfK, dfSK



dfST.head()
names = find_corr(dfST, cc= .9)

print(len(names))

dfST  = dfST[names]
DF_PRODUCTO = pd.concat([DF, dfST], axis = 1)

del dfST, df, df_producto

DF_PRODUCTO.head()


#CHI SQUARE TEST

from sklearn.feature_selection import  chi2

a, b = chi2(df_producto_D.loc[IX], y_train)

scatter(log10(b), log10(a), alpha = .5, s = 10), grid()

sns.jointplot(x = log10(b+1e-300), y = log10(a + 1e-300), kind="hex")



COLCD = df_producto_D.columns[a>1e2]

print(len(COLCD))

df_producto_D = df_producto_D[COLCD]

df_producto_D.head()
DF_PRODUCTO = pd.concat([DF_PRODUCTO, df_producto_D], axis = 1)

del df_producto_D

DF_PRODUCTO.head()
df_endeudamiento = pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_ENDEUDAMIENTO_F.txt', index_col = 'ID_CLIENTE')

gc.collect()

df_endeudamiento.head()

df = pd.DataFrame()

df['name'] = df_endeudamiento.columns

df['counts'] = 0

CC = df_endeudamiento.columns

for k, c in enumerate(CC):

    #print(c, df_producto[c].unique().size)

    df['counts'].loc[k] = df_endeudamiento[c].unique().size

df.sort_values('counts')[0:40]
print(df_endeudamiento.shape)

df_endeudamiento = pd.concat([df_endeudamiento, MES_T0.loc[df_endeudamiento.index] ], axis = 1)

# works with months minor a month to evaluate

df_endeudamiento = df_endeudamiento[df_endeudamiento.MES<df_endeudamiento.MES_T0]

print(df_endeudamiento.shape)

df_endeudamiento.head()
df_endeudamiento.MES = MES_to_X(df_endeudamiento.MES)

df_endeudamiento.MES_T0 = MES_to_X(df_endeudamiento.MES_T0)



IX = intersect1d(df_stock_train.index , df_endeudamiento.index)

y_train = df_stock_train.loc[IX].FUGA_3M

print(len(IX), len(y_train))


dz = df_endeudamiento.reset_index().set_index(['ID_CLIENTE','MES', 'MES_T0'])

dz.drop(columns = 'CD_BANCO', inplace = True)

dz.head()


df = dz.copy()

dff1 = reduce_mem_usage( (df -df.groupby(['ID_CLIENTE']).shift(1)) )

dff_sh = reduce_mem_usage( (df.groupby(['ID_CLIENTE']).shift(1)/df ) )

dflog = reduce_mem_usage( df.apply(np.log) )





dff_sh.replace([np.inf, -np.inf], np.nan, inplace = True)

dff_sh.fillna(1, inplace = True)

dff1.fillna(0, inplace = True)

dflog.replace([np.inf, -np.inf], np.nan, inplace = True)

dflog.fillna(1, inplace = True)
dz = mean_dates_4(df.reset_index().set_index('ID_CLIENTE') )

names, PV, ST = KVS_TEST(dz.loc[IX], y_train, pv0 = 1e-5)

print(len(names), dz.shape)

scatter(log10(array(PV)) , ( array(ST)) , alpha = .5, s = 10), grid()

sns.jointplot(x = log10( array(PV)+1e-300), y = ( array(ST)   ), kind="hex")
dfA = mean_test(df.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_', pv0 = 1e-3 )

dfD = mean_test(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_DIFF_', pv0 = 1e-3 )

dfSH = mean_test(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_SH_', pv0 = 1e-3 )

dfLOG = mean_test( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_LOG_', pv0 = 1e-3 )



del  dff1, dff_sh, dflog, dz, df

DF = pd.concat([dfA, dfD, dfSH, dfLOG], axis = 1)

del dfA, dfD, dfSH, dfLOG

DF.head()
df = df_endeudamiento.reset_index().set_index(['ID_CLIENTE','MES', 'MES_T0'])

df.drop(columns = 'CD_BANCO', inplace = True)

dfSTD =  df.groupby('ID_CLIENTE').std()

dfK = df.groupby(['ID_CLIENTE']).apply( pd.DataFrame.kurt )

dfSK = df.groupby(['ID_CLIENTE']).apply( pd.DataFrame.skew )



print(dfSTD.shape, dfK.shape, dfSK.shape)



names, pv, st = KVS_TEST(dfSTD.loc[IX], y_train,  pv0 = 1e-3 )

dfSTD = dfSTD[names]

print(len(names))



names, pv, st = KVS_TEST(dfK.loc[IX], y_train,  pv0 = 1e-3 )

dfK = dfK[names] 

print(len(names))



names, pv, st = KVS_TEST(dfSK.loc[IX], y_train,  pv0 = 1e-3)

dfSK = dfSK[names]

print(len(names))



cc = [ c + '_STD' for c in dfSTD.columns]

dfSTD.columns = cc



cc = [ c + '_KURT' for c in dfK.columns]

dfK.columns = cc



cc = [ c + '_SKEW' for c in dfSK.columns]

dfSK.columns = cc



dfST = pd.concat([dfSTD, dfK, dfSK], axis = 1)

del dfSTD, dfK, dfSK, df



dfST.head()
a = find_corr(dfST, cc = .95)

dfST = dfST[a]

print(len(a))
DF_ENDEUDAMIENTO = pd.concat([DF, dfST], axis = 1)

del dfST, DF

DF_ENDEUDAMIENTO.head()
dCD = df_endeudamiento['CD_BANCO'].apply(lambda x: 'CD_' + str(x)  )

dCD.head()
dCD1 = df_endeudamiento['ST_CREDITO'].apply(lambda x: 'ST_CREDITO_' + str(x)  )

dCD1.head()
dCD = pd.concat([dCD, dCD1], axis =1)

dCD.head()
dCD = pd.get_dummies(dCD)

print(dCD.shape)

dCD.head()
dCD = dCD.groupby('ID_CLIENTE').sum()

print(dCD.shape)

dCD.head()
from sklearn.feature_selection import  chi2

a, b = chi2(dCD.loc[IX], y_train)

scatter(log10(b), log10(a), alpha = .5, s = 10), grid()

sns.jointplot(x = log10(b + 1e-300), y = log10(a + 1e-300), kind="hex")



COLCD = dCD.columns[a>80]

print(len(COLCD))

dCD = dCD[COLCD]

dCD.head()
DF_ENDEUDAMIENTO = pd.concat([DF_ENDEUDAMIENTO,dCD ], axis = 1)

del dCD, df_endeudamiento

print(DF_ENDEUDAMIENTO.shape)

DF_ENDEUDAMIENTO.head()
df = pd.concat([df_stock_train, df_stock_test], axis =0)

ID_CLIENTE = df.index

df.head()
fh_n = df.FH_NACIMIENTO

fh_a = df.FH_ALTA

fh_n = pd.to_datetime(fh_n, format='%Y-%m-%d', errors='ignore').dt.year + (pd.to_datetime(fh_n, format='%Y-%m-%d', errors='ignore').dt.month - 1)/12

fh_a = pd.to_datetime(fh_a, format='%Y-%m-%d', errors='ignore').dt.year + (pd.to_datetime(fh_a, format='%Y-%m-%d', errors='ignore').dt.month - 1)/12



fh_n.fillna(value = fh_n.median(),inplace=True)

fh_a.fillna(value=fh_a.median(), inplace=True)

df.FH_NACIMIENTO = 2020-fh_n

df.FH_ALTA = 2020-fh_a

df.info()

df.head()



COLD = df.dtypes[df.dtypes == 'object'].index

COLD
[ print(df[c].unique(), size(df[c].unique() ) ) for c in COLD ]
COL_N = [ c for c in df.columns if c not in COLD]

#COL_N.remove( 'ID_CLIENTE')

COL_N
IX = intersect1d(df_stock_train.index , df.index)

y_train = df_stock_train.loc[IX].FUGA_3M

print(y_train.shape)

for c in COLD[1:]:

    df[c].fillna(-999, inplace = True)

df[COLD[1:]].head()
# Label Encoding

from sklearn.preprocessing import LabelEncoder

for f in COLD[1:]: 

    lbl = LabelEncoder()

    lbl.fit(list(df[f].values) )

    df[f] = lbl.transform(list(df[f].values))

df[COLD[1:]].head()
df.columns
DF_STOCK  = df[COL_N + COLD[1:].tolist()]

DF_STOCK.head()
XF = pd.concat([DF_STOCK, DF_ENDEUDAMIENTO, DF_PRODUCTO], axis =  1)

del DF_STOCK, DF_ENDEUDAMIENTO, DF_PRODUCTO

XF.head()
df_transac = pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_TRANSAC_CANALES_F.txt', index_col = 'ID_CLIENTE')

CNA  = [c for c in df_transac.columns if c.count('CT_')==1]

CNUM  = [c for c in df_transac.columns if c not in CNA]

print(df_transac.shape, len(CNA), len(CNUM) )

print(CNA)

print(CNUM)

df_transac['CTA'] = df_transac[CNA].mean(axis = 1) 

df_transac = df_transac[ list(CNUM) + ['CTA']]

df_transac.rename({'PERIODO': 'MES'}, axis=1, inplace=True)



print(df_transac.shape)

df_transac = pd.concat([df_transac, MES_T0.loc[df_transac.index] ], axis = 1)

# works with months minor a month to evaluate

df_transac = df_transac[df_transac.MES<df_transac.MES_T0]

print(df_transac.shape)

df_transac.head()
df_transac.MES = MES_to_X(df_transac.MES)

df_transac.MES_T0 = MES_to_X(df_transac.MES_T0)

IX1 = intersect1d(IX , df_transac.index)

y_train1 =y_train.loc[IX1]

print(len(IX1), len(y_train1))
dz = df_transac.reset_index().set_index(['ID_CLIENTE','MES', 'MES_T0'])



df = dz.copy()

dff1 = reduce_mem_usage( (df -df.groupby(['ID_CLIENTE']).shift(1)) )

dff_sh = reduce_mem_usage( (df.groupby(['ID_CLIENTE']).shift(1)/df ) )

dflog = reduce_mem_usage( df.apply(np.log) )





dff_sh.replace([np.inf, -np.inf], np.nan, inplace = True)

dff_sh.fillna(1, inplace = True)

dff1.fillna(0, inplace = True)

dflog.replace([np.inf, -np.inf], np.nan, inplace = True)

dflog.fillna(1, inplace = True)



dfA = mean_test(df.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_', pv0 = 1e-10 )

dfD = mean_test(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_DIFF_', pv0 = 1e-10 )

dfSH = mean_test(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_SH_', pv0 = 1e-10 )

dfLOG = mean_test( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_LOG_', pv0 = 1e-10 )



del  dff1, dff_sh, dflog, dz, df



DF = pd.concat([dfA, dfD, dfSH, dfLOG], axis = 1)

del dfA, dfD, dfSH, dfLOG

DF.head()
df = df_transac.reset_index().set_index(['ID_CLIENTE','MES', 'MES_T0'])

dfSTD =  df.groupby('ID_CLIENTE').std()

dfK = df.groupby(['ID_CLIENTE']).apply( pd.DataFrame.kurt )

dfSK = df.groupby(['ID_CLIENTE']).apply( pd.DataFrame.skew )



print(dfSTD.shape, dfK.shape, dfSK.shape)



names, pv , st = KVS_TEST(dfSTD.loc[IX], y_train,  pv0 = 1e-10 )

dfSTD = dfSTD[names]

print(len(names))



names, pv, st = KVS_TEST(dfK.loc[IX], y_train,  pv0 = 1e-10 )

dfK = dfK[names] 

print(len(names))



names, pv, st = KVS_TEST(dfSK.loc[IX], y_train,  pv0 = 1e-10 )

dfSK = dfSK[names]

print(len(names))



cc = [ c + '_STD' for c in dfSTD.columns]

dfSTD.columns = cc



cc = [ c + '_KURT' for c in dfK.columns]

dfK.columns = cc



cc = [ c + '_SKEW' for c in dfSK.columns]

dfSK.columns = cc



dfST = pd.concat([dfSTD, dfK, dfSK], axis = 1)

del dfSTD, dfK, dfSK, df



a = find_corr(dfST, cc = .95)

dfST = dfST[a]

print(len(a))



dfST.head()
DF_TRANSAC = pd.concat([DF, dfST], axis = 1)

del dfST, DF

DF_TRANSAC.head()
XF = pd.concat([XF , DF_TRANSAC ], axis = 1)



XF.drop(columns = 'FUGA_3M', inplace = True)

XF.fillna(-999, inplace = True)



X_train = XF.loc[IX].copy()

X_test = XF.loc[df_stock_test.index].copy()

print(X_train.shape, X_test.shape, XF.shape)

del XF
COLF_FEATURES = ['RGO_RIESGO_SIST', 'RGO_SDO_MEDIO_PASIVO', 'TP_VIVI', 'TP_PERSONA', 'RGO_PASIVO_LOG_41', 'RGO_SDO_VALORES_STD', 'NU_CTA_NOMINA_1', 'ST_CREDITO_42', 'RGO_SDPREST_EMP_41', 'RGO_PASIVO_42', 'RGO_SDINDIRECTO_LOG_41', 'ST_CREDITO_LOG_42', 'CD_BANCO_CD_6', 'RGO_SEGU_NO_VINC_42', 'RGO_SEGU_NO_VINC_LOG_42', 'RGO_PASIVO_LOG_42', 'RGO_RIEGO_BBVA', 'NU_CTA_NOMINA_0', 'RGO_SDPREST_EMP_42', 'RGO_MARGEN_OPER_MES', 'RGO_PASIVO_41', 'RGO_SDPREST_EMP_LOG_43', 'NU_CTA_TDC_PNATURAL_0', 'CD_SBS_CPP', 'ST_CREDITO_43', 'RGO_SDLEA_PEM_41', 'MES_T0', 'ST_CREDITO_LOG_43', 'RGO_SEGU_NO_VINC_STD', 'NU_CTA_TDD_STD', 'NU_CTA_CTS_SH_', 'RGO_SEGU_NO_VINC_41', 'CD_BANCO_CD_237', 'FH_ALTA', 'CD_SBS_DDP', 'RGO_SEGU_NO_VINC_LOG_41', 'NU_CTA_TDD_LOG_43', 'NU_CTA_TDD_KURT', 'RGO_SDPREST_EMP_43', 'NU_CTA_CTS_STD', 'RGO_MARGEN_AHORRO_LOG_41', 'RGO_SEGU_VINC_LOG_42', 'NU_CTA_TDD_SKEW', 'CD_SBS_NORMAL', 'RGO_MARGEN_COBRANZA_LIB_41', 'RGO_SDLEA_PEM_LOG_41', 'NU_CTA_TDD_SH_41', 'ST_CREDITO_SH_43', 'NU_CTA_TDD_SH_', 'RGO_MARGEN_COBRANZA_LIB_LOG_41', 'RGO_SDPREST_EMP_LOG_42', 'RGO_SDINDIRECTO_42', 'RGO_SDO_MEDIO_ACTIVO', 'RGO_SDINDIRECTO_LOG_43', 'RGO_SEGU_VINC_41', 'RGO_MARGEN_COBRANZA_LIB_LOG_42', 'RGO_SDLEA_PEM_SH_42', 'RGO_SDMICROEMPRESA_41', 'RGO_MARGEN_COBRANZA_LIB_42', 'RGO_SDINDIRECTO_41', 'RGO_SD_SINLEASING_41', 'RGO_SEGU_VINC_LOG_41', 'RGO_SDMICROEMPRESA_42', 'RGO_MARGEN_OPER_ACUM', 'NU_CTA_TDD_SH_43', 'RGO_MARGEN_PREST_VEHIC_41', 'RGO_SDO_VALORES_LOG_41', 'RGO_SDLEA_PEM_LOG_43', 'FH_NACIMIENTO', 'RGO_PASIVO_STD', 'RGO_SDINDIRECTO_43', 'RGO_SDPREST_EMP_DIFF_42', 'RGO_MARGEN_CTS_LOG_41', 'RGO_SDGTIA_HIPOTEC_LOG_41', 'RGO_SDLEA_PEM_SH_41', 'RGO_SDO_VALORES_41', 'RGO_MARGEN_CTE_LOG_41', 'RGO_SDLEA_PEM_LOG_42', 'RGO_SDLEA_PEM_SH_43', 'NU_CTA_AHORRO_SKEW']
X_train = X_train[COLF_FEATURES]

X_test = X_test[COLF_FEATURES]
%%time





NFOLDS = 15

folds = StratifiedKFold(n_splits=NFOLDS)



columns =  X_train.columns #COL #COLF[0:150] #  # COLF[0:85] # X_train.columns

splits = folds.split(X_train, y_train)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X_train.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



model = XGBClassifier(base_score=0.3286505054538241, booster='gbtree',

       colsample_bylevel=1, colsample_bytree=1, gamma=0,

       learning_rate=0.15625549474872114, max_delta_step=0, max_depth= 9,

       min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,

       nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1, tree_method='gpu_hist', predictor='gpu_predictor' )



for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train1, X_valid = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]

    y_train1, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(y_train1.sum(), y_valid.sum(), X_train1.shape, X_valid.shape)

    

    model.fit(X_train1, y_train1,

        eval_set=[(X_train1, y_train1), (X_valid, y_valid)],

        eval_metric='auc',

        verbose=10,

        early_stopping_rounds=20)

    

    feature_importances[f'fold_{fold_n + 1}'] = model.feature_importances_

    

    y_pred_valid = model.predict_proba(X_valid)[:,1]

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += model.predict_proba(X_test[columns])[:,1] / NFOLDS

    

    del X_train1, X_valid, y_train1, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y_train, y_oof)}")

gc.collect()
plt.figure(figsize=(16, 16))

feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(15));





df_stock_submit = pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_STOCK_SUBMIT_SAMPLE.csv')

df_stock_submit.FUGA_3M = y_preds

df_stock_submit.head()



dzpL10_AVG = df_stock_submit.copy()

dzpL10_AVG.to_csv('submit_bbva.csv', index = False)

dzpL10_AVG = pd.read_csv('submit_bbva.csv')

dzpL10_AVG.head()







from IPython.display import HTML





def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submit_bbva.csv')