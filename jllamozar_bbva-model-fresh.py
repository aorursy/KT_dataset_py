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





a = find_corr(df_producto, cc = .98)

print(len(a))

df_producto = df_producto[a]



df_producto.MES = MES_to_X(df_producto.MES)

df_producto.MES_T0 = MES_to_X(df_producto.MES_T0)



IX = intersect1d(df_stock_train.index , df_producto.index)

y_train = df_stock_train.loc[IX].FUGA_3M

print(len(IX), len(y_train))
def mean_dates_3(df, namecol = '_'):

    #########################################

    A = df.loc[(df.MES > df.MES_T0-3) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '21' for c in CC]

    A.columns = CC

    df0 = A

    #######################3333

    A = df.loc[(df.MES<=df.MES_T0-3) & (df.MES>df.MES_T0-5) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '22' for c in CC]

    A.columns = CC

    df1 = A

    #######################3333

    A = df.loc[(df.MES<=df.MES_T0-5) & (df.MES>df.MES_T0-7) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '23' for c in CC]

    A.columns = CC

    df2 = A

    #####################3

    #A = df.loc[(df.MES<=df.MES_T0-4) & (df.MES>df.MES_T0-7) ]

    #A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    #A = A.groupby('ID_CLIENTE').mean()

    #CC = A.columns

    #CC = [c + namecol + '32' for c in CC]

    #A.columns = CC

    #df2 = A

    #######################################################

    A = df.loc[(df.MES>df.MES_T0-10) & (df.MES<=df.MES_T0-7) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '33' for c in CC]

    A.columns = CC

    df3 = A

    #####################################################

    #######################################################

    A = df.loc[(df.MES>df.MES_T0-13) & (df.MES<=df.MES_T0-10) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').mean()

    CC = A.columns

    CC = [c + namecol + '34' for c in CC]

    A.columns = CC

    df4 = A

    #######################################################

    #A = df.loc[(df.MES>df.MES_T0-13) & (df.MES<=df.MES_T0-11) ]

    #A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    #A = A.groupby('ID_CLIENTE').mean()

    #CC = A.columns

    #CC = [c + namecol + '26' for c in CC]

    #A.columns = CC

    #df5 = A

    

    DF = pd.concat([df0, df1, df2, df3, df4], axis =  1)

    #DF.fillna(0, inplace = True)

    del df4, df1, df2, df3, A, df0

    return DF



def mean_test_3(df0, IX, y_train, namecol = '_', pv0 = 1e-5 ):



    df = mean_dates_3(df0, namecol = namecol)

    

    names, pv, st = KVS_TEST(df.loc[IX], y_train , pv0 = pv0)

    u = find_corr(df[names], cc= .9)

    print( len(u) )

    df = df[u]

    return df 

def mean_dates_3_max(df, namecol = '_'):

    #########################################

    A = df.loc[(df.MES > df.MES_T0-3) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').max()

    CC = A.columns

    CC = [c + namecol + '21' for c in CC]

    A.columns = CC

    df0 = A

    #######################3333

    A = df.loc[(df.MES<=df.MES_T0-3) & (df.MES>df.MES_T0-5) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').max()

    CC = A.columns

    CC = [c + namecol + '22' for c in CC]

    A.columns = CC

    df1 = A

    #######################3333

    A = df.loc[(df.MES<=df.MES_T0-5) & (df.MES>df.MES_T0-7) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').max()

    CC = A.columns

    CC = [c + namecol + '23' for c in CC]

    A.columns = CC

    df2 = A

    #####################3

    #A = df.loc[(df.MES<=df.MES_T0-4) & (df.MES>df.MES_T0-7) ]

    #A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    #A = A.groupby('ID_CLIENTE').mean()

    #CC = A.columns

    #CC = [c + namecol + '32' for c in CC]

    #A.columns = CC

    #df2 = A

    #######################################################

    A = df.loc[(df.MES>df.MES_T0-10) & (df.MES<=df.MES_T0-7) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').max()

    CC = A.columns

    CC = [c + namecol + '33' for c in CC]

    A.columns = CC

    df3 = A

    #####################################################

    #######################################################

    A = df.loc[(df.MES>df.MES_T0-13) & (df.MES<=df.MES_T0-10) ]

    A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    A = A.groupby('ID_CLIENTE').max()

    CC = A.columns

    CC = [c + namecol + '34' for c in CC]

    A.columns = CC

    df4 = A

    #######################################################

    #A = df.loc[(df.MES>df.MES_T0-13) & (df.MES<=df.MES_T0-11) ]

    #A.drop(columns = ['MES', 'MES_T0'], inplace = True)

    #A = A.groupby('ID_CLIENTE').mean()

    #CC = A.columns

    #CC = [c + namecol + '26' for c in CC]

    #A.columns = CC

    #df5 = A

    

    DF = pd.concat([df0, df1, df2, df3, df4], axis =  1)

    #DF.fillna(0, inplace = True)

    del df4, df1, df2, df3, A, df0

    return DF



def mean_test_3_max(df0, IX, y_train, namecol = '_', pv0 = 1e-5 ):



    df = mean_dates_3_max(df0, namecol = namecol)

    

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


df1 = df_producto.reset_index()

df = []

j = 1

for c in df1.drop(["ID_CLIENTE", "MES", "MES_T0"], axis=1).columns:

    print("haciendo", c, j)

    temp = pd.crosstab(df1.ID_CLIENTE, df1[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    j = j+1

    df.append(temp)

    #df.append(temp.apply(lambda x: x / x.sum(), axis=1))

df = pd.concat(df, axis=1)

del df1

df.head()
IX = intersect1d(df_stock_train.index , df.index)

y_train = df_stock_train.FUGA_3M

print(len(IX), df_producto.shape, df_stock_train.shape)

from sklearn.feature_selection import  chi2

a, b = chi2(df.loc[IX], y_train.loc[IX])

scatter(log10(b), log10(a), alpha = .5, s = 10), grid()

sns.jointplot(x = log10(b +1e-300), y = log10(a + 1e-300), kind="hex")



COLD = df.columns[b<1e-3]

print(len(COLD))
DF_PRODUCTO_COUNT = df[COLD]

del df 

gc.collect()

DF_PRODUCTO_COUNT.head()
IX = intersect1d(df_stock_train.index , dz.reset_index().ID_CLIENTE )

y_train = df_stock_train.loc[IX].FUGA_3M



dz = mean_dates_3(df_producto)

print(dz.shape, df_producto.shape)



dz.head()
names, PV, ST = KVS_TEST(dz.loc[IX], y_train.loc[IX], pv0 = 1e-5)

print(len(names), dz.shape)

scatter(log10(array(PV)) , ( array(ST)) , alpha = .5, s = 10), grid()

sns.jointplot(x = log10( array(PV)+1e-300), y = ( array(ST)   ), kind="hex")
%%time

dfA = mean_test_3(df_producto, IX, y_train, namecol = '_', pv0 = 1e-3 )

dfD = mean_test_3(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_DIFF_', pv0 = 1e-3)

dfSH = mean_test_3(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_SH_', pv0 = 1e-3 )

dfLOG = mean_test_3( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_LOG_', pv0 = 1e-3 )





dfA_MX = mean_test_3_max(df_producto, IX, y_train, namecol = '_MAX_', pv0 = 1e-3 )

dfD_MX = mean_test_3_max(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_MAX_DIFF_', pv0 = 1e-3)

dfSH_MX = mean_test_3_max(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_MAX_SH_', pv0 = 1e-3 )

dfLOG_MX= mean_test_3_max( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_MAX_LOG_', pv0 = 1e-3 )







del  dff1, dff_sh, dflog, dz

DF_PRODUCTO_MX = pd.concat([dfA_MX, dfD_MX, dfSH_MX, dfLOG_MX], axis = 1)

del dfA_MX, dfD_MX, dfSH_MX, dfLOG_MX

DF_PRODUCTO_MX.head()
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
DF_PRODUCTO = pd.concat([DF,DF_PRODUCTO_MX, dfST, DF_PRODUCTO_COUNT], axis = 1)

del dfST, df, df_producto, DF, DF_PRODUCTO_MX, DF_PRODUCTO_COUNT

DF_PRODUCTO.head()
# THE BEST 600 FEATURES

COLF = ['RGO_PASIVO_MAX_21', 'RGO_SDO_VALORES_STD', 'RGO_ACTIVO_MAX_21', 'RGO_PASIVO_MAX_23', 'RGO_PASIVO_LOG_21', 'RGO_ACTIVO_LOG_22', 'RGO_PASIVO_LOG_23', 'NU_CTA_TDD_1', 'RGO_SEGU_NO_VINC_STD', 'NU_CTA_TDD_0', 'RGO_MARGEN_TDC_PNATURAL_10', 'RGO_SDO_PREST_HIP_11', 'RGO_ACTIVO_22', 'RGO_MARGEN_COBRANZA_LIB_MAX_21', 'RGO_ACTIVO_MAX_22', 'RGO_SDO_TDC_PNATURAL_20', 'RGO_MARGEN_CONFIRMING_MAX_21', 'RGO_MARGEN_STANDBY_MAX_21', 'RGO_SEGU_NO_VINC_LOG_33', 'NU_CTA_NOMINA_1', 'RGO_MARGEN_TDC_PNATURAL_19', 'RGO_SEGU_NO_VINC_21', 'NU_CTA_CTS_1', 'NU_CTA_NOMINA_STD', 'RGO_ACTIVO_LOG_21', 'NU_CTA_TDC_PNATURAL_1', 'NU_CTA_NOMINA_0', 'NU_CTA_NOMINA_23', 'RGO_SEGU_VINC_STD', 'RGO_SEGU_NO_VINC_MAX_21', 'RGO_SDO_TDC_PNATURAL_11', 'NU_CTA_NOMINA_SH_22', 'RGO_SDO_TDC_PJURIDICA_9', 'RGO_SEGU_VINC_LOG_21', 'RGO_SEGU_VINC_MAX_21', 'RGO_SEGU_NO_VINC_MAX_33', 'RGO_MARGEN_CTE_33', 'RGO_PASIVO_21', 'RGO_PASIVO_MAX_22', 'NU_CTA_TDC_PNATURAL_0', 'RGO_MARGEN_CONFIRMING_LOG_22', 'RGO_SDO_PREST_COM_MAX_SH_21', 'RGO_SDO_VALORES_6', 'RGO_MARGEN_TDC_PNATURAL_20', 'RGO_MARGEN_PREST_CONS_MAX_SH_34', 'NU_CTA_VALORES_0', 'RGO_PASIVO_23', 'RGO_SDO_VALORES_13', 'RGO_SDO_FMUTUO_MAX_SH_21', 'RGO_MARGEN_PREST_COM_3', 'RGO_PASIVO_LOG_22', 'RGO_MARGEN_AVAL_8', 'RGO_MARGEN_AVAL_11', 'RGO_MARGEN_COBRANZA_LIB_21', 'NU_CTA_TDD_STD', 'NU_CTA_PREST_HIP_0', 'NU_CTA_CTE_0', 'RGO_ACTIVO_STD', 'RGO_SDO_DESC_LETRA_MAX_22', 'RGO_PASIVO_20', 'RGO_SDO_TDC_PJURIDICA_4', 'RGO_MARGEN_TDC_PNATURAL_LOG_22', 'RGO_MARGEN_CTE_13', 'RGO_MARGEN_AHORRO_20', 'RGO_MARGEN_AHORRO_MAX_33', 'RGO_SDO_COMEX_6', 'RGO_SEGU_NO_VINC_LOG_22', 'RGO_MARGEN_AHORRO_9', 'RGO_SDO_CTE_LOG_22', 'RGO_PASIVO_22', 'RGO_MARGEN_CTE_LOG_21', 'RGO_PASIVO_19', 'RGO_MARGEN_CTE_17', 'RGO_SEGU_NO_VINC_12', 'RGO_MARGEN_STANDBY_8', 'RGO_PASIVO_6', 'RGO_SDO_TDC_PNATURAL_MAX_21', 'RGO_SEGU_NO_VINC_16', 'RGO_SEGU_NO_VINC_LOG_21', 'RGO_MARGEN_PREST_CONS_20', 'RGO_MARGEN_COBRANZA_LIB_15', 'RGO_SEGU_NO_VINC_MAX_SH_33', 'RGO_ACTIVO_20', 'RGO_MARGEN_LSNG_1', 'RGO_SDO_LSNG_MAX_21', 'NU_CTA_CTS_0', 'RGO_MARGEN_TDC_PNATURAL_MAX_33', 'RGO_SDO_TDC_PNATURAL_21', 'RGO_SDO_PREST_COM_8', 'RGO_SEGU_VINC_17', 'RGO_MARGEN_COBRANZA_LIB_11', 'RGO_MARGEN_CTE_MAX_22', 'RGO_SDO_CTS_1', 'RGO_SDO_LSNG_7', 'RGO_MARGEN_PREST_CONS_6', 'RGO_PASIVO_MAX_DIFF_22', 'RGO_MARGEN_VALORES_1', 'RGO_SEGU_NO_VINC_10', 'RGO_MARGEN_AVAL_MAX_SH_22', 'RGO_SEGU_NO_VINC_13', 'RGO_MARGEN_CTS_18', 'RGO_SDO_COMEX_20', 'RGO_MARGEN_PREST_COM_20', 'NU_CTA_NOMINA_LOG_22', 'RGO_SDO_PREST_VEHIC_MAX_SH_22', 'RGO_MARGEN_COMEX_MAX_21', 'RGO_MARGEN_AVAL_MAX_33', 'RGO_MARGEN_AHORRO_MAX_21', 'RGO_ACTIVO_MAX_33', 'RGO_SDO_TDC_PJURIDICA_MAX_DIFF_21', 'NU_CTA_NOMINA_SKEW', 'RGO_MARGEN_COMEX_1', 'RGO_SEGU_NO_VINC_9', 'RGO_MARGEN_COBRANZA_LIB_DIFF_33', 'NU_CTA_NOMINA_SH_33', 'RGO_MARGEN_COMEX_20', 'RGO_MARGEN_CTE_MAX_33', 'NU_CTA_NOMINA_34', 'RGO_MARGEN_CTS_17', 'RGO_MARGEN_CARTERA_6', 'RGO_SEGU_VINC_1', 'RGO_MARGEN_STANDBY_14', 'RGO_SDO_CTS_LOG_33', 'RGO_SDO_VALORES_LOG_21', 'RGO_MARGEN_STANDBY_LOG_33', 'RGO_MARGEN_STANDBY_6', 'RGO_MARGEN_COBRANZA_LIB_STD', 'RGO_PASIVO_SH_33', 'RGO_SDO_AVAL_MAX_21', 'RGO_MARGEN_TDC_PNATURAL_16', 'RGO_SDO_PREST_CONS_MAX_33', 'RGO_SDO_LSNG_MAX_33', 'RGO_MARGEN_TDC_PJURIDICA_4', 'RGO_ACTIVO_MAX_34', 'RGO_SEGU_VINC_34', 'RGO_MARGEN_STANDBY_12', 'RGO_MARGEN_AVAL_MAX_SH_21', 'RGO_MARGEN_COBRANZA_LIB_SH_33', 'RGO_MARGEN_VALORES_MAX_SH_23', 'RGO_MARGEN_PREST_HIP_13', 'RGO_SDO_CTE_MAX_23', 'RGO_SDO_AVAL_MAX_DIFF_21', 'RGO_SDO_LSNG_SH_21', 'RGO_SDO_CTE_MAX_SH_33', 'RGO_MARGEN_TDC_PNATURAL_MAX_SH_34', 'RGO_MARGEN_AHORRO_MAX_SH_34', 'RGO_MARGEN_COBRANZA_GAR_17', 'RGO_SDO_COMEX_14', 'RGO_MARGEN_PREST_CONS_9', 'RGO_SDO_TDC_PNATURAL_LOG_33', 'RGO_MARGEN_CTS_7', 'NU_CTA_NOMINA_LOG_33', 'RGO_SEGU_NO_VINC_MAX_34', 'RGO_PASIVO_7', 'NU_CTA_PREST_CONS_1', 'RGO_SEGU_NO_VINC_2', 'RGO_MARGEN_CONFIRMING_7', 'NU_CTA_TDC_PNATURAL_MAX_LOG_34', 'RGO_SEGU_VINC_3', 'RGO_MARGEN_PREST_CONS_2', 'RGO_SDO_PREST_CONS_6', 'RGO_MARGEN_PZO_13', 'RGO_SDO_PREST_CONS_MAX_SH_33', 'RGO_SEGU_NO_VINC_MAX_SH_34', 'RGO_MARGEN_COBRANZA_LIB_LOG_23', 'NU_CTA_NOMINA_DIFF_22', 'RGO_MARGEN_CTS_LOG_21', 'RGO_ACTIVO_MAX_SH_34', 'RGO_MARGEN_COBRANZA_GAR_MAX_DIFF_34', 'RGO_SDO_PREST_CONS_MAX_SH_21', 'RGO_SEGU_NO_VINC_MAX_DIFF_22', 'NU_CTA_PZO_0', 'RGO_MARGEN_AHORRO_MAX_SH_21', 'RGO_SDO_TDC_PJURIDICA_MAX_DIFF_33', 'NU_CTA_DOMI_0', 'RGO_MARGEN_TDC_PNATURAL_14', 'RGO_MARGEN_STANDBY_LOG_22', 'NU_CTA_AHORRO_MAX_SH_33', 'RGO_MARGEN_CONFIRMING_MAX_22', 'RGO_MARGEN_PREST_COM_MAX_SH_22', 'RGO_SDO_PREST_COM_22', 'RGO_SDO_LSNG_MAX_DIFF_21', 'RGO_SEGU_NO_VINC_MAX_SH_22', 'RGO_MARGEN_CONFIRMING_MAX_23', 'RGO_SDO_FMUTUO_MAX_22', 'RGO_SDO_COMEX_MAX_SH_23', 'RGO_MARGEN_CONFIRMING_LOG_23', 'RGO_SEGU_VINC_MAX_SH_33', 'RGO_MARGEN_CTE_MAX_LOG_21', 'RGO_MARGEN_COMEX_18', 'RGO_MARGEN_STANDBY_LOG_23', 'RGO_MARGEN_CONFIRMING_LOG_21', 'RGO_MARGEN_PREST_COM_9', 'RGO_MARGEN_LSNG_MAX_21', 'RGO_SDO_LSNG_22', 'RGO_MARGEN_COBRANZA_LIB_17', 'RGO_MARGEN_TDC_PNATURAL_LOG_23', 'RGO_MARGEN_CTE_DIFF_22', 'RGO_MARGEN_CARTERA_MAX_22', 'RGO_MARGEN_AVAL_7', 'RGO_SDO_PREST_CONS_MAX_SH_23', 'RGO_MARGEN_STANDBY_10', 'RGO_MARGEN_CTE_MAX_SH_22', 'RGO_MARGEN_STANDBY_SH_23', 'RGO_MARGEN_VALORES_MAX_22', 'RGO_ACTIVO_14', 'RGO_SDO_FMUTUO_LOG_33', 'RGO_SDO_AVAL_MAX_SH_34', 'RGO_MARGEN_LSNG_12', 'RGO_SDO_TDC_PNATURAL_MAX_33', 'RGO_MARGEN_AHORRO_DIFF_33', 'RGO_MARGEN_PREST_HIP_MAX_33', 'RGO_MARGEN_STANDBY_LOG_21', 'RGO_SEGU_NO_VINC_MAX_23', 'RGO_SEGU_NO_VINC_MAX_22', 'RGO_SDO_TDC_PNATURAL_LOG_22', 'RGO_MARGEN_LSNG_MAX_34', 'RGO_ACTIVO_MAX_SH_22', 'RGO_ACTIVO_33', 'RGO_MARGEN_TDC_PJURIDICA_20', 'RGO_MARGEN_STANDBY_MAX_LOG_22', 'RGO_MARGEN_CTE_LOG_33', 'RGO_MARGEN_AHORRO_LOG_21', 'RGO_MARGEN_CTE_21', 'RGO_MARGEN_AVAL_MAX_DIFF_33', 'RGO_SDO_PREST_VEHIC_STD', 'RGO_SDO_PREST_VEHIC_MAX_DIFF_21', 'RGO_MARGEN_VALORES_3', 'RGO_SDO_LSNG_MAX_22', 'NU_CTA_NOMINA_KURT', 'RGO_MARGEN_COMEX_MAX_33', 'RGO_SEGU_VINC_MAX_SH_34', 'RGO_SDO_TDC_PNATURAL_6', 'RGO_MARGEN_TDC_PJURIDICA_MAX_DIFF_22', 'RGO_MARGEN_LSNG_11', 'RGO_SEGU_VINC_LOG_33', 'RGO_MARGEN_PREST_CONS_MAX_21', 'RGO_SDO_LSNG_LOG_21', 'RGO_MARGEN_CTE_LOG_22', 'RGO_SDO_PREST_VEHIC_14', 'RGO_MARGEN_VALORES_9', 'RGO_SEGU_VINC_MAX_22', 'RGO_MARGEN_COBRANZA_GAR_SH_23', 'NU_CTA_TDD_SKEW', 'NU_CTA_TDD_KURT', 'RGO_SDO_PZO_7', 'RGO_MARGEN_CONFIRMING_LOG_33', 'RGO_MARGEN_COBRANZA_LIB_MAX_SH_22', 'RGO_SDO_DESC_LETRA_MAX_SH_23', 'RGO_MARGEN_VALORES_MAX_SH_34', 'NU_CTA_NOMINA_DIFF_21', 'RGO_SDO_COMEX_SH_22', 'RGO_SDO_PREST_HIP_14', 'RGO_SDO_PREST_COM_MAX_21', 'RGO_SDO_AVAL_15', 'RGO_PASIVO_STD', 'RGO_MARGEN_LSNG_LOG_33', 'RGO_SDO_FMUTUO_MAX_33', 'RGO_MARGEN_LSNG_MAX_SH_21', 'RGO_SDO_TDC_PNATURAL_MAX_SH_33', 'RGO_MARGEN_PREST_VEHIC_MAX_22', 'RGO_SDO_VALORES_19', 'RGO_MARGEN_AVAL_SH_34', 'RGO_SDO_PREST_VEHIC_MAX_SH_21', 'RGO_SDO_PREST_COM_20', 'RGO_SDO_LSNG_21', 'RGO_SEGU_VINC_DIFF_34', 'RGO_MARGEN_TDC_PNATURAL_7', 'RGO_SDO_PREST_VEHIC_LOG_33', 'RGO_SDO_PREST_HIP_SH_34', 'RGO_SDO_FMUTUO_MAX_SH_34', 'RGO_PASIVO_LOG_33', 'RGO_MARGEN_LSNG_MAX_33', 'RGO_ACTIVO_MAX_SH_23', 'RGO_SDO_PREST_CONS_MAX_SH_34', 'RGO_MARGEN_LSNG_MAX_SH_34', 'RGO_SDO_LSNG_MAX_SH_22', 'RGO_MARGEN_PREST_HIP_MAX_SH_34', 'RGO_SDO_VALORES_MAX_22', 'RGO_SDO_DESC_LETRA_13', 'RGO_MARGEN_AHORRO_16', 'RGO_MARGEN_TDC_PNATURAL_LOG_33', 'RGO_SDO_CTE_KURT', 'RGO_MARGEN_CONFIRMING_12', 'RGO_SDO_VALORES_LOG_22', 'RGO_SDO_PREST_HIP_MAX_SH_34', 'RGO_SDO_TDC_PJURIDICA_MAX_SH_21', 'RGO_SDO_TDC_PJURIDICA_MAX_SH_23', 'RGO_MARGEN_AHORRO_MAX_SH_22', 'RGO_MARGEN_CONFIRMING_18', 'RGO_MARGEN_AVAL_MAX_23', 'RGO_ACTIVO_MAX_SH_33', 'RGO_MARGEN_PREST_HIP_MAX_SH_23', 'RGO_MARGEN_CARTERA_LOG_21', 'RGO_MARGEN_PREST_HIP_SH_34', 'RGO_MARGEN_COBRANZA_GAR_20', 'RGO_MARGEN_AVAL_MAX_SH_34', 'RGO_SDO_VALORES_MAX_SH_34', 'RGO_MARGEN_COBRANZA_LIB_LOG_33', 'RGO_MARGEN_COBRANZA_GAR_LOG_33', 'RGO_MARGEN_CTE_MAX_SH_23', 'RGO_SDO_TDC_PJURIDICA_MAX_DIFF_23', 'RGO_MARGEN_PREST_CONS_SH_21', 'RGO_SDO_DESC_LETRA_DIFF_23', 'RGO_MARGEN_CONFIRMING_2', 'RGO_SEGU_VINC_MAX_DIFF_34', 'RGO_SDO_VALORES_LOG_33', 'RGO_SDO_FMUTUO_DIFF_34', 'RGO_SDO_CTE_3', 'RGO_MARGEN_PREST_HIP_DIFF_23', 'RGO_MARGEN_COBRANZA_LIB_LOG_21', 'NU_CTA_TDD_SH_34', 'RGO_SDO_COMEX_10', 'RGO_SDO_CTS_SH_34', 'RGO_MARGEN_CTS_5', 'RGO_MARGEN_TDC_PNATURAL_MAX_21', 'RGO_SEGU_VINC_6', 'RGO_SDO_LSNG_MAX_DIFF_33', 'RGO_MARGEN_PZO_MAX_SH_33', 'RGO_MARGEN_AVAL_SH_22', 'RGO_MARGEN_CONFIRMING_LOG_34', 'RGO_MARGEN_COBRANZA_LIB_3', 'RGO_MARGEN_CARTERA_SH_21', 'RGO_SDO_COMEX_MAX_SH_22', 'RGO_PASIVO_MAX_SH_33', 'RGO_SDO_TDC_PNATURAL_DIFF_21', 'RGO_SDO_CTE_MAX_SH_23', 'RGO_MARGEN_COBRANZA_LIB_MAX_SH_23', 'RGO_SDO_AVAL_MAX_34', 'RGO_SDO_LSNG_MAX_SH_21', 'RGO_SDO_TDC_PNATURAL_MAX_SH_23', 'RGO_MARGEN_COMEX_13', 'RGO_SDO_COMEX_4', 'RGO_SDO_TDC_PNATURAL_SH_33', 'RGO_MARGEN_COMEX_LOG_33', 'RGO_SEGU_NO_VINC_SH_21', 'RGO_MARGEN_COBRANZA_LIB_MAX_SH_34', 'RGO_SDO_PREST_COM_MAX_SH_22', 'RGO_MARGEN_LSNG_MAX_DIFF_33', 'RGO_SDO_AVAL_MAX_SH_23', 'RGO_MARGEN_COMEX_9', 'RGO_MARGEN_TDC_PJURIDICA_MAX_33', 'RGO_SDO_CTE_MAX_DIFF_21', 'RGO_SDO_AVAL_KURT', 'RGO_SDO_PREST_COM_SH_33', 'RGO_MARGEN_COBRANZA_LIB_MAX_SH_33', 'RGO_SDO_DESC_LETRA_15', 'RGO_SDO_TDC_PJURIDICA_19', 'RGO_SDO_PREST_HIP_MAX_DIFF_33', 'RGO_SDO_CTE_DIFF_21', 'RGO_SDO_VALORES_SH_33', 'RGO_MARGEN_COBRANZA_LIB_MAX_33', 'RGO_MARGEN_LSNG_7', 'RGO_SDO_PREST_HIP_MAX_DIFF_34', 'RGO_MARGEN_PREST_HIP_MAX_DIFF_33', 'RGO_MARGEN_CTS_13', 'RGO_SDO_TDC_PNATURAL_MAX_DIFF_34', 'RGO_MARGEN_TDC_PNATURAL_1', 'RGO_MARGEN_PZO_20', 'RGO_MARGEN_LSNG_MAX_DIFF_34', 'RGO_SDO_CTE_11', 'RGO_SDO_AVAL_SH_21', 'RGO_MARGEN_CARTERA_MAX_DIFF_34', 'RGO_MARGEN_CTE_LOG_34', 'RGO_MARGEN_AVAL_MAX_21', 'RGO_SDO_FMUTUO_SH_21', 'RGO_ACTIVO_5', 'RGO_SDO_TDC_PJURIDICA_12', 'RGO_MARGEN_LSNG_3', 'RGO_MARGEN_PREST_CONS_8', 'RGO_SEGU_VINC_LOG_23', 'RGO_MARGEN_TDC_PNATURAL_STD', 'RGO_MARGEN_AHORRO_11', 'RGO_MARGEN_PREST_COM_7', 'RGO_MARGEN_VALORES_MAX_21', 'RGO_MARGEN_TDC_PJURIDICA_MAX_SH_22', 'RGO_ACTIVO_23', 'RGO_SDO_PREST_COM_MAX_SH_34', 'NU_CTA_AHORRO_SH_34', 'RGO_ACTIVO_18', 'RGO_MARGEN_LSNG_6', 'RGO_MARGEN_PREST_COM_33', 'RGO_SDO_DESC_LETRA_SH_23', 'RGO_MARGEN_VALORES_SH_21', 'RGO_ACTIVO_MAX_DIFF_21', 'NU_CTA_LSNG_0', 'RGO_MARGEN_VALORES_MAX_SH_22', 'RGO_SEGU_VINC_MAX_SH_21', 'RGO_MARGEN_CARTERA_MAX_DIFF_22', 'RGO_MARGEN_COMEX_LOG_21', 'RGO_SDO_PREST_CONS_SH_34', 'RGO_MARGEN_TDC_PNATURAL_MAX_SH_21', 'RGO_MARGEN_PZO_SH_34', 'RGO_SEGU_VINC_LOG_34', 'RGO_MARGEN_TDC_PJURIDICA_MAX_LOG_21', 'RGO_SDO_AVAL_21', 'RGO_MARGEN_PZO_MAX_34', 'RGO_SDO_PZO_LOG_33', 'RGO_MARGEN_CARTERA_SH_22', 'RGO_MARGEN_PZO_LOG_21', 'RGO_SDO_AVAL_MAX_DIFF_34', 'RGO_SDO_FMUTUO_MAX_DIFF_22', 'RGO_MARGEN_CTS_LOG_22', 'RGO_MARGEN_PREST_VEHIC_6', 'RGO_MARGEN_CARTERA_STD', 'RGO_MARGEN_PREST_VEHIC_DIFF_33', 'RGO_MARGEN_PREST_CONS_MAX_SH_22', 'RGO_SDO_VALORES_MAX_23', 'RGO_SDO_DESC_LETRA_MAX_SH_33', 'RGO_MARGEN_COBRANZA_GAR_SH_33', 'RGO_SDO_PREST_COM_MAX_SH_23', 'RGO_SDO_TDC_PNATURAL_LOG_34', 'RGO_MARGEN_VALORES_MAX_DIFF_22', 'RGO_MARGEN_PREST_CONS_MAX_DIFF_22', 'NU_CTA_TDC_PJURIDICA_0', 'RGO_MARGEN_STANDBY_5', 'RGO_MARGEN_TDC_PNATURAL_SH_21', 'RGO_SDO_LSNG_19', 'RGO_SDO_COMEX_SH_23', 'RGO_MARGEN_VALORES_7', 'RGO_SEGU_VINC_MAX_DIFF_22', 'RGO_SDO_CTS_MAX_DIFF_34', 'RGO_MARGEN_LSNG_LOG_23', 'RGO_SDO_LSNG_LOG_34', 'RGO_SDO_LSNG_SH_33', 'RGO_MARGEN_PZO_MAX_SH_22', 'RGO_SEGU_NO_VINC_MAX_SH_23', 'RGO_MARGEN_TDC_PJURIDICA_MAX_DIFF_34', 'RGO_SDO_COMEX_MAX_23', 'RGO_SDO_VALORES_SH_21', 'NU_CTA_TDD_3', 'RGO_ACTIVO_DIFF_21', 'RGO_MARGEN_AVAL_16', 'RGO_SDO_PREST_VEHIC_15', 'RGO_PASIVO_3', 'RGO_MARGEN_LSNG_DIFF_34', 'RGO_SEGU_VINC_SH_21', 'RGO_SDO_DESC_LETRA_LOG_21', 'RGO_MARGEN_STANDBY_STD', 'RGO_MARGEN_CARTERA_LOG_22', 'RGO_SEGU_NO_VINC_18', 'RGO_SEGU_NO_VINC_SH_23', 'RGO_SDO_PZO_MAX_21', 'RGO_MARGEN_PREST_VEHIC_MAX_33', 'RGO_SDO_PREST_HIP_MAX_SH_23', 'RGO_MARGEN_CTE_SH_22', 'RGO_MARGEN_COBRANZA_LIB_MAX_34', 'RGO_MARGEN_TDC_PJURIDICA_34', 'RGO_SDO_VALORES_MAX_DIFF_33', 'RGO_PASIVO_KURT', 'RGO_SDO_DESC_LETRA_LOG_22', 'RGO_MARGEN_AVAL_SH_21', 'RGO_MARGEN_CARTERA_MAX_DIFF_21', 'RGO_SDO_TDC_PJURIDICA_MAX_SH_22', 'RGO_ACTIVO_SH_22', 'RGO_MARGEN_AVAL_MAX_SH_23', 'RGO_SDO_PREST_HIP_9', 'NU_CTA_TDD_SH_33', 'RGO_SDO_TDC_PNATURAL_MAX_DIFF_33', 'RGO_MARGEN_PREST_CONS_LOG_33', 'RGO_SDO_TDC_PNATURAL_3', 'RGO_SDO_CTS_DIFF_33', 'RGO_MARGEN_STANDBY_SH_34', 'RGO_SDO_PREST_VEHIC_SH_22', 'RGO_SDO_PREST_COM_SH_23', 'RGO_SDO_AVAL_SKEW', 'RGO_SDO_PREST_VEHIC_MAX_SH_34', 'RGO_MARGEN_AVAL_MAX_34', 'RGO_SDO_LSNG_MAX_DIFF_22', 'RGO_SDO_DESC_LETRA_SH_33', 'RGO_PASIVO_MAX_DIFF_34', 'RGO_SDO_TDC_PJURIDICA_8', 'RGO_MARGEN_CTS_12', 'RGO_SDO_LSNG_STD', 'RGO_MARGEN_COMEX_MAX_SH_34', 'RGO_SDO_FMUTUO_MAX_DIFF_34', 'RGO_MARGEN_LSNG_SH_33', 'RGO_MARGEN_CTS_MAX_22', 'RGO_MARGEN_PREST_CONS_KURT', 'RGO_SDO_CTS_MAX_DIFF_23', 'RGO_SDO_PREST_CONS_11', 'NU_CTA_PREST_VEHIC_0', 'RGO_SDO_TDC_PJURIDICA_MAX_33', 'RGO_MARGEN_CTE_MAX_SH_33', 'RGO_SDO_PREST_HIP_KURT', 'RGO_MARGEN_CTE_KURT', 'RGO_MARGEN_PREST_CONS_SH_34', 'RGO_MARGEN_COBRANZA_LIB_MAX_SH_21', 'RGO_SDO_FMUTUO_LOG_21', 'RGO_MARGEN_PZO_MAX_DIFF_33', 'RGO_SDO_COMEX_MAX_21', 'RGO_MARGEN_TDC_PNATURAL_18', 'RGO_SDO_CTS_STD', 'RGO_SDO_VALORES_LOG_34', 'RGO_SEGU_NO_VINC_MAX_DIFF_21', 'RGO_MARGEN_LSNG_MAX_23', 'RGO_MARGEN_STANDBY_MAX_33', 'RGO_MARGEN_STANDBY_MAX_DIFF_34', 'RGO_MARGEN_AHORRO_SH_33', 'RGO_MARGEN_AVAL_34', 'RGO_SDO_TDC_PJURIDICA_MAX_21', 'RGO_SDO_PREST_VEHIC_SH_21', 'RGO_SDO_VALORES_MAX_DIFF_23', 'RGO_MARGEN_COBRANZA_LIB_DIFF_21', 'RGO_MARGEN_TDC_PNATURAL_MAX_SH_23', 'RGO_MARGEN_VALORES_13', 'RGO_SDO_PREST_COM_LOG_22', 'RGO_MARGEN_COMEX_LOG_22', 'RGO_MARGEN_TDC_PJURIDICA_MAX_34', 'RGO_MARGEN_COBRANZA_LIB_33', 'RGO_MARGEN_CARTERA_DIFF_22', 'RGO_MARGEN_COMEX_KURT', 'RGO_SDO_TDC_PNATURAL_STD', 'RGO_MARGEN_AVAL_STD', 'RGO_MARGEN_TDC_PNATURAL_MAX_23', 'RGO_MARGEN_PREST_HIP_MAX_DIFF_34', 'RGO_MARGEN_STANDBY_MAX_LOG_23', 'RGO_ACTIVO_SH_21', 'RGO_MARGEN_PZO_SH_21', 'RGO_SDO_COMEX_KURT', 'RGO_MARGEN_AVAL_LOG_33', 'RGO_MARGEN_PREST_COM_LOG_21', 'RGO_MARGEN_LSNG_MAX_SH_33', 'RGO_MARGEN_CONFIRMING_MAX_SH_21', 'RGO_MARGEN_PREST_COM_SKEW', 'RGO_MARGEN_PREST_VEHIC_SH_23', 'RGO_MARGEN_CONFIRMING_MAX_SH_22', 'RGO_SDO_COMEX_DIFF_22', 'RGO_PASIVO_16', 'RGO_MARGEN_PZO_LOG_34', 'RGO_MARGEN_CTS_MAX_SH_22', 'RGO_SDO_DESC_LETRA_6', 'RGO_MARGEN_PREST_VEHIC_LOG_22', 'RGO_SDO_LSNG_12', 'RGO_SDO_CTS_DIFF_21', 'RGO_SDO_PREST_VEHIC_MAX_DIFF_23', 'RGO_SDO_VALORES_LOG_23', 'RGO_SDO_PREST_HIP_3', 'RGO_SDO_DESC_LETRA_MAX_SH_34', 'RGO_MARGEN_TDC_PJURIDICA_MAX_SH_33', 'RGO_MARGEN_VALORES_LOG_21', 'RGO_MARGEN_PREST_COM_LOG_33', 'RGO_MARGEN_CTS_19', 'RGO_MARGEN_COBRANZA_LIB_MAX_23', 'RGO_MARGEN_CONFIRMING_MAX_33', 'RGO_SDO_COMEX_LOG_22', 'RGO_SDO_VALORES_1', 'RGO_MARGEN_CARTERA_LOG_33', 'RGO_SDO_PREST_CONS_SH_33', 'RGO_MARGEN_PREST_CONS_SH_33', 'RGO_MARGEN_LSNG_14', 'RGO_SDO_LSNG_MAX_SH_34', 'RGO_SDO_CTE_SH_23', 'RGO_SDO_TDC_PNATURAL_SH_22', 'RGO_MARGEN_AVAL_LOG_22', 'RGO_SEGU_NO_VINC_LOG_23', 'RGO_MARGEN_PZO_MAX_DIFF_34', 'RGO_ACTIVO_LOG_33', 'RGO_MARGEN_PREST_HIP_MAX_DIFF_23', 'RGO_SDO_PREST_HIP_MAX_SH_21', 'RGO_MARGEN_AHORRO_MAX_23', 'RGO_MARGEN_CTS_LOG_33', 'RGO_SDO_DESC_LETRA_2', 'RGO_MARGEN_COBRANZA_GAR_STD', 'RGO_MARGEN_COMEX_MAX_SH_33', 'RGO_SDO_CTE_MAX_DIFF_34', 'RGO_SDO_DESC_LETRA_STD', 'RGO_MARGEN_PZO_MAX_DIFF_22', 'RGO_MARGEN_PREST_CONS_MAX_SH_21', 'RGO_MARGEN_COBRANZA_GAR_1', 'RGO_SDO_TDC_PNATURAL_SH_21', 'RGO_MARGEN_TDC_PJURIDICA_MAX_SH_34', 'RGO_SDO_COMEX_LOG_33', 'RGO_SDO_VALORES_MAX_DIFF_21', 'RGO_SEGU_NO_VINC_17', 'RGO_MARGEN_COBRANZA_LIB_MAX_DIFF_21', 'RGO_SDO_FMUTUO_SKEW', 'RGO_MARGEN_VALORES_19', 'RGO_MARGEN_CTS_SH_33', 'RGO_SDO_FMUTUO_KURT', 'RGO_SDO_VALORES_MAX_34', 'RGO_SDO_DESC_LETRA_MAX_21', 'RGO_SDO_TDC_PJURIDICA_SH_34', 'RGO_SEGU_NO_VINC_LOG_34', 'RGO_MARGEN_PREST_COM_8', 'RGO_ACTIVO_34', 'RGO_SDO_TDC_PNATURAL_SKEW', 'RGO_MARGEN_CTE_20', 'RGO_MARGEN_TDC_PNATURAL_33', 'RGO_SDO_DESC_LETRA_LOG_23', 'RGO_MARGEN_AVAL_LOG_21', 'RGO_SDO_AVAL_MAX_SH_21', 'RGO_MARGEN_TDC_PJURIDICA_SH_23', 'RGO_SEGU_VINC_SH_34', 'RGO_SDO_TDC_PJURIDICA_SH_21', 'RGO_MARGEN_TDC_PJURIDICA_SH_21', 'RGO_MARGEN_LSNG_LOG_22', 'RGO_SDO_TDC_PNATURAL_MAX_SH_34', 'RGO_MARGEN_PREST_VEHIC_LOG_33', 'RGO_SEGU_VINC_MAX_33']

print(len(COLF))
DF = DF_PRODUCTO[COLF].copy()

del DF_PRODUCTO

DF_PRODUCTO = DF.copy()

del DF

DF_PRODUCTO.head()
gc.collect()
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
print(len(IX), df.shape, y_train.shape)
dz = mean_dates_3(df.reset_index().set_index('ID_CLIENTE') )

names, PV, ST = KVS_TEST(dz.loc[IX], y_train.loc[IX], pv0 = 1e-5)

print(len(names), dz.shape)

scatter(log10(array(PV)) , ( array(ST)) , alpha = .5, s = 10), grid()

sns.jointplot(x = log10( array(PV)+1e-300), y = ( array(ST)   ), kind="hex")


%%time 

dfA = mean_test_3(df_endeudamiento, IX, y_train, namecol = '_', pv0 = 1e-3 )

dfD = mean_test_3(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_DIFF_', pv0 = 1e-3)

dfSH = mean_test_3(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_SH_', pv0 = 1e-3 )

dfLOG = mean_test_3( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_LOG_', pv0 = 1e-3 )





dfA_MX = mean_test_3_max(df_endeudamiento, IX, y_train, namecol = '_MAX_', pv0 = 1e-3 )

dfD_MX = mean_test_3_max(dff1.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_MAX_DIFF_', pv0 = 1e-3)

dfSH_MX = mean_test_3_max(dff_sh.reset_index().set_index('ID_CLIENTE'), IX, y_train, namecol = '_MAX_SH_', pv0 = 1e-3 )

dfLOG_MX= mean_test_3_max( dflog.reset_index().set_index('ID_CLIENTE') , IX, y_train, namecol = '_MAX_LOG_', pv0 = 1e-3 )





del  dff1, dff_sh, dflog, dz, df



DF_ENDEUDAMIENTO_MX = pd.concat([dfA_MX, dfD_MX, dfSH_MX, dfLOG_MX], axis = 1)

del dfA_MX, dfD_MX, dfSH_MX, dfLOG_MX

DF_ENDEUDAMIENTO_MX.head()
DF = pd.concat([dfA, dfD, dfSH, dfLOG], axis = 1)

del dfA, dfD, dfSH, dfLOG

DF.head()
%%time

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
DF_ENDEUDAMIENTO = pd.concat([DF, DF_ENDEUDAMIENTO_MX, dfST], axis = 1)

del dfST, DF, DF_ENDEUDAMIENTO_MX

DF_ENDEUDAMIENTO.head()
df1 = df_endeudamiento.reset_index()

df1.fillna(-999, inplace = True)

df = []

j = 1

for c in df1.drop(["ID_CLIENTE", "MES", "MES_T0"], axis=1).columns:

    print("haciendo", c, j)

    temp = pd.crosstab(df1.ID_CLIENTE, df1[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    j = j+1

    df.append(temp)

    #df.append(temp.apply(lambda x: x / x.sum(), axis=1))

df = pd.concat(df, axis=1)

del df1

df.head()
IX = intersect1d(df_stock_train.index , df.index)



from sklearn.feature_selection import  chi2

a, b = chi2(df.loc[IX], y_train.loc[IX])

scatter(log10(b), log10(a), alpha = .5, s = 10), grid()

sns.jointplot(x = log10(b +1e-300), y = log10(a + 1e-300), kind="hex")



COLD = df.columns[b<1e-3]

print(len(COLD))
DF_ENDEUDAMIENTO_COUNT= df[COLD]

del df , df_endeudamiento



n = find_corr(DF_ENDEUDAMIENTO_COUNT, cc = .95)

print( len(n) )



DF_ENDEUDAMIENTO_COUNT = DF_ENDEUDAMIENTO_COUNT[n] 

gc.collect()
DF = pd.concat([DF_ENDEUDAMIENTO , DF_ENDEUDAMIENTO_COUNT], axis =1)

del DF_ENDEUDAMIENTO, DF_ENDEUDAMIENTO_COUNT

DF_ENDEUDAMIENTO = DF.copy()

del DF

DF_ENDEUDAMIENTO.head()
IX = intersect1d(df_stock_train.index, DF_ENDEUDAMIENTO.index)

X_train = DF_ENDEUDAMIENTO.loc[IX]

y_train = df_stock_train.loc[IX].FUGA_3M

X_train.head()

# THE BEST 500 FEATURES

COLF = ['RGO_SDPREST_EMP_MAX_21', 'ST_CREDITO_5.0', 'RGO_SDPREST_EMP_MAX_33', 'RGO_SD_NOPREFE_18', 'ST_CREDITO_LOG_33', 'RGO_SDLEA_PEM_LOG_33', 'ST_CREDITO_33', 'RGO_SDMICROEMPRESA_MAX_33', 'RGO_SDPREST_EMP_MAX_34', 'RGO_SDMICROEMPRESA_MAX_34', 'ST_CREDITO_MAX_SH_34', 'RGO_SDLEA_PEM_MAX_SH_34', 'RGO_SDGRANDES_EMP_10', 'RGO_SDACEPBANCA_LOG_33', 'CD_BANCO_1', 'RGO_SD_SINLEASING_MAX_LOG_21', 'RGO_SDMICROEMPRESA_MAX_21', 'RGO_SD_SINLEASING_MAX_LOG_33', 'RGO_SDPREST_EMP_LOG_34', 'ST_CREDITO_21', 'ST_CREDITO_LOG_34', 'RGO_SDCREDITO_MAX_SH_33', 'RGO_SDPREST_EMP_21', 'RGO_SD_SINLEASING_MAX_LOG_22', 'CD_BANCO_34', 'RGO_SDLEA_PEM_6', 'RGO_SDPEQUENIAS_EMP_33', 'RGO_SDLEA_PEM_15', 'RGO_SDLEA_PEM_MAX_SH_33', 'RGO_SDMICROEMPRESA_20', 'RGO_SDLEA_PEM_34', 'RGO_SDFACTORING_9', 'RGO_SDLEA_PEM_MAX_34', 'CD_BANCO_101', 'RGO_SDCREDITO_MAX_33', 'RGO_SDLEA_PEM_LOG_21', 'CD_BANCO_82', 'CD_BANCO_140', 'RGO_SDCREDITO_3', 'RGO_SD_SINLEASING_MAX_LOG_23', 'RGO_SDPEQUENIAS_EMP_MAX_21', 'RGO_SDPREST_EMP_MAX_SH_21', 'CD_BANCO_MAX_34', 'RGO_SDLEA_PEM_MAX_LOG_21', 'RGO_SDCREDITO_MAX_21', 'CD_BANCO_109', 'RGO_SDPREST_EMP_23', 'RGO_SDLEA_PEM_LOG_34', 'CD_BANCO_6', 'RGO_SDGTIA_PREFE_7', 'ST_CREDITO_SH_34', 'RGO_SDLEA_PEM_10', 'RGO_SDMICROEMPRESA_MAX_DIFF_21', 'RGO_SDCREDITO_20', 'RGO_SDGTIA_HIPOTEC_MAX_21', 'RGO_SDPEQUENIAS_EMP_DIFF_33', 'RGO_SDGTIA_HIPOTEC_DIFF_23', 'RGO_SDPEQUENIAS_EMP_21', 'CD_BANCO_228', 'RGO_SDMICROEMPRESA_MAX_SH_21', 'RGO_SDDESCLETRAS_MAX_23', 'RGO_SD_NOPREFE_MAX_LOG_23', 'RGO_SDGTIA_HIPOTEC_MAX_DIFF_23', 'RGO_SDPREST_EMP_33', 'RGO_SDCOMEX_20', 'CD_BANCO_156', 'RGO_SDLEA_PEM_MAX_33', 'RGO_SDGTIA_HIPOTEC_2', 'RGO_SDPEQUENIAS_EMP_LOG_21', 'RGO_SDACEPBANCA_MAX_SH_22', 'RGO_SDMICROEMPRESA_MAX_SH_33', 'RGO_SDPEQUENIAS_EMP_MAX_23', 'RGO_SDINDIRECTO_21', 'RGO_SDMICROEMPRESA_33', 'RGO_SDCARTFIANZA_5', 'ST_CREDITO_MAX_21', 'RGO_SDMICROEMPRESA_16', 'RGO_SDCOMEX_MAX_LOG_33', 'RGO_SD_NOPREFE_7', 'RGO_SDPREST_EMP_MAX_SH_23', 'RGO_SDGTIA_HIPOTEC_20', 'RGO_SDCOMEX_MAX_SH_23', 'RGO_SDCREDITO_MAX_23', 'RGO_SDMEDIANAS_EMP_MAX_22', 'RGO_SD_NOPREFE_21', 'RGO_SDLEA_PEM_DIFF_33', 'RGO_SDMEDIANAS_EMP_MAX_21', 'RGO_SDDESCLETRAS_17', 'RGO_SDDESCLETRAS_11', 'RGO_SDCREDITO_1', 'RGO_SDLEASING_19', 'RGO_SD_NOPREFE_MAX_DIFF_22', 'RGO_SDMEDIANAS_EMP_20', 'RGO_SD_NOPREFE_22', 'CD_BANCO_239', 'RGO_SDDESCLETRAS_1', 'RGO_SDFACTORING_MAX_21', 'RGO_SDPREST_EMP_MAX_23', 'RGO_SD_NOPREFE_SH_21', 'RGO_SDFACTORING_14', 'RGO_SDPEQUENIAS_EMP_MAX_33', 'RGO_SDDESCLETRAS_DIFF_22', 'RGO_SDCARTFIANZA_DIFF_22', 'RGO_SDINDIRECTO_MAX_SH_21', 'RGO_SDLEA_PEM_MAX_DIFF_33', 'CD_BANCO_73', 'RGO_SDPREST_EMP_SH_23', 'RGO_SDCREDITO_LOG_33', 'RGO_SDPEQUENIAS_EMP_MAX_SH_33', 'RGO_SDCOMEX_MAX_SH_33', 'RGO_SDDESCLETRAS_DIFF_21', 'RGO_SDTARJETA_EMP_20', 'RGO_SDACEPBANCA_3', 'RGO_SD_NOPREFE_19', 'RGO_SD_NOPREFE_MAX_21', 'RGO_SDPREST_EMP_8', 'RGO_SDCREDITO_MAX_DIFF_23', 'RGO_SD_SINLEASING_34', 'RGO_SDINDIRECTO_13', 'RGO_SDFACTORING_6', 'RGO_SD_NOPREFE_MAX_22', 'ST_CREDITO_STD', 'RGO_SDINDIRECTO_MAX_SH_23', 'RGO_SDGTIA_HIPOTEC_DIFF_21', 'RGO_SD_NOPREFE_MAX_DIFF_23', 'RGO_SDLEA_PEM_DIFF_22', 'RGO_SDGRANDES_EMP_23', 'RGO_SDLEA_PEM_SH_34', 'RGO_SDINDIRECTO_DIFF_21', 'RGO_SDLEA_PEM_MAX_SH_22', 'RGO_SD_NOPREFE_LOG_21', 'RGO_SDMICROEMPRESA_DIFF_22', 'RGO_SDCREDITO_DIFF_22', 'RGO_SDMEDIANAS_EMP_MAX_SH_21', 'RGO_SDGRANDES_EMP_MAX_23', 'RGO_SDTOTAL_LOG_21', 'RGO_SDPREST_EMP_MAX_DIFF_22', 'ST_CREDITO_22', 'CD_BANCO_236', 'RGO_SDGTIA_HIPOTEC_MAX_DIFF_21', 'RGO_SDCARTFIANZA_DIFF_33', 'RGO_SDPEQUENIAS_EMP_9', 'RGO_SDGTIA_PREFE_MAX_33', 'RGO_SDGRANDES_EMP_6', 'RGO_SDMEDIANAS_EMP_DIFF_21', 'RGO_SDACEPBANCA_DIFF_21', 'RGO_SDTOTAL_MAX_SH_22', 'RGO_SDMICROEMPRESA_22', 'RGO_SDMICROEMPRESA_MAX_23', 'RGO_SDGTIA_PREFE_SH_23', 'RGO_SDTOTAL_MAX_SH_33', 'RGO_SD_NOPREFE_MAX_SH_23', 'RGO_SDLEA_PEM_SH_33', 'CD_BANCO_231', 'RGO_SDGTIA_PREFE_MAX_SH_33', 'RGO_SDMEDIANAS_EMP_MAX_DIFF_22', 'RGO_SDLEASING_33', 'ST_CREDITO_34', 'RGO_SDCREDITO_MAX_SH_22', 'RGO_SDGTIA_PREFE_22', 'RGO_SDACEPBANCA_MAX_SH_33', 'RGO_SDINDIRECTO_LOG_21', 'RGO_SDACEPBANCA_MAX_DIFF_33', 'RGO_SDINDIRECTO_MAX_SH_33', 'RGO_SDCREDITO_SH_33', 'RGO_SDMICROEMPRESA_MAX_SH_23', 'RGO_SDMICROEMPRESA_SH_23', 'RGO_SD_SINLEASING_MAX_34', 'RGO_SDCREDITO_33', 'RGO_SDGTIA_HIPOTEC_MAX_LOG_23', 'RGO_SD_NOPREFE_MAX_SH_21', 'RGO_SDMICROEMPRESA_STD', 'RGO_SDDESCLETRAS_LOG_22', 'RGO_SDFACTORING_DIFF_23', 'RGO_SDLEA_PEM_MAX_LOG_23', 'RGO_SDACEPBANCA_SH_22', 'RGO_SDCARTFIANZA_SH_33', 'RGO_SDCREDITO_DIFF_23', 'RGO_SDLEASING_SH_21', 'RGO_SDINDIRECTO_MAX_DIFF_33', 'RGO_SD_NOPREFE_10', 'RGO_SDMICROEMPRESA_LOG_23', 'RGO_SDCREDITO_15', 'RGO_SDCREDITO_SH_23', 'RGO_SDCOMEX_22', 'RGO_SDINDIRECTO_DIFF_33', 'RGO_SDPREST_EMP_MAX_SH_22', 'RGO_SDTARJETA_EMP_MAX_SH_23', 'CD_BANCO_230', 'RGO_SDPREST_EMP_MAX_22', 'RGO_SDMICROEMPRESA_MAX_SH_22', 'RGO_SDPEQUENIAS_EMP_MAX_SH_23', 'RGO_SDDIRECTO_SKEW', 'RGO_SD_NOPREFE_SH_33', 'RGO_SDPREST_EMP_LOG_22', 'RGO_SDLEA_PEM_DIFF_34', 'RGO_SDFACTORING_SH_33', 'RGO_SDGRANDES_EMP_MAX_SH_23', 'RGO_SDPREST_EMP_10', 'RGO_SDDESCLETRAS_MAX_DIFF_21', 'RGO_SDMEDIANAS_EMP_MAX_SH_22', 'RGO_SDPREST_EMP_LOG_21', 'RGO_SDCREDITO_SH_22', 'RGO_SDCOMEX_3', 'RGO_SDCOMEX_SKEW', 'RGO_SD_NOPREFE_LOG_23', 'RGO_SD_SINLEASING_20', 'RGO_SDGRANDES_EMP_SH_21', 'RGO_SDACEPBANCA_DIFF_22', 'RGO_SDFACTORING_MAX_SH_23', 'RGO_SDTARJETA_EMP_SH_33', 'RGO_SDTARJETA_EMP_MAX_23', 'RGO_SDDESCLETRAS_SH_21', 'RGO_SDINDIRECTO_SH_21', 'RGO_SDMICROEMPRESA_12', 'RGO_SDCARTFIANZA_MAX_SH_22', 'RGO_SD_SINLEASING_19', 'RGO_SDCARTFIANZA_14', 'RGO_SDTOTAL_SH_23', 'RGO_SDLEASING_1', 'CD_BANCO_248', 'RGO_SDINDIRECTO_LOG_33', 'RGO_SDFACTORING_DIFF_21', 'RGO_SDCREDITO_34', 'RGO_SDPEQUENIAS_EMP_SH_33', 'RGO_SDDESCLETRAS_LOG_23', 'RGO_SD_NOPREFE_MAX_SH_22', 'RGO_SDLEA_PEM_16', 'RGO_SDLEA_PEM_SKEW', 'RGO_SDACEPBANCA_SH_21', 'RGO_SDGRANDES_EMP_MAX_SH_33', 'RGO_SD_NOPREFE_6', 'RGO_SDINDIRECTO_LOG_22', 'RGO_SD_SINLEASING_SH_23', 'RGO_SDGRANDES_EMP_DIFF_23', 'RGO_SDLEA_PEM_19', 'RGO_SD_SINLEASING_LOG_21', 'RGO_SDGRANDES_EMP_MAX_LOG_21', 'RGO_SDLEA_PEM_LOG_22', 'RGO_SDTOTAL_MAX_SH_23', 'RGO_SDPEQUENIAS_EMP_MAX_DIFF_22', 'RGO_SDGTIA_PREFE_DIFF_33', 'RGO_SDMEDIANAS_EMP_SH_21', 'RGO_SDGTIA_HIPOTEC_5', 'RGO_SDGTIA_HIPOTEC_LOG_33', 'RGO_SDMICROEMPRESA_18', 'RGO_SDLEA_PEM_LOG_23', 'CD_BANCO_174', 'RGO_SDCOMEX_SH_21', 'RGO_SDCARTFIANZA_SH_21', 'RGO_SDLEASING_LOG_22', 'RGO_SDDESCLETRAS_MAX_34', 'RGO_SDGTIA_PREFE_MAX_SH_21', 'RGO_SDDESCLETRAS_SH_22', 'RGO_SD_NOPREFE_SH_23', 'RGO_SDCREDITO_10', 'RGO_SDINDIRECTO_DIFF_23', 'ST_CREDITO_DIFF_21', 'RGO_SD_NOPREFE_DIFF_21', 'RGO_SDMEDIANAS_EMP_33', 'RGO_SDDESCLETRAS_MAX_DIFF_23', 'RGO_SDGRANDES_EMP_MAX_DIFF_21', 'RGO_SDCREDITO_SH_21', 'RGO_SDMICROEMPRESA_DIFF_23', 'RGO_SDCARTFIANZA_STD', 'RGO_SDDESCLETRAS_MAX_SH_21', 'RGO_SDFACTORING_SH_22', 'RGO_SD_NOPREFE_SH_22', 'RGO_SDMICROEMPRESA_SH_22', 'CD_BANCO_237', 'RGO_SDPEQUENIAS_EMP_DIFF_22', 'RGO_SDGTIA_HIPOTEC_MAX_DIFF_33', 'RGO_SDPEQUENIAS_EMP_23', 'RGO_SDTARJETA_EMP_DIFF_21', 'RGO_SDMICROEMPRESA_SH_33', 'RGO_SDACEPBANCA_STD', 'RGO_SDFACTORING_LOG_21', 'RGO_SDACEPBANCA_8', 'RGO_SDLEASING_SH_22', 'RGO_SDINDIRECTO_LOG_23', 'RGO_SDLEASING_DIFF_34', 'RGO_SDDESCLETRAS_16', 'RGO_SDACEPBANCA_MAX_SH_21', 'RGO_SDPREST_EMP_18', 'RGO_SDCARTFIANZA_LOG_33', 'RGO_SDINDIRECTO_DIFF_22', 'RGO_SDDESCLETRAS_SH_23', 'RGO_SDTOTAL_LOG_22', 'RGO_SDPREST_EMP_DIFF_22', 'RGO_SDINDIRECTO_33', 'RGO_SDFACTORING_LOG_34', 'RGO_SDMEDIANAS_EMP_SH_22', 'RGO_SDTARJETA_EMP_SH_23', 'RGO_SDMICROEMPRESA_DIFF_33', 'RGO_SDLEA_PEM_MAX_DIFF_22', 'RGO_SDCOMEX_LOG_22', 'RGO_SDLEA_PEM_SH_22', 'RGO_SDCREDITO_LOG_21', 'RGO_SDPREST_EMP_15', 'RGO_SDDESCLETRAS_LOG_33', 'RGO_SDFACTORING_LOG_22', 'RGO_SDTARJETA_EMP_DIFF_22', 'RGO_SD_NOPREFE_LOG_33', 'RGO_SDDESCLETRAS_MAX_SH_23', 'RGO_SDCARTFIANZA_MAX_SH_33', 'RGO_SDINDIRECTO_SH_22', 'RGO_SDLEA_PEM_MAX_SH_21', 'RGO_SDCARTFIANZA_DIFF_21', 'RGO_SDLEA_PEM_STD', 'RGO_SDPEQUENIAS_EMP_DIFF_23', 'RGO_SDTOTAL_SH_21', 'RGO_SDGTIA_PREFE_23', 'RGO_SDGRANDES_EMP_SH_22', 'RGO_SD_NOPREFE_13', 'RGO_SDCARTFIANZA_LOG_23', 'RGO_SDPREST_EMP_DIFF_21', 'RGO_SDACEPBANCA_7', 'RGO_SD_SINLEASING_STD', 'RGO_SDLEASING_SH_33', 'RGO_SDMICROEMPRESA_MAX_DIFF_33', 'RGO_SDPEQUENIAS_EMP_SH_23', 'RGO_SDCREDITO_MAX_DIFF_21', 'RGO_SDGTIA_PREFE_16', 'RGO_SDTOTAL_LOG_34', 'RGO_SDPREST_EMP_SH_22', 'RGO_SDCREDITO_LOG_22', 'RGO_SDPREST_EMP_DIFF_33', 'RGO_SDFACTORING_SH_23', 'RGO_SDPREST_EMP_SH_33', 'RGO_SDCOMEX_MAX_DIFF_22', 'RGO_SDMICROEMPRESA_SH_21', 'RGO_SDLEASING_LOG_21', 'RGO_SDLEASING_6', 'RGO_SDCOMEX_SH_22', 'RGO_SDGTIA_HIPOTEC_SH_33', 'RGO_SDGTIA_HIPOTEC_MAX_SH_33', 'RGO_SDMICROEMPRESA_13', 'RGO_SDLEA_PEM_14', 'RGO_SDCARTFIANZA_MAX_LOG_23', 'RGO_SDFACTORING_12', 'RGO_SDDESCLETRAS_SH_33', 'RGO_SDACEPBANCA_LOG_21', 'RGO_SDTARJETA_EMP_LOG_21', 'RGO_SDCOMEX_DIFF_21', 'RGO_SDLEA_PEM_7', 'RGO_SDMEDIANAS_EMP_SH_33', 'RGO_SDGRANDES_EMP_SH_23', 'RGO_SDGTIA_PREFE_DIFF_21', 'RGO_SDCARTFIANZA_SH_22', 'RGO_SDGTIA_PREFE_SH_21', 'ST_CREDITO_SH_33', 'RGO_SDDESCLETRAS_STD', 'RGO_SD_NOPREFE_17', 'ST_CREDITO_SH_21', 'RGO_SDTARJETA_EMP_LOG_22', 'RGO_SDCREDITO_MAX_SH_21', 'RGO_SDACEPBANCA_17', 'RGO_SDMEDIANAS_EMP_MAX_DIFF_21', 'RGO_SDTARJETA_EMP_SH_22', 'RGO_SDTOTAL_DIFF_23', 'RGO_SDCREDITO_MAX_22', 'RGO_SDCREDITO_DIFF_21', 'RGO_SDACEPBANCA_19', 'RGO_SDDESCLETRAS_MAX_21', 'RGO_SDLEA_PEM_DIFF_21', 'RGO_SD_NOPREFE_SKEW', 'RGO_SD_NOPREFE_MAX_DIFF_33', 'RGO_SDFACTORING_MAX_SH_33', 'RGO_SD_SINLEASING_DIFF_22', 'RGO_SDGTIA_HIPOTEC_MAX_DIFF_22', 'RGO_SDINDIRECTO_SH_23', 'RGO_SDGTIA_HIPOTEC_SH_22', 'RGO_SDTARJETA_EMP_LOG_23', 'RGO_SDGTIA_PREFE_MAX_SH_22', 'RGO_SDPEQUENIAS_EMP_LOG_23', 'RGO_SDGTIA_HIPOTEC_16', 'RGO_SDDESCLETRAS_8', 'RGO_SDLEASING_5', 'RGO_SDPREST_EMP_LOG_33', 'RGO_SDDESCLETRAS_14', 'RGO_SDDESCLETRAS_MAX_SH_22', 'RGO_SDCREDITO_SKEW', 'RGO_SDPREST_EMP_SKEW', 'RGO_SDCREDITO_MAX_DIFF_22', 'RGO_SDACEPBANCA_MAX_SH_23', 'RGO_SDINDIRECTO_SH_33', 'RGO_SDCOMEX_DIFF_33', 'RGO_SDGTIA_HIPOTEC_19', 'RGO_SDMEDIANAS_EMP_DIFF_23', 'RGO_SDLEA_PEM_SH_23', 'RGO_SDTARJETA_EMP_MAX_SH_33', 'RGO_SDPEQUENIAS_EMP_SH_21', 'RGO_SDGRANDES_EMP_MAX_DIFF_23', 'RGO_SDGTIA_HIPOTEC_LOG_21', 'RGO_SDLEASING_18', 'RGO_SDCARTFIANZA_MAX_SH_23', 'RGO_SDGTIA_HIPOTEC_STD', 'RGO_SDCOMEX_MAX_SH_21', 'RGO_SDPEQUENIAS_EMP_LOG_34', 'RGO_SDCARTFIANZA_SH_23', 'RGO_SDINDIRECTO_MAX_LOG_33', 'RGO_SDGRANDES_EMP_17', 'RGO_SDPEQUENIAS_EMP_KURT', 'RGO_SDCARTFIANZA_MAX_DIFF_23', 'RGO_SDGRANDES_EMP_LOG_21', 'RGO_SD_SINLEASING_DIFF_23', 'RGO_SDFACTORING_MAX_DIFF_21', 'RGO_SDCOMEX_11', 'RGO_SDTARJETA_EMP_DIFF_23', 'RGO_SDFACTORING_MAX_SH_22', 'RGO_SD_SINLEASING_MAX_SH_33', 'RGO_SDMEDIANAS_EMP_SH_23', 'RGO_SDCARTFIANZA_21', 'RGO_SDCOMEX_16', 'RGO_SDCOMEX_LOG_33', 'ST_CREDITO_1.0', 'RGO_SDFACTORING_LOG_33', 'RGO_SDPEQUENIAS_EMP_SH_22', 'RGO_SDGTIA_HIPOTEC_DIFF_22', 'RGO_SDMICROEMPRESA_11', 'RGO_SDMEDIANAS_EMP_MAX_33', 'RGO_SDMICROEMPRESA_LOG_22', 'RGO_SDLEA_PEM_SH_21', 'RGO_SD_NOPREFE_DIFF_23', 'RGO_SDPEQUENIAS_EMP_MAX_SH_22', 'RGO_SDGTIA_PREFE_MAX_SH_23', 'RGO_SDACEPBANCA_LOG_23', 'RGO_SDINDIRECTO_MAX_DIFF_21', 'RGO_SDACEPBANCA_MAX_DIFF_22', 'RGO_SDGTIA_PREFE_MAX_21', 'RGO_SDCOMEX_MAX_SH_22', 'RGO_SDCOMEX_MAX_DIFF_21', 'RGO_SD_NOPREFE_33', 'RGO_SDCARTFIANZA_22', 'RGO_SDCOMEX_SH_23', 'RGO_SDMEDIANAS_EMP_LOG_21', 'RGO_SDCARTFIANZA_MAX_DIFF_21', 'RGO_SDMICROEMPRESA_MAX_22', 'RGO_SDACEPBANCA_SH_23', 'RGO_SDACEPBANCA_13', 'RGO_SD_SINLEASING_SH_22', 'RGO_SDGTIA_HIPOTEC_MAX_LOG_22', 'RGO_SD_SINLEASING_LOG_34', 'RGO_SDCOMEX_MAX_DIFF_33', 'RGO_SDGTIA_PREFE_SH_22', 'RGO_SDMICROEMPRESA_4', 'RGO_SDCOMEX_5', 'RGO_SDCREDITO_DIFF_33', 'RGO_SDLEA_PEM_KURT', 'RGO_SDGRANDES_EMP_MAX_LOG_33', 'RGO_SDACEPBANCA_DIFF_33', 'RGO_SDGTIA_PREFE_MAX_DIFF_33', 'RGO_SDMICROEMPRESA_DIFF_21', 'RGO_SDDESCLETRAS_22', 'RGO_SD_NOPREFE_1', 'RGO_SDLEA_PEM_MAX_DIFF_23', 'RGO_SDMICROEMPRESA_17', 'RGO_SDACEPBANCA_LOG_22', 'RGO_SDINDIRECTO_SKEW', 'RGO_SDACEPBANCA_12', 'RGO_SDGTIA_PREFE_LOG_33', 'RGO_SDACEPBANCA_DIFF_23', 'RGO_SDGRANDES_EMP_MAX_SH_21', 'RGO_SDDESCLETRAS_MAX_DIFF_33', 'RGO_SD_SINLEASING_8', 'RGO_SDACEPBANCA_LOG_34', 'RGO_SDLEA_PEM_DIFF_23', 'RGO_SDTARJETA_EMP_DIFF_33', 'RGO_SDTOTAL_MAX_SH_21', 'RGO_SDDESCLETRAS_DIFF_23', 'RGO_SDPEQUENIAS_EMP_MAX_DIFF_33', 'RGO_SDGTIA_HIPOTEC_MAX_SH_23', 'RGO_SD_NOPREFE_MAX_DIFF_34', 'RGO_SDTOTAL_KURT', 'RGO_SD_SINLEASING_MAX_DIFF_21', 'RGO_SDMICROEMPRESA_MAX_DIFF_22', 'RGO_SDCREDITO_STD', 'RGO_SDFACTORING_SH_21', 'RGO_SDGRANDES_EMP_SH_33', 'RGO_SDLEASING_SH_23', 'RGO_SDLEASING_SKEW', 'RGO_SDTOTAL_SH_22', 'RGO_SDCOMEX_LOG_21', 'RGO_SDLEASING_MAX_22', 'CD_BANCO_123', 'RGO_SDCOMEX_LOG_23', 'RGO_SDGTIA_HIPOTEC_10', 'RGO_SD_NOPREFE_LOG_34', 'RGO_SDTOTAL_SH_33', 'RGO_SDGRANDES_EMP_STD', 'RGO_SDGTIA_PREFE_DIFF_23', 'RGO_SDGRANDES_EMP_MAX_DIFF_22', 'RGO_SDTOTAL_SKEW', 'RGO_SDGTIA_HIPOTEC_MAX_SH_34', 'RGO_SDLEASING_11', 'ST_CREDITO_SH_23', 'RGO_SDLEASING_KURT', 'RGO_SDGTIA_HIPOTEC_14', 'RGO_SDGRANDES_EMP_LOG_23', 'RGO_SDCOMEX_SH_33']

print(len(COLF))

DF = DF_ENDEUDAMIENTO[COLF]

del DF_ENDEUDAMIENTO

DF_ENDEUDAMIENTO = DF.copy()

del DF

DF_ENDEUDAMIENTO.head()
df = pd.concat([df_stock_train, df_stock_test], axis =0)

ID_CLIENTE = df.index

df.head()
fh_n = df.FH_NACIMIENTO

fh_a = df.FH_ALTA

fh_n = pd.to_datetime(fh_n, format='%Y-%m-%d', errors='ignore').dt.year + (pd.to_datetime(fh_n, format='%Y-%m-%d', errors='ignore').dt.month - 1)/12

fh_a = pd.to_datetime(fh_a, format='%Y-%m-%d', errors='ignore').dt.year + (pd.to_datetime(fh_a, format='%Y-%m-%d', errors='ignore').dt.month - 1)/12



#fh_n.fillna(value = fh_n.median(),inplace=True)

#fh_a.fillna(value=fh_a.median(), inplace=True)

df.FH_NACIMIENTO = 2020-fh_n

df.FH_ALTA = 2020-fh_a

df['FH_NACIMIENTO_year'] = pd.to_datetime(fh_n, format='%Y-%m-%d', errors='ignore').dt.year

df['FH_NACIMIENTO_month'] = pd.to_datetime(fh_n, format='%Y-%m-%d', errors='ignore').dt.month

df['FH_ALTA_year'] = pd.to_datetime(fh_a, format='%Y-%m-%d', errors='ignore').dt.year

df['FH_ALTA_month'] = pd.to_datetime(fh_a, format='%Y-%m-%d', errors='ignore').dt.month



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

y_train.head()
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
COLZ  = ['RGO_PASIVO_MAX_21', 'TP_VIVI', 'RGO_RIESGO_SIST', 'RGO_SDO_VALORES_STD', 'RGO_SDPREST_EMP_MAX_21', 'TP_PERSONA', 'NU_CTA_TDD_1', 'RGO_SDPREST_EMP_MAX_33', 'NU_CTA_NOMINA_1', 'RGO_SDMICROEMPRESA_MAX_33', 'RGO_PASIVO_MAX_23', 'RGO_SDMICROEMPRESA_MAX_21', 'RGO_SDPREST_EMP_MAX_34', 'RGO_SDMICROEMPRESA_MAX_34', 'RGO_SDO_MEDIO_PASIVO', 'ST_CREDITO_33', 'CD_BANCO_6', 'ST_CREDITO_LOG_33', 'RGO_MARGEN_TDC_PNATURAL_10', 'NU_CTA_NOMINA_LOG_22', 'RGO_SEGU_NO_VINC_STD', 'RGO_SDPREST_EMP_MAX_22', 'RGO_PASIVO_LOG_23', 'RGO_SDLEA_PEM_MAX_LOG_21', 'RGO_MARGEN_COBRANZA_LIB_MAX_21', 'RGO_PASIVO_LOG_21', 'RGO_SDMICROEMPRESA_MAX_22', 'ST_CREDITO_SH_33', 'RGO_SDO_PREST_HIP_11', 'RGO_RIEGO_BBVA', 'NU_CTA_CTS_1', 'NU_CTA_NOMINA_STD', 'NU_CTA_NOMINA_0', 'RGO_SDPREST_EMP_MAX_23', 'MES_T0', 'RGO_SDO_TDC_PNATURAL_20', 'RGO_SEGU_NO_VINC_LOG_33', 'RGO_SDLEA_PEM_MAX_SH_34', 'RGO_SDLEA_PEM_6', 'RGO_PASIVO_23', 'NU_CTA_VALORES_0', 'RGO_MARGEN_TDC_PNATURAL_20', 'RGO_MARGEN_OPER_MES', 'CD_SBS_CPP', 'FH_ALTA', 'CD_BANCO_237', 'NU_CTA_DOMI_0', 'NU_CTA_TDD_0', 'NU_CTA_NOMINA_23', 'NU_CTA_NOMINA_LOG_33', 'RGO_PASIVO_MAX_22', 'NU_CTA_CTS_0', 'RGO_MARGEN_LSNG_12', 'RGO_SDO_TDC_PNATURAL_11', 'RGO_MARGEN_PREST_HIP_13', 'ST_CREDITO_SH_34', 'RGO_MARGEN_STANDBY_MAX_21', 'NU_CTA_TDD_STD', 'CD_SBS_NORMAL', 'RGO_SDO_TDC_PNATURAL_MAX_21', 'ST_CREDITO_34', 'RGO_SEGU_VINC_6', 'NU_CTA_NOMINA_SKEW', 'NU_CTA_TDC_PNATURAL_1', 'RGO_SD_SINLEASING_MAX_LOG_23', 'RGO_PASIVO_LOG_22', 'RGO_MARGEN_CONFIRMING_MAX_21', 'CD_SBS_DDP', 'RGO_SDLEA_PEM_16', 'RGO_SEGU_NO_VINC_16', 'RGO_SDO_CTE_11', 'RGO_MARGEN_PREST_CONS_6', 'RGO_MARGEN_STANDBY_LOG_21', 'RGO_MARGEN_CTE_LOG_21', 'NU_CTA_TDC_PNATURAL_0', 'RGO_SDPREST_EMP_33', 'RGO_PASIVO_LOG_33', 'RGO_PASIVO_3', 'RGO_MARGEN_VALORES_3', 'RGO_SDPREST_EMP_LOG_33', 'RGO_MARGEN_CTE_13', 'RGO_SEGU_VINC_1', 'RGO_MARGEN_COBRANZA_LIB_DIFF_33', 'RGO_SDMICROEMPRESA_18', 'RGO_MARGEN_AHORRO_LOG_21', 'RGO_SDO_MEDIO_ACTIVO', 'RGO_MARGEN_PREST_CONS_MAX_SH_34', 'NU_CTA_CTE_0', 'RGO_SDLEA_PEM_LOG_33', 'RGO_SDINDIRECTO_LOG_22', 'ST_CREDITO_21', 'RGO_PASIVO_20', 'RGO_SDO_VALORES_LOG_21', 'RGO_SDGTIA_HIPOTEC_14', 'ST_CREDITO_22', 'RGO_ACTIVO_5', 'CD_BANCO_156', 'RGO_MARGEN_COBRANZA_LIB_21', 'RGO_ACTIVO_18', 'RGO_SEGU_NO_VINC_LOG_21', 'RGO_SEGU_NO_VINC_MAX_21', 'RGO_PASIVO_22', 'ST_CREDITO_5.0', 'RGO_MARGEN_TDC_PNATURAL_1', 'RGO_SDLEA_PEM_LOG_34', 'RGO_MARGEN_OPER_ACUM', 'NU_CTA_TDD_KURT', 'RGO_MARGEN_STANDBY_10', 'RGO_MARGEN_AHORRO_16', 'RGO_SEGU_NO_VINC_MAX_33', 'RGO_MARGEN_PREST_COM_8', 'RGO_MARGEN_PREST_COM_9', 'RGO_SEGU_NO_VINC_13', 'RGO_SDLEA_PEM_LOG_21', 'RGO_MARGEN_CARTERA_6', 'RGO_SEGU_VINC_3', 'RGO_MARGEN_AHORRO_MAX_21', 'RGO_SDLEA_PEM_SH_34', 'CD_BANCO_82', 'RGO_ACTIVO_MAX_SH_22', 'RGO_MARGEN_PREST_COM_3', 'RGO_SDO_AVAL_21', 'RGO_SDO_CTS_SH_34', 'RGO_MARGEN_PREST_CONS_20', 'RGO_SEGU_NO_VINC_LOG_23', 'RGO_SEGU_VINC_STD', 'RGO_SDO_LSNG_7', 'RGO_MARGEN_COBRANZA_LIB_15', 'RGO_SEGU_NO_VINC_LOG_22', 'RGO_SD_SINLEASING_MAX_LOG_21', 'RGO_MARGEN_AVAL_MAX_21', 'FH_NACIMIENTO', 'RGO_SDO_VALORES_13', 'RGO_SDLEA_PEM_LOG_22', 'RGO_SDPREST_EMP_LOG_22', 'RGO_SDO_VALORES_6', 'RGO_ACTIVO_LOG_21', 'RGO_SDO_TDC_PJURIDICA_MAX_21', 'RGO_SDMICROEMPRESA_20', 'RGO_SDO_PREST_COM_MAX_SH_34', 'RGO_MARGEN_COBRANZA_LIB_LOG_33', 'RGO_SDINDIRECTO_LOG_21', 'RGO_SEGU_NO_VINC_18', 'NU_CTA_LSNG_0', 'RGO_SDPREST_EMP_23', 'NU_CTA_TDD_SKEW', 'RGO_SDPREST_EMP_21', 'RGO_SDINDIRECTO_DIFF_21', 'ST_CREDITO_SH_21', 'RGO_SDPREST_EMP_LOG_34', 'RGO_MARGEN_COBRANZA_GAR_1', 'RGO_SDO_COMEX_20', 'RGO_MARGEN_PREST_CONS_MAX_21', 'RGO_ACTIVO_SH_22', 'RGO_MARGEN_STANDBY_LOG_23', 'RGO_SDO_PREST_COM_SH_33', 'RGO_SDO_DESC_LETRA_STD', 'RGO_SDO_PREST_COM_22', 'RGO_SDO_TDC_PNATURAL_6', 'RGO_SDINDIRECTO_SH_21', 'RGO_SDLEA_PEM_LOG_23', 'RGO_SDO_LSNG_MAX_DIFF_21', 'RGO_SDO_COMEX_SH_22', 'RGO_SDO_CTE_MAX_SH_23', 'RGO_SDO_LSNG_SH_21', 'RGO_SDO_FMUTUO_MAX_DIFF_22', 'RGO_SDO_VALORES_LOG_33', 'RGO_SDPEQUENIAS_EMP_33', 'RGO_SD_SINLEASING_34', 'RGO_MARGEN_TDC_PNATURAL_LOG_22', 'RGO_SDGTIA_HIPOTEC_MAX_SH_34', 'RGO_MARGEN_CTE_SH_22', 'RGO_SDO_PREST_HIP_3', 'RGO_SDCOMEX_3', 'RGO_MARGEN_STANDBY_8', 'RGO_MARGEN_PZO_SH_21', 'RGO_MARGEN_CONFIRMING_LOG_22', 'RGO_SDO_AVAL_15', 'CD_BANCO_123', 'RGO_SDACEPBANCA_19', 'RGO_SDLEA_PEM_SH_33', 'RGO_SDDESCLETRAS_MAX_SH_22', 'RGO_SDCREDITO_DIFF_33', 'RGO_SDO_COMEX_6', 'CD_BANCO_101', 'RGO_MARGEN_PREST_CONS_8', 'RGO_SEGU_NO_VINC_MAX_SH_33', 'RGO_MARGEN_COBRANZA_LIB_LOG_23', 'RGO_SDLEA_PEM_15', 'RGO_SEGU_NO_VINC_SH_23', 'RGO_SDLEA_PEM_MAX_LOG_23', 'RGO_SDPEQUENIAS_EMP_DIFF_33', 'RGO_SDO_PREST_COM_MAX_SH_21', 'RGO_SDDESCLETRAS_11', 'ST_CREDITO_LOG_34', 'RGO_SDMICROEMPRESA_17', 'RGO_MARGEN_PREST_HIP_SH_34', 'RGO_MARGEN_COBRANZA_LIB_LOG_21', 'RGO_SD_SINLEASING_MAX_DIFF_21', 'RGO_SDMEDIANAS_EMP_MAX_33', 'RGO_SDLEA_PEM_19', 'RGO_MARGEN_CTS_17', 'RGO_SDLEA_PEM_SH_22', 'RGO_SDO_LSNG_MAX_DIFF_22', 'RGO_MARGEN_CTS_MAX_SH_22', 'RGO_SDCREDITO_LOG_33', 'RGO_MARGEN_LSNG_MAX_SH_21', 'RGO_MARGEN_AVAL_MAX_SH_21', 'RGO_SDGRANDES_EMP_SH_22', 'RGO_SD_NOPREFE_13', 'RGO_SDFACTORING_9', 'RGO_MARGEN_CTE_LOG_34', 'RGO_MARGEN_CTE_LOG_22', 'RGO_MARGEN_AVAL_MAX_SH_22', 'RGO_MARGEN_CTS_LOG_21', 'RGO_SDDESCLETRAS_MAX_SH_23', 'RGO_MARGEN_PREST_CONS_SH_34', 'RGO_MARGEN_AHORRO_20', 'RGO_SDFACTORING_SH_33', 'RGO_MARGEN_VALORES_19']
XF_ = XF[COLZ]

del XF

XF = XF_.copy()

del XF_

XF.fillna(-999, inplace = True)

XF.head()
IX = [c for c in XF.index if c in df_stock_train.index]

IT = df_stock_test.index

print(len(IX), len(IT))

y_train = y_train.loc[IX]

y_train.head()
X_train = XF.loc[IX]

X_test = XF.loc[IT]
gc.collect()
%%time





NFOLDS = 15

folds = StratifiedKFold(n_splits=NFOLDS)



columns =  COLZ[0:160]

splits = folds.split(X_train, y_train)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X_train.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



 

model = XGBClassifier(base_score=.24624935923356592, booster='gbtree',

       colsample_bylevel=1, colsample_bytree=1, gamma=0,

       #learning_rate=0.15625549474872114, max_delta_step=0, max_depth= 9,

       learning_rate= 0.07546821436642331, max_delta_step=0, max_depth= 9, 

       min_child_weight=1, missing=None, n_estimators=200, n_jobs=-1,

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

        verbose=False,

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



#Mean AUC = 0.868979287472545

#Out of folds AUC = 0.8681966724862532


plt.figure(figsize=(16, 16))

feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(15));
df_stock_submit = pd.read_csv('/kaggle/input/attrition-persona-juridica-bbva/DATA_STOCK_SUBMIT_SAMPLE.csv')

df_stock_submit.FUGA_3M = y_preds

df_stock_submit.head()



dzpL10_AVG = df_stock_submit.copy()

dzpL10_AVG.to_csv('submit_bbva_33222M_COUNT_MAX_15K_160F_9D.csv', index = False)



from IPython.display import HTML





def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



create_download_link(filename='submit_bbva_33222M_COUNT_MAX_15K_160F_9D.csv')
