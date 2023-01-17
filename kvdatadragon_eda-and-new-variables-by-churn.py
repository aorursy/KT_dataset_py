# Nativos

import os

import sys



#calculo

import numpy as np

import pandas as pd



#modelamiento

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

from sklearn.model_selection import StratifiedKFold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectFromModel, VarianceThreshold

import xgboost as xgb

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import RFE

import category_encoders as ce



#grafico

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline

sns.set(style="whitegrid")



#warning ignore future

import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings("ignore")



subfolder = "../input"

print(os.listdir(subfolder))
set_parameter_csv = {

    'sep': ',',

    'encoding': 'ISO-8859-1',

    'low_memory': False

}



train = pd.read_csv('{}/churn_data_train.csv'.format(subfolder), **set_parameter_csv).round(2)

display(train.head(3))

test = pd.read_csv('{}/churn_data_test.csv'.format(subfolder), **set_parameter_csv).round(2)

display(test.head(3))

sub = pd.read_csv('{}/sample_submit.csv'.format(subfolder), **set_parameter_csv)

display(sub.head(3))
# CHECK COLUMNS

print(train.columns)

print("="*100)

print(test.columns)
def view_cat(data, col_init, col_out, **kwargs):

    color_label = kwargs.get('color_label', 'black')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    

    cross = pd.crosstab(data[col_out], data[col_init])

    sum_total = sum([cross[col].sum() for col in cross.columns])

    sns.heatmap(

        cross/sum_total, 

        annot=True, ax=axes[0], center=0, cmap="YlGnBu", fmt='.2%'

    )

    sns.barplot(

        x=col_init, y=col_out, data=data, ax=axes[1]

    )



def view_numeric(data, col_init, col_out, **kwargs):

    color_label = kwargs.get('color_label', 'black')

    bins = kwargs.get('bins', 3)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))



    sns.violinplot(x=col_out, y=col_init, data=data, ax=axes[0])

    sns.distplot(data[col_init], ax=axes[1])
# CHECK NULL

def search_null(data):

    return data.isnull().sum().where(lambda _ : _ > 0).dropna()



display(search_null(train))

print("/"*55)

display(search_null(test))
# CHECK ORDER OF NULLS

import missingno as msno



msno.matrix(train)
msno.matrix(test)
def correct_string_null(val):

    try:

        return float(val)

    except:

        return float(np.nan)

    

train['MontoCargadoTotal'] = train['MontoCargadoTotal'].apply(correct_string_null)

test['MontoCargadoTotal'] = test['MontoCargadoTotal'].apply(correct_string_null)
mode_adultomayor = train['AdultoMayor'].mode()[0]



train['AdultoMayor'].fillna(mode_adultomayor, inplace=True)

test['AdultoMayor'].fillna(mode_adultomayor, inplace=True)



train['AdultoMayor'] = train['AdultoMayor'].astype(str)

test['AdultoMayor'] = test['AdultoMayor'].astype(str)



train['AdultoMayor'].value_counts(dropna=False)
display(train.describe().T)

display(test.describe().T)
cols_str = list(test.describe(include=['object', 'bool']).columns[1:])

print(cols_str)
cols_num = list(test.describe().columns)

print(cols_num)
for col in cols_str:

    print("%%%%% ", col)

    display(train.groupby(by=[col])[cols_num].agg(['mean', 'median', 'min', 'max']))
col_target = 'Churn'



for val in train['Socio'].unique():

    sub_train = train[train['Socio'] == val][col_target]

    print(val, sub_train.shape)

    display(sub_train.value_counts(dropna=False, normalize=True))
def replace_null_socio(val, mes_cliente):

    if str(val).lower() == 'nan':

        return 'Si' if mes_cliente < 20 else 'No'

    

    return val



train['Socio'] = train[['Socio', 'MesesCliente']].apply(

    lambda _: replace_null_socio(_.Socio, _.MesesCliente), axis=1

)

test['Socio'] = test[['Socio', 'MesesCliente']].apply(

    lambda _: replace_null_socio(_.Socio, _.MesesCliente), axis=1

)
def repplace_null_col_numeric(val, es_socio, serie_replace):

    if str(val).lower() == 'nan':

        return serie_replace[es_socio]

    return val



for col in cols_num:

    median_socio = train.groupby(by=['Socio'])[col].median()

    print(col)

    display(median_socio)

    

    train[col] = train[[col, 'Socio']].apply(

        lambda _: repplace_null_col_numeric(_[col], _.Socio, median_socio), axis=1

    )

    test[col] = test[[col, 'Socio']].apply(

        lambda _: repplace_null_col_numeric(_[col], _.Socio, median_socio), axis=1

    )

    view_numeric(train, col, 'Churn')
# ADD NEW VAR 'deuda'

train['deuda'] = train[['MesesCliente', 'MontoCargadoMes', 'MontoCargadoTotal']].apply(

    lambda _: _.MesesCliente * _.MontoCargadoMes - _.MontoCargadoTotal, axis=1

)

test['deuda'] = test[['MesesCliente', 'MontoCargadoMes', 'MontoCargadoTotal']].apply(

    lambda _: _.MesesCliente * _.MontoCargadoMes - _.MontoCargadoTotal, axis=1

)
# ADD NEW VAR CATEGORYC 'MesesCliente_cat'

def meses_to_cat(val):

    if val <= 3:

        return 'nuevo'

    elif val <= 10:

        return 'reciente'

    elif val <= 30:

        return 'normal'

    else:

        return 'amtiguo'



train['MesesCliente_cat'] = train['MesesCliente'].apply(meses_to_cat)

test['MesesCliente_cat'] = test['MesesCliente'].apply(meses_to_cat)
for col in cols_num:

    if col != 'deuda':

        train[col + '_log'] = np.log(train[col])

        test[col + '_log'] = np.log(test[col])
def meses_log_to_cat(val):

    if val < 4:

        return 'nuevo'

    else:

        return 'amtiguo'



train['MesesCliente_cat'] = train['MesesCliente_log'].apply(meses_log_to_cat)

test['MesesCliente_cat'] = test['MesesCliente_log'].apply(meses_log_to_cat)
def total_to_cat(val):

    if val < 30:

        return 'menor30'

    if val < 70:

        return 'menor60'

    else:

        return 'mayor70'



train['MontoCargadoMes_cat'] = train['MontoCargadoMes'].apply(total_to_cat)

test['MontoCargadoMes_cat'] = test['MontoCargadoMes'].apply(total_to_cat)
def monto_total_to_cat(val):

    if val < 1500:

        return 'menor1500'

    elif val < 3000:

        return 'menor3000'

    else:

        return 'mayo3000'



train['MontoCargadoTotal_cat'] = train['MontoCargadoTotal'].apply(monto_total_to_cat)

test['MontoCargadoTotal_cat'] = test['MontoCargadoTotal'].apply(monto_total_to_cat)
display(train.drop(['ID'], axis=1).describe(include=['object', 'bool']).T)

display(test.drop(['ID'], axis=1).describe(include=['object', 'bool']).T)
cols_same_nan = ['SeguridadOnline','RespaldoOnline','ProteccionDispositivo',

                 'SoporteTecnico','TransmisionTV', 'TransmisionPeliculas']



for col in cols_str:

    if col in cols_same_nan:

        train[col].fillna('lostdata', inplace=True)

        test[col].fillna('lostdata', inplace=True)

    else:

        val_mode = train[col].mode()[0]

        train[col].fillna(val_mode, inplace=True)

        test[col].fillna(val_mode, inplace=True)

        

    view_cat(train, col, 'Churn')
cols_same_nan = ['SeguridadOnline','RespaldoOnline','ProteccionDispositivo',

                 'SoporteTecnico', 'TransmisionPeliculas', 'TransmisionTV']



train_no_servico = train[train['ServicioInternet'] == 'No']



for col in cols_same_nan:

    display(train_no_servico[col].value_counts(dropna=False))

    print("/"*100)
def fix_cols_same_nan(val):

    if val in ['Sin servicio de internet', 'lostdata']:

        return 'NotieneInternet'

    return val



for col in cols_same_nan:

    train[col] = train[col].apply(fix_cols_same_nan)

    test[col] = test[col].apply(fix_cols_same_nan)
for col in cols_str:

    tmp_group_by = train.groupby([col])['MontoCargadoMes'].median().round(3)

    display(tmp_group_by)

    

    col_flg = 'flg_{}_mayorMedianMontoMes'.format(col)

    train[col_flg] = train.apply(

        lambda x: 1 if x.MontoCargadoMes >= tmp_group_by[x[col]] else 0, axis = 1)



    test[col_flg] = test.apply(

        lambda x: 1 if x.MontoCargadoMes >= tmp_group_by[x[col]] else 0, axis = 1)
cols_dummies_one = ['FacturacionElectronica', 'Sexo', 'ServicioTelefonico', 'LineasMultiples',

                    'AdultoMayor', 'Socio', 'Dependientes', 'TransmisionTV', 'TransmisionPeliculas']



train = pd.get_dummies(train, drop_first=True, columns=cols_dummies_one)

test = pd.get_dummies(test, drop_first=True, columns=cols_dummies_one)

train.shape, test.shape
cols_dummies_many = ['MetodoPago', 'TerminoContrato', 'SoporteTecnico', 'SeguridadOnline', 'RespaldoOnline',

                    'ProteccionDispositivo']



train = pd.get_dummies(train, drop_first=False, columns=cols_dummies_many)

test = pd.get_dummies(test, drop_first=False, columns=cols_dummies_many)

train.shape, test.shape
col_wctm = 'deuda'

y_train = train[col_target]  

X_train = train.drop(['ID', 'Churn', col_wctm], axis=1).reset_index(drop=True)

X_test = test.drop(['ID', col_wctm], axis=1).reset_index(drop=True)



del train

del test

X_train.shape, X_test.shape, y_train.shape
te = ce.target_encoder.TargetEncoder(

    drop_invariant=True, return_df=True

)

    

X_train = te.fit_transform(X_train, y_train)

X_test = te.transform(X_test)

X_train.shape, X_test.shape, y_train.shape
X_train.dtypes