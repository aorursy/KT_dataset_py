import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import copy

import csv

from IPython.display import display, HTML

sns.set(style="whitegrid", color_codes=True)
#Read data and explore

limit_rows=800000

data_m1 = pd.read_csv("../input/train_ver2.csv", nrows=limit_rows)



display(data_m1)
data_m1 = data_m1.loc[data_m1['fecha_dato'] == '2015-01-28',:]
data_m1.ncodpers.unique().shape
data_m1.shape
data_m1.isnull().sum()
df = data_m1.drop(["ult_fec_cli_1t", "conyuemp"], axis=1) 
#Impute missing values in the income column 

grouped        = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()

new_incomes    = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]

new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")

df.sort_values("nomprov",inplace=True)

df             = df.reset_index()

new_incomes    = new_incomes.reset_index()



df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()

df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()

df.sort_values(by="fecha_dato",inplace=True)
df.shape
df.dropna(inplace=True)
df.shape
# Change datatype

df["age"]   = pd.to_numeric(df["age"], errors="coerce") 

df["antiguedad"]   = pd.to_numeric(df["antiguedad"], errors="coerce") 

df["indrel_1mes"]   = pd.to_numeric(df["indrel_1mes"], errors="coerce") 
df[df["antiguedad"]<0] = 0
df.dtypes
corr = df.corr()
sns.heatmap(corr, vmax=.3,

            square=True, 

            linewidths=.5, cbar_kws={"shrink": .5})

plt.xticks(rotation=90)

plt.yticks(rotation=0)
df.ind_nomina_ult1 = df.ind_nomina_ult1.astype('int')

df.ind_nom_pens_ult1 = df.ind_nom_pens_ult1.astype('int')
df = df.loc[df.segmento != '2', :]
df.ind_empleado.value_counts()
#For now lets keep keep only N

df = df.loc[df.ind_empleado == 'N', :]
df.shape
df.pais_residencia.value_counts()
#Lets stick with Spain

df = df.loc[df.pais_residencia == 'ES', :]
df.sexo.value_counts()
df.fecha_alta = pd.to_datetime(df.fecha_alta)

df.fecha_alta.describe()
#Lets make a column -- month of joining

df['month_of_joining'] = df.fecha_alta.dt.month
plt.hist(df.month_of_joining, bins=12)
df.ind_nuevo.value_counts()
#Lets stick with all old customers

df = df.loc[df.ind_nuevo == 0, :]
df.antiguedad.describe()


sns.distplot(df.antiguedad, bins=240 )
df.indrel.value_counts()
#Lets focus only on primary customers

df = df.loc[df.indrel == 1, :]
df.indrel_1mes.value_counts()
#Lets focus on 1

df = df.loc[df.indrel_1mes == 1, :]
df.tiprel_1mes.value_counts()
df.indresi.value_counts()
df.indext.value_counts()
df.canal_entrada.value_counts()
top3 = ['KHE', 'KAT', 'KFC']

df.loc[~df.canal_entrada.isin(top3), 'canal_entrada'] = 'OTHER' 
df.canal_entrada.value_counts()
df.indfall.value_counts()
#Indfall mostly N

df = df.loc[df.indfall == 'N', :]
df.tipodom.value_counts()
df.cod_prov.value_counts()
df.cod_prov.unique().shape
df.ind_actividad_cliente.value_counts()
df.renta.describe()
df.loc[df.renta >= 0.15* 10**7, 'renta'] = 0.15 * 10**7
plt.hist(np.log(df.renta), bins=40)

df.renta = np.log(df.renta)
df_a = df.loc[:, ['sexo', 'ind_actividad_cliente']].join(df.loc[:, "ind_ahor_fin_ult1": "ind_recibo_ult1"])

df_a = df_a.groupby(['sexo', 'ind_actividad_cliente']).sum()

df_a = df_a.T

df_a.head()
df_a.plot(kind='barh', stacked=True, fontsize=10, figsize=[10,8], colormap='gist_ncar')

plt.title('Popularity of products by sex and activity index', fontsize=20) 

plt.xlabel('Number of customers', fontsize=17) 

plt.ylabel('Products_names', fontsize=17) 

plt.legend(["Sex:H; Activity_Ind:0", "Sex:H; Activity_Ind:1", "Sex:V; Activity_Ind:0", 

            "Sex:V; Activity_Ind:1"], prop={'size':15}) 
df.sexo = df.sexo.astype('category')

df.tiprel_1mes = df.tiprel_1mes.astype('category')

df.indext = df.indext.astype('category')

df.canal_entrada = df.canal_entrada.astype('category')

df.ind_actividad_cliente = df.ind_actividad_cliente.astype('category')

df.segmento = df.segmento.astype('category')

df.renta = df.renta.astype('category')

df.ind_nomina_ult1 = df.ind_nomina_ult1.astype('int64')

df.ind_nom_pens_ult1 = df.ind_nom_pens_ult1.astype('int64')



df_model = df.drop(['fecha_dato','fecha_alta', 'ind_empleado', 'pais_residencia','ind_nuevo', 'indrel', 'indrel_1mes', 'indresi', 

                   'cod_prov', 'nomprov', 'indfall', 'tipodom', 'index'], axis=1)

df_model.dtypes
df_model.groupby('ind_actividad_cliente')['tiprel_1mes'].value_counts()
mapping_dict = {

'sexo'          : {'V':0, 'H':1},

#'ind_nuevo'     : {'0':0, '1':1, -99:1},

#'indrel'        : {'1':0, '99':1, -99:1},

#'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},

'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},

#'indresi'       : {-99:0, 'S':1, 'N':2},

'indext'        : {-99:0, 'S':0, 'N':1},

#'conyuemp'      : {-99:0, 'S':1, 'N':2},

#'indfall'       : {-99:0, 'S':1, 'N':2},

#'tipodom'       : {-99:0, '1':1},

#'ind_actividad_cliente' : {'0':0, '1':1, -99:2},

'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:3},

#'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},

'canal_entrada' : {'KHE':4, 'KAT':3, 'KFC':2, 'OTHER':1}

}

cat_cols = list(mapping_dict.keys())

for col in cat_cols:

    print(col)

    df_model[col] = df_model[col].apply(lambda x: mapping_dict[col][x])

    
IDcol = 'ncodpers'

target = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',

          'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

          'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

          'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']



predictors = ['sexo', 'age', 'antiguedad', 'tiprel_1mes', 'indext', 'canal_entrada', 'ind_actividad_cliente', 'renta',

              'segmento', 'month_of_joining']
df_model.shape
X = df_model[predictors]

y = df_model[target]
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf1 = DecisionTreeClassifier(min_samples_split=50, max_depth=10)



clf1.fit(X_train, y_train)
clf1.feature_importances_
clf1.score(X_test, y_test)
clf1.score(X_test, y_test)
y_test.sum()
y_test_pred= clf1.predict(X_test)

for i in range(0,24):

    print(metrics.confusion_matrix(y_test_pred[:,i], y_test.iloc[:,i]))
#Lets start predicting them one by one

i = 2



clf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=-1, min_samples_split=50)

clf.fit(X_train, y_train.iloc[:,i])



print(metrics.confusion_matrix(y_train.iloc[:,i], clf.predict(X_train)))

print(metrics.confusion_matrix(y_test.iloc[:,i], clf.predict(X_test)))



print(metrics.precision_score(y_train.iloc[:,i], clf.predict(X_train)))

print(metrics.precision_score(y_test.iloc[:,i], clf.predict(X_test)))

print(X_train.columns,clf.feature_importances_)
IDcol = 'ncodpers'

target = 'ind_cco_fin_ult1'

#target = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',

#          'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

#          'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

#          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

#          'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']



predictors = ['sexo', 'age', 'antiguedad', 'tiprel_1mes', 'indext', 'canal_entrada', 'ind_actividad_cliente', 'renta',

              'segmento', 'month_of_joining',

             'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', #'ind_cco_fin_ult1', 

              'ind_cder_fin_ult1', 'ind_cno_fin_ult1',

          'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

          'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

          'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

X = df_model[predictors]

y = df_model[target]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



clf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=-1, min_samples_split=50)

clf.fit(X_train, y_train)



print(metrics.confusion_matrix(y_train, clf.predict(X_train)))

print(metrics.confusion_matrix(y_test, clf.predict(X_test)))



print(metrics.precision_score(y_train, clf.predict(X_train)))

print(metrics.precision_score(y_test, clf.predict(X_test)))



print(clf.feature_importances_, X_train.columns)
IDcol = 'ncodpers'

target = 'ind_cco_fin_ult1'

#target = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',

#          'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

#          'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

#          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

#          'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']



predictors = ['sexo', 'age', 'antiguedad', 'tiprel_1mes', 'indext', 'canal_entrada', 'ind_actividad_cliente', 'renta',

              'segmento', 'month_of_joining',

             'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', #'ind_cco_fin_ult1', 

              'ind_cder_fin_ult1', 'ind_cno_fin_ult1',

          'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

          'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

          'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

X = df_model[predictors]

y = df_model[target]



X_train = X.iloc[0:400000,:] 

X_test  = X.iloc[400000:, :]  

y_train = y.iloc[0:400000]

y_test  = y.iloc[400000:]



clf = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=-1, min_samples_split=50, random_state=1)

clf.fit(X_train, y_train)



print(metrics.confusion_matrix(y_train, clf.predict(X_train)))

print(metrics.confusion_matrix(y_test, clf.predict(X_test)))



print(metrics.precision_score(y_train, clf.predict(X_train)))

print(metrics.precision_score(y_test, clf.predict(X_test)))



print(clf.feature_importances_, X_train.columns)
#Get customer ID for customers with false positives

tmp1 = y_test - clf.predict(X_test)

#tmp2 = y_train - clf.predict(X_train)

#tmp = pd.concat([tmp1, tmp2], ignore_index=True)

fp_custid = df_model.iloc[400000:,:].ncodpers.loc[tmp1.values <0]
X_fp = df_model.iloc[400000:,:].loc[df_model.ncodpers.isin(fp_custid), predictors]

y_fp = df_model.iloc[400000:].loc[df_model.ncodpers.isin(fp_custid), target]

metrics.confusion_matrix(y_fp, clf.predict(X_fp))
tot = pd.read_csv('../input/train_ver2.csv', usecols=['ncodpers', 'ind_cco_fin_ult1'])
tot.ncodpers.unique().shape
#df_custid = tot.loc[tot.ncodpers.isin(fp_custid),:]

#df_custid.shape
changed_status = []

for custid in fp_custid:

    tmp = tot.loc[tot.ncodpers == custid, 'ind_cco_fin_ult1']

    if tmp.sum() == 0:

        continue

    else:

        changed_status.append(custid)

    
len(changed_status)