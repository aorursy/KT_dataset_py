# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
limit_rows   = 10000
df_train = pd.read_csv("/kaggle/input/santander-product-recommendation/train_ver2.csv.zip", nrows=limit_rows)
df_test = pd.read_csv("/kaggle/input/santander-product-recommendation/test_ver2.csv.zip")
limit_rows = 10000
df_train = pd.read_csv("/kaggle/input/santander-product-recommendation/train_ver2.csv.zip", nrows=limit_rows)
df_train
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_test = df_test.drop((missing_data[missing_data['Total'] > 2000]).index,1)
df_train = df_train.drop((missing_data[missing_data['Total'] > 2000]).index,1)
list(df_train.columns)
list(df_train.select_dtypes(include=['O']))
df_train["fecha_dato"] = pd.to_datetime(df_train["fecha_dato"],format="%Y-%m-%d")
df_train["fecha_alta"] = pd.to_datetime(df_train["fecha_alta"],format="%Y-%m-%d")
df_train["fecha_dato"].unique()
df_test = df_test.dropna(axis='index', how='any', subset=['indrel_1mes'])
df_test = df_test.dropna(axis='index', how='any', subset=['tiprel_1mes'])
print("succesfull")
df_train = df_train.dropna(axis='index', how='any', subset=['indrel_1mes'])
df_train = df_train.dropna(axis='index', how='any', subset=['tiprel_1mes'])
df_train
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_test = df_test.dropna(axis='index', how='any', subset=['sexo'])
df_train = df_train.dropna(axis='index', how='any', subset=['sexo'])
df_train
df_train["month"] = pd.DatetimeIndex(df_train["fecha_dato"]).month
df_train["age"]   = pd.to_numeric(df_train["age"], errors="coerce")
df_test["month"] = pd.DatetimeIndex(df_test["fecha_dato"]).month
df_test["age"]   = pd.to_numeric(df_test["age"], errors="coerce")
df_train
df_train.loc[df_train.age < 18,"age"]  = df_train.loc[(df_train.age >= 18) & (df_train.age <= 30),"age"].mean(skipna=True)
df_train.loc[df_train.age > 100,"age"] = df_train.loc[(df_train.age >= 30) & (df_train.age <= 100),"age"].mean(skipna=True)
df_train["age"].fillna(df_train["age"].mean(),inplace=True)
df_train["age"] = df_train["age"].astype(int)
df_train
df_train["pais_residencia"].unique()
country_d = {}
country=['ES', 'CH', 'DE', 'GB', 'BE', 'DJ', 'IE', 'QA', 'US', 'VE', 'DO',
       'SE', 'AR', 'CA', 'PL', 'CN', 'CM', 'FR', 'AT', 'RO', 'LU', 'PT',
       'CL', 'IT', 'MR', 'MX', 'SN', 'BR', 'CO', 'PE', 'RU', 'LT', 'EE',
       'MA', 'HN', 'BG', 'NO', 'GT', 'UA', 'NL', 'GA', 'IL', 'JP', 'EC',
       'IN', 'DZ', 'ET', 'SA', 'HU', 'JM', 'CI', 'CU', 'BO', 'TG', 'TN',
       'NG', 'AU', 'GR', 'DK', 'LB', 'UY', 'TH', 'SG', 'MD', 'SK', 'AD',
       'BY', 'HK', 'HR', 'EG', 'GQ', 'PR', 'ZA', 'PA', 'KE', 'TR', 'FI',
       'BA', 'SV', 'PY', 'PK', 'KR', 'AO', 'GN', 'IS', 'TW', 'MK', 'VN',
       'CZ', 'CR', 'MZ', 'MT', 'LY', 'GH', 'KH', 'AE', 'RS', 'OM', 'GE',
       'NI', 'GI', 'NZ', 'MM', 'PH', 'KW', 'BM', 'CG', 'ML', 'AL', 'ZW',
       'CF', 'GM', 'CD', 'BZ', 'KZ', 'GW', 'SL', 'LV']
for i,item in enumerate(country):
    country_d[item] = i
country_d
%%time
df_train["pais_residencia"] = df_train["pais_residencia"].map(country_d)
list(df_train["pais_residencia"].unique())
'''
%%time
count = 0
for item in df_train["pais_residencia"]:
    for key, value in country_d.items():
        if key == item:
            df_train.loc[count, "pais_residencia"] = value
            count=count+1
            break
print("wow succesfull!")
'''

df_train
df_train["pais_residencia"].unique()
df_test.pop("fecha_alta")
df_train.pop("fecha_alta")
df_train
df_test.pop("fecha_dato")
df_train.pop("fecha_dato")
df_test
%%time
df_test["pais_residencia"] = df_test["pais_residencia"].map(country_d)
df_test["pais_residencia"].unique()
'''
%%time
count = 0
for item in df_test["pais_residencia"]:
    for key, value in country_d.items():
        if key == item:
            df_test.loc[count, "pais_residencia"] = value
            count=count+1
            break
print("wow succesfull! again")
'''
list(df_train.select_dtypes(include=['O']))
df_train["ind_empleado"].unique()
df_test["ieF"] = df_test["ind_empleado"].map({'F':1,'N':0})
df_test["ieN"] = df_test["ind_empleado"].map({'F':0,'N':1})
df_test.pop("ind_empleado")
df_train["ieF"] = df_train["ind_empleado"].map({'F':1,'N':0})
df_train["ieN"] = df_train["ind_empleado"].map({'F':0,'N':1})
df_train.pop("ind_empleado")
df_train = df_train.loc[df_train['pais_residencia'] != "ES"]
df_train['pais_residencia'].unique()
df_train["pais_residencia"] = df_train["pais_residencia"].astype('int')

df_train["pais_residencia"].dtype
df_train["sexo"].unique()
df_train["sexV"] = df_train["sexo"].map({'V':1,'H':0})
df_train["sexH"] = df_train["sexo"].map({'V':0,'H':1})
df_train.pop("sexo")

df_test["sexV"] = df_test["sexo"].map({'V':1,'H':0})
df_test["sexH"] = df_test["sexo"].map({'V':0,'V':1})
df_test.pop("sexo")
list(df_train.select_dtypes(include=['O']))
df_train["tiprel_1mes"].unique()
df_train["t1mA"] = df_train["tiprel_1mes"].map({'A':1,'I':0})
df_train["t1mI"] = df_train["tiprel_1mes"].map({'A':0,'I':1})
df_train.pop("tiprel_1mes")
df_test["t1mA"] = df_test["tiprel_1mes"].map({'A':1,'I':0})
df_test["t1mI"] = df_test["tiprel_1mes"].map({'A':0,'I':1})
df_test.pop("tiprel_1mes")
df_train["indresi"].unique()
df_train["iS"] = df_train["indresi"].map({'S':1,'N':0})
df_train["iN"] = df_train["indresi"].map({'S':0,'N':1})
df_train.pop("indresi")
df_test["iS"] = df_test["indresi"].map({'S':1,'N':0})
df_test["iN"] = df_test["indresi"].map({'S':0,'N':1})
df_test.pop("indresi")
df_train["indext"].unique()
df_train["idS"] = df_train["indext"].map({'S':1,'N':0})
df_train["idN"] = df_train["indext"].map({'S':0,'N':1})
df_train.pop("indext")
df_test["idS"] = df_test["indext"].map({'S':1,'N':0})
df_test["idN"] = df_test["indext"].map({'S':0,'N':1})
df_test.pop("indext")
df_train["indfall"].unique()
df_train["ifS"] = df_train["indfall"].map({'S':1,'N':0})
df_train["ifN"] = df_train["indfall"].map({'S':0,'N':1})
df_train.pop("indfall")
df_test["ifS"] = df_test["indfall"].map({'S':1,'N':0})
df_test["ifN"] = df_test["indfall"].map({'S':0,'N':1})
df_test.pop("indfall")
list(df_train.columns)
list(df_test["renta"].unique())
#%%time
#df_train["renta"] = df_train["renta"].replace({'         NA': 0})
'''
%%time
count = 0
for item in df_train["renta"]:
    if "         NA" == item:
        df_train.loc[count, "renta"] = 0
    count=count+1
'''
%%time
df_test["renta"] = df_test["renta"].replace({'         NA': 0})
df_test["renta"] = df_test["renta"].astype(float)
list(df_test["renta"].unique())
df_test
'''
%%time
count = 0
for item in df_test["renta"]:
    if "         NA" == item:
        df_test.loc[count, "renta"] = 0
    count=count+1
'''
df_test
count_nan = len(df_train) - df_train.count()
nan_list = []
for item in df_train:
    if len(item) - df_train[item].count() != 0:
        nan_list.append(item)
print(count_nan)
print(nan_list)
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(60)
df_train = df_train.dropna(axis='index', how='any', subset = nan_list)
df_train
list(df_train.select_dtypes(include=['O']))
df_test["canal_entrada"].unique()
canal_entrada_d = {}
canal_entrada_l=['KAT', 'KHE', 'KFC', 'KHN', 'KFA', 'KHM', 'KHL', 'RED', 'KHQ',
       'KHO', 'KHK', 'KAZ', 'KEH', 'KBG', 'KHF', 'KHC', 'KHD', 'KAK',
       'KAD', 'KDH', 'KGC', 'KFK', 'KGN', 'KHP', 'KHS', '013', 'KFD',
       'KEW', 'KEG', 'KAE', 'KAB', 'KAG', 'KCC', 'KAF', 'KAC', 'KAY',
       'KES', 'KDU', 'KAA', 'KAP', 'KEN', 'KBZ', 'KEY', '007', 'KBJ',
       'KDM', 'KFP', 'KDR', 'KAN', 'KEV', 'KCH', 'KAR', 'KAI', 'KAS',
       'KCI', 'KFJ', 'KCF', 'KAQ', 'KAJ', 'KEJ', 'KEL', 'KFT', '025',
       'KAH', 'KAW', 'KFS', 'KDT', 'KGV', 'KFG', 'KAL', 'KGY', 'KCM',
       'KCB', 'KDQ', 'KFN', 'KGX', 'KBR', 'KEZ', 'KCG', 'KAO', 'KEU',
       'KCL', 'KBM', 'KFF', 'KFI', 'KFL', 'KFH', 'KCU', 'KBH', 'KBB',
       'KCO', 'KEI', 'KDY', 'KCD', 'KBO', 'KEA', 'KBL', 'KDO', 'KCA',
       'KBU', 'KBQ', 'KDV', 'KCK', 'KDC', '004', 'KCE', 'KDE', 'KEC',
       'KDX', 'KDA', 'KDS', 'KFU', 'KEB', 'KDG', 'KAM', 'KDF', 'KCN',
       'KDZ', 'KCS', 'KCR', 'KDW', 'KEO', 'KFM', 'KEK', 'KED', 'KDN',
       'KBF', 'KGW', 'KCQ', 'KEQ', 'KBW', 'KBE', 'KAV', 'KBY', 'KFV',
       'KBS', 'KDP', 'KFE', 'KBD', 'KBV', 'KDD', 'KEF', 'KCP', 'KBX',
       'KDB', 'KBN', 'KCT', 'KCV', 'KBP', 'KCX', 'KAU', 'KFR', 'KDI',
       'KFB', 'KGU', 'KHA', 'K00', 'KEE', 'KHR', 'KCJ', 'KEM', 'KDL']
for i,item in enumerate(canal_entrada_l):
    canal_entrada_d[item] = i
canal_entrada_d
df_train["canal_entrada"] = df_train["canal_entrada"].map(canal_entrada_d)
df_test["canal_entrada"] = df_test["canal_entrada"].map(canal_entrada_d)
'''
%%time
count = 0
for item in df_train["canal_entrada"]:
    for key, value in canal_entrada_d.items():
        if key == item:
            df_train.loc[count, "canal_entrada"] = value
            count=count+1
            break
print("wow succesfull!")
'''
df_train
df_train["canal_entrada"].unique()
'''
df_train["canal_entrada"] = df_train["canal_entrada"].astype('int')
df_test["canal_entrada"] = df_test["canal_entrada"].astype('int')
'''
count_nan = len(df_train) - df_train.count()
count_nan
list(df_train.select_dtypes(include=['O']))
df_train["antiguedad"] = df_train["antiguedad"].astype('int')
df_test["antiguedad"] = df_test["antiguedad"].astype('int')
df_test["nomprov"].unique()
nomprov_d = {}
nomprov_l=['MALAGA', 'CIUDAD REAL', 'ZARAGOZA', 'TOLEDO', 'LEON', 'CACERES',
       'ZAMORA', 'SALAMANCA', 'HUESCA', 'AVILA', 'SEGOVIA', 'LUGO',
       'BARCELONA', 'MADRID', 'ALICANTE', 'SORIA', 'SEVILLA', 'CANTABRIA',
       'BALEARS, ILLES', 'VALLADOLID', 'PONTEVEDRA', 'VALENCIA', 'TERUEL',
       'CORUÑA, A', 'OURENSE', 'JAEN', 'GIRONA', 'RIOJA, LA', 'ALBACETE',
       'BURGOS', 'MURCIA', 'CADIZ', 'BADAJOZ', 'CUENCA', 'ALMERIA',
       'GUADALAJARA', 'PALENCIA', 'CASTELLON', 'PALMAS, LAS', 'CORDOBA',
       'LERIDA', 'HUELVA', 'GRANADA', 'ASTURIAS',
       'SANTA CRUZ DE TENERIFE', 'MELILLA', 'TARRAGONA', 'CEUTA']
for i,item in enumerate(nomprov_l):
    nomprov_d[item] = i
nomprov_d
df_train["nomprov"] = df_train["nomprov"].map(nomprov_d)
df_test["nomprov"] = df_test["nomprov"].map(nomprov_d)
df_test["segmento"].unique()
segmento_replice = {
    '01 - TOP' :           1,
    '02 - PARTICULARES'  : 2,
    '03 - UNIVERSITARIO' : 3
}
df_train["segmento"] = df_train["segmento"].map(segmento_replice)
df_test["segmento"] = df_test["segmento"].map(segmento_replice)
print("WOW! AMAZING! POGNALEEEE!!!")
print("WOW! AMAZING! POGNALEEEE!!!")
print("WOW! AMAZING! POGNALEEEE!!!")
print("WOW! AMAZING! POGNALEEEE!!!")
print("WOW! AMAZING! POGNALEEEE!!!")
list(df_train.columns)
columns = ['ind_ahor_fin_ult1',
 'ind_aval_fin_ult1',
 'ind_cco_fin_ult1',
 'ind_cder_fin_ult1',
 'ind_cno_fin_ult1',
 'ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1',
 'ind_ctop_fin_ult1',
 'ind_ctpp_fin_ult1',
 'ind_deco_fin_ult1',
 'ind_deme_fin_ult1',
 'ind_dela_fin_ult1',
 'ind_ecue_fin_ult1',
 'ind_fond_fin_ult1',
 'ind_hip_fin_ult1',
 'ind_plan_fin_ult1',
 'ind_pres_fin_ult1',
 'ind_reca_fin_ult1',
 'ind_tjcr_fin_ult1',
 'ind_valo_fin_ult1',
 'ind_viv_fin_ult1',
 'ind_nomina_ult1',
 'ind_nom_pens_ult1',
 'ind_recibo_ult1']
train = df_train.drop(columns, axis='columns')
train
df_test
df_test.fillna(0)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, df_train[columns].values, test_size = 0.2, 
                                                  random_state = 669)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


params = {
    'n_estimators': 10,
    'max_depth': 10,
}

xgbc = XGBClassifier(**params)


ova_xgbc = OneVsRestClassifier(xgbc)
ova_xgbc.fit(X_train, y_train)
ova_preds = ova_xgbc.predict(X_val)
df_test["renta"]
preds= ova_xgbc.predict(df_test)
len(preds)
preds_array = []
for item in preds:
    for i, value in enumerate(item):
        if value == 1:
            preds_array.append(columns[i])
            break
for i in range(13161):
    preds_array.append(columns[0])
preds_array = np.array(preds_array)
type(preds)
type(preds_array)
len(preds_array)
929615-916454
db=pd.read_csv("/kaggle/input/santander-product-recommendation/sample_submission.csv.zip")
db
db['added_products'] = preds_array
db.to_csv("submission.csv", index = False)
