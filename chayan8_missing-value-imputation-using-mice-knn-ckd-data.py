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
!pip install impyute
from impyute.imputation.cs import mice

from sklearn.preprocessing import OrdinalEncoder
df = pd.read_csv('../input/ckdisease/kidney_disease.csv')
cols_names={"bp":"blood_pressure",

          "sg":"specific_gravity",

          "al":"albumin",

          "su":"sugar",

          "rbc":"red_blood_cells",

          "pc":"pus_cell",

          "pcc":"pus_cell_clumps",

          "ba":"bacteria",

          "bgr":"blood_glucose_random",

          "bu":"blood_urea",

          "sc":"serum_creatinine",

          "sod":"sodium",

          "pot":"potassium",

          "hemo":"haemoglobin",

          "pcv":"packed_cell_volume",

          "wc":"white_blood_cell_count",

          "rc":"red_blood_cell_count",

          "htn":"hypertension",

          "dm":"diabetes_mellitus",

          "cad":"coronary_artery_disease",

          "appet":"appetite",

          "pe":"pedal_edema",

          "ane":"anemia"}



df.rename(columns=cols_names, inplace=True)
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')

df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df.drop(["id"],axis=1,inplace=True)
numerical_features = []

categorical_features = []



for i in df.drop('classification', axis=1).columns:

    if df[i].nunique()>7:

        numerical_features.append(i)

    else:

        categorical_features.append(i)
#Replace incorrect values

df['diabetes_mellitus'] = df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'})

df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')

df['classification'] = df['classification'].replace(to_replace = 'ckd\t', value = 'ckd')
df.loc[:,categorical_features].isnull().sum().sort_values(ascending=False)
df.loc[:,numerical_features].isnull().sum().sort_values(ascending=False)
to_encode = [feat for feat in categorical_features if df[feat].dtype=='object']
to_encode
ode = OrdinalEncoder(dtype = int)
def encode(data):

    '''function to encode non-nan data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = ode.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data



#create a for loop to iterate through each column in the data

for columns in to_encode:

    encode(df[columns])
df.loc[:, categorical_features].head(10)
X = df.drop('classification', axis=1)
X_train = X.loc[:300,]

X_test = X.loc[300:,]
# MICE requires float values

X_train_numerical = X_train.loc[:,numerical_features].astype('float64')
# Passing the numpy arrays to mice

X_train_numerical_imputed = mice(X_train_numerical.values)
X_train.loc[:,numerical_features].isna().sum().sort_values(ascending=False)
X_train.loc[:,numerical_features] = X_train_numerical_imputed
X_train.loc[:,numerical_features].isna().sum().sort_values(ascending=False)
from fancyimpute import KNN
imputer = KNN()
X_train_imputed = pd.DataFrame(np.round(imputer.fit_transform(X_train)),columns = X_train.columns)
X_train_imputed.isnull().sum()
X_train_imputed.describe().T
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train_imputed)

X_train_scaled = scaler.transform(X_train_imputed)
X_train_scaled = pd.DataFrame(data=X_train_scaled, columns = X_train.columns)
X_train_scaled.describe()
# MICE requires float values

X_test_numerical = X_test.loc[:,numerical_features].astype('float64')
X_test_numerical_imputed = mice(X_test_numerical.values)

X_test.loc[:,numerical_features] = X_test_numerical_imputed
X_test_imputed = pd.DataFrame(np.round(imputer.fit_transform(X_test)),columns = X_test.columns)
scaler.fit(X_test_imputed)

X_test_scaled = scaler.transform(X_test_imputed)
X_test_scaled = pd.DataFrame(data=X_test_scaled, columns = X_test.columns)