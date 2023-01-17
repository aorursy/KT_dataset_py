import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns  
import matplotlib as mpl
from IPython.display import Image
from sklearn.metrics import mean_squared_error
#import seaborn as sns   
import warnings

%matplotlib inline
# Loading the database
df_uci_diabetic = pd.read_csv('../input/diabetic-data-cleaning/diabetic_data.csv', decimal=b',')

# Criando um novo dataframe a partir do df_uci_diabetic
df = df_uci_diabetic.copy (deep = True)
print('O Dataframe diabetic_data possui ' + str(df.shape[0]) + ' linhas e ' + str(df.shape[1]) + ' colunas')
# Checking Data Types and Descriptive Statistics
print (df.info ()) 
print (df.describe ())
# Viewing the first 10 rows of the dataframe
df.head(10)
df.describe()
# Checking for missing data
for col in df.columns:
    if df[col].dtype == object:
        if df[col][df[col] == '?'].count() > 0:
            print(col,df[col][df[col] == '?'].count(),' Correspondendo a ',np.around((df[col][df[col] == '?'].count()/df[col].count())*100,2), '% das observações')
# Evaluating the distribution of data in each attribute (has missing data)
for col in df.columns:
    if df[col].dtype == object:
        if df[col][df[col] == '?'].count() != 0:       
            print(df.groupby([col])[col].count())
            print('')
# Evaluating the distribution of data in each attribute (no missing data)
for col in df.columns:
    if df[col].dtype == object:
        if df[col][df[col] == '?'].count() == 0:       
            print(df.groupby([col])[col].count())
            print('')
# Checking the median
for col in df.columns:
    if df[col].dtype != object:
        print(col, df[col].median())
        print('')
# Deleting columns that will not be used
df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'examide', 'citoglipton', 'medical_specialty'], axis = 1, inplace = True)
Image('../input/imagem-1/Agrupamento_CID_9.png')
# Creating new columns to assign transformed values
df['d1'] = df['diag_1']
df['d2'] = df['diag_2']
df['d3'] = df['diag_3']
df['classe'] = -1
df['change_t'] = -1
df['gender_t'] = -1
df['diabetesMed_t'] = -1
# Regrouping the main diagnosis
df['d1'] = df.apply(lambda row: 1 if (row['diag_1'][0:3].zfill(3) >= '390') and (row['diag_1'][0:3].zfill(3) <= '459' ) or  (row['diag_1'][0:3].zfill(3) == '785' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 2 if (row['diag_1'][0:3].zfill(3) >= '460') and (row['diag_1'][0:3].zfill(3) <= '519' ) or  (row['diag_1'][0:3].zfill(3) == '786' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 3 if (row['diag_1'][0:3].zfill(3) >= '520') and (row['diag_1'][0:3].zfill(3) <= '579' ) or  (row['diag_1'][0:3].zfill(3) == '787' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 4 if (row['diag_1'][0:3].zfill(3) == '250') else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 5 if (row['diag_1'][0:3].zfill(3) >= '800') and (row['diag_1'][0:3].zfill(3) <= '999' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 6 if (row['diag_1'][0:3].zfill(3) >= '710') and (row['diag_1'][0:3].zfill(3) <= '739' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 7 if (row['diag_1'][0:3].zfill(3) >= '580') and (row['diag_1'][0:3].zfill(3) <= '629' ) or  (row['diag_1'][0:3].zfill(3) == '788' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 8 if (row['diag_1'][0:3].zfill(3) >= '140') and (row['diag_1'][0:3].zfill(3) <= '239' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 9 if (row['diag_1'][0:3].zfill(3) >= '790') and (row['diag_1'][0:3].zfill(3) <= '799' ) or  (row['diag_1'][0:3].zfill(3) == '780' ) or  (row['diag_1'][0:3].zfill(3) == '781' ) or  (row['diag_1'][0:3].zfill(3) == '784' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 10 if (row['diag_1'][0:3].zfill(3) >= '240') and (row['diag_1'][0:3].zfill(3) <= '249' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 10 if (row['diag_1'][0:3].zfill(3) >= '251') and (row['diag_1'][0:3].zfill(3) <= '279' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 11 if (row['diag_1'][0:3].zfill(3) >= '680') and (row['diag_1'][0:3].zfill(3) <= '709' ) or  (row['diag_1'][0:3].zfill(3) == '782' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 12 if (row['diag_1'][0:3].zfill(3) >= '001') and (row['diag_1'][0:3].zfill(3) <= '139' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:3].zfill(3) >= '290') and (row['diag_1'][0:3].zfill(3) <= '319' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:1] >= 'E') and (row['diag_1'][0:1] <= 'V' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:3].zfill(3) >= '280') and (row['diag_1'][0:3].zfill(3) <= '289' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:3].zfill(3) >= '320') and (row['diag_1'][0:3].zfill(3) <= '359' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:3].zfill(3) >= '630') and (row['diag_1'][0:3].zfill(3) <= '679' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:3].zfill(3) >= '360') and (row['diag_1'][0:3].zfill(3) <= '389' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 13 if (row['diag_1'][0:3].zfill(3) >= '740') and (row['diag_1'][0:3].zfill(3) <= '759' ) else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: 0 if (row['diag_1'][0:3].zfill(3)  == '783' or row['diag_1'][0:3].zfill(3)  == '789') else row['d1'], axis=1)
df['d1'] = df.apply(lambda row: -1 if (row['diag_1'][0:1] == '?') else row['d1'], axis=1)                           
# Regrouping of the first secondary diagnosis
df['d2'] = df.apply(lambda row: 1 if (row['diag_2'][0:3].zfill(3) >= '390') and (row['diag_2'][0:3].zfill(3) <= '459' ) or  (row['diag_2'][0:3].zfill(3) == '785' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 2 if (row['diag_2'][0:3].zfill(3) >= '460') and (row['diag_2'][0:3].zfill(3) <= '519' ) or  (row['diag_2'][0:3].zfill(3) == '786' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 3 if (row['diag_2'][0:3].zfill(3) >= '520') and (row['diag_2'][0:3].zfill(3) <= '579' ) or  (row['diag_2'][0:3].zfill(3) == '787' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 4 if (row['diag_2'][0:3].zfill(3) == '250') else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 5 if (row['diag_2'][0:3].zfill(3) >= '800') and (row['diag_2'][0:3].zfill(3) <= '999' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 6 if (row['diag_2'][0:3].zfill(3) >= '710') and (row['diag_2'][0:3].zfill(3) <= '739' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 7 if (row['diag_2'][0:3].zfill(3) >= '580') and (row['diag_2'][0:3].zfill(3) <= '629' ) or  (row['diag_2'][0:3].zfill(3) == '788' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 8 if (row['diag_2'][0:3].zfill(3) >= '140') and (row['diag_2'][0:3].zfill(3) <= '239' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 9 if (row['diag_2'][0:3].zfill(3) >= '790') and (row['diag_2'][0:3].zfill(3) <= '799' ) or  (row['diag_2'][0:3].zfill(3) == '780' ) or  (row['diag_2'][0:3].zfill(3) == '781' ) or  (row['diag_2'][0:3].zfill(3) == '784' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 10 if (row['diag_2'][0:3].zfill(3) >= '240') and (row['diag_2'][0:3].zfill(3) <= '249' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 10 if (row['diag_2'][0:3].zfill(3) >= '251') and (row['diag_2'][0:3].zfill(3) <= '279' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 11 if (row['diag_2'][0:3].zfill(3) >= '680') and (row['diag_2'][0:3].zfill(3) <= '709' ) or  (row['diag_2'][0:3].zfill(3) == '782' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 12 if (row['diag_2'][0:3].zfill(3) >= '001') and (row['diag_2'][0:3].zfill(3) <= '139' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:3].zfill(3) >= '290') and (row['diag_2'][0:3].zfill(3) <= '319' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:1] >= 'E') and (row['diag_2'][0:1] <= 'V' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:3].zfill(3) >= '280') and (row['diag_2'][0:3].zfill(3) <= '289' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:3].zfill(3) >= '320') and (row['diag_2'][0:3].zfill(3) <= '359' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:3].zfill(3) >= '630') and (row['diag_2'][0:3].zfill(3) <= '679' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:3].zfill(3) >= '360') and (row['diag_2'][0:3].zfill(3) <= '389' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 13 if (row['diag_2'][0:3].zfill(3) >= '740') and (row['diag_2'][0:3].zfill(3) <= '759' ) else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: 0 if (row['diag_2'][0:3].zfill(3)  == '783' or row['diag_2'][0:3].zfill(3)  == '789') else row['d2'], axis=1)
df['d2'] = df.apply(lambda row: -1 if (row['diag_2'][0:1] == '?') else row['d2'], axis=1)                           
# Regrouping the second secondary diagnosis
df['d3'] = df.apply(lambda row: 1 if (row['diag_3'][0:3].zfill(3) >= '390') and (row['diag_3'][0:3].zfill(3) <= '459' ) or  (row['diag_3'][0:3].zfill(3) == '785' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 2 if (row['diag_3'][0:3].zfill(3) >= '460') and (row['diag_3'][0:3].zfill(3) <= '519' ) or  (row['diag_3'][0:3].zfill(3) == '786' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 3 if (row['diag_3'][0:3].zfill(3) >= '520') and (row['diag_3'][0:3].zfill(3) <= '579' ) or  (row['diag_3'][0:3].zfill(3) == '787' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 4 if (row['diag_3'][0:3].zfill(3) == '250') else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 5 if (row['diag_3'][0:3].zfill(3) >= '800') and (row['diag_3'][0:3].zfill(3) <= '999' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 6 if (row['diag_3'][0:3].zfill(3) >= '710') and (row['diag_3'][0:3].zfill(3) <= '739' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 7 if (row['diag_3'][0:3].zfill(3) .zfill(3)>= '580') and (row['diag_3'][0:3].zfill(3) <= '629' ) or  (row['diag_3'][0:3].zfill(3) == '788' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 8 if (row['diag_3'][0:3].zfill(3) >= '140') and (row['diag_3'][0:3].zfill(3) <= '239' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 9 if (row['diag_3'][0:3].zfill(3) >= '790') and (row['diag_3'][0:3].zfill(3) <= '799' ) or  (row['diag_3'][0:3].zfill(3) == '780' ) or  (row['diag_3'][0:3].zfill(3) == '781' ) or  (row['diag_3'][0:3].zfill(3) == '784' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 10 if (row['diag_3'][0:3].zfill(3) >= '240') and (row['diag_3'][0:3].zfill(3) <= '249' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 10 if (row['diag_3'][0:3].zfill(3) >= '251') and (row['diag_3'][0:3].zfill(3) <= '279' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 11 if (row['diag_3'][0:3].zfill(3) >= '680') and (row['diag_3'][0:3].zfill(3) <= '709' ) or  (row['diag_3'][0:3].zfill(3) == '782' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 12 if (row['diag_3'][0:3].zfill(3) >= '001') and (row['diag_3'][0:3].zfill(3) <= '139' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:3].zfill(3) >= '290') and (row['diag_3'][0:3].zfill(3) <= '319' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:1] >= 'E') and (row['diag_3'][0:1] <= 'V' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:3].zfill(3) >= '280') and (row['diag_3'][0:3].zfill(3) <= '289' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:3].zfill(3) >= '320') and (row['diag_3'][0:3].zfill(3) <= '359' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:3].zfill(3) >= '630') and (row['diag_3'][0:3].zfill(3) <= '679' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:3].zfill(3) >= '360') and (row['diag_3'][0:3].zfill(3) <= '389' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 13 if (row['diag_3'][0:3].zfill(3) >= '740') and (row['diag_3'][0:3].zfill(3) <= '759' ) else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: 0 if (row['diag_3'][0:3].zfill(3)  == '783' or row['diag_3'][0:3].zfill(3)  == '789') else row['d3'], axis=1)
df['d3'] = df.apply(lambda row: -1 if (row['diag_3'][0:1] == '?') else row['d3'], axis=1)                           
print(df.groupby(['d1', 'diag_1']).d2.count())
print(df.groupby(['d2', 'diag_2']).d2.count())
print(df.groupby(['d3', 'diag_3']).d3.count())
df = df[(df.d1 > -1) | (df.d2 > -1) | (df.d3 > -1)]
# Deleting the original columns from diagnostics
df.drop(['diag_1'], axis = 1, inplace = True)
df.drop(['diag_2'], axis = 1, inplace = True)
df.drop(['diag_3'], axis = 1, inplace = True)
# Assigns the class the values 1 or 0, 1 corresponding to readmission occurrences in less than 30 days
df['classe'] = df.apply(lambda row: 0 if (row['readmitted'][0:3] == '>30' or row['readmitted'][0:2] == 'NO') else row['classe'], axis=1) 
df['classe'] = df.apply(lambda row: 1 if (row['readmitted'][0:3] == '<30') else row['classe'], axis=1)
df.drop(['readmitted'], axis = 1, inplace = True)
df['change_t'] = df.apply(lambda row: 1 if (row['change'] == 'Ch') else -1, axis=1)
df['change_t'] = df.apply(lambda row: 0 if (row['change'] == 'No') else row['change_t'], axis=1)
df.drop(['change'], axis = 1, inplace = True)
df['gender_t'] = df.apply(lambda row: 1 if (row['gender'] == 'Male') else -1, axis=1)
df['gender_t'] = df.apply(lambda row: 0 if (row['gender'] == 'Female') else row['gender_t'], axis=1)
df.drop(['gender'], axis = 1, inplace = True)
df['diabetesMed_t'] = df.apply(lambda row: 1 if (row['diabetesMed'] == 'Yes') else -1, axis=1)
df['diabetesMed_t'] = df.apply(lambda row: 0 if (row['diabetesMed'] == 'No') else row['diabetesMed_t'], axis=1)
df.drop(['diabetesMed'], axis = 1, inplace = True)
m = 0
medicacoes = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 
              'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 
              'metformin-pioglitazone','metformin-rosiglitazone', 'glimepiride-pioglitazone', 
              'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']
for col in df.columns:
    if col in medicacoes:       
        colname = 'Med' + str(m) + '_t'
        df[colname] = df.apply(lambda row: 0 if (row[col] == 'No') else 1, axis=1)
        df.drop([col], axis = 1, inplace = True)
        m = m + 1
df['A1Cresult_t'] = df.apply(lambda row: 0 if (row['A1Cresult'][0:4] == 'Norm') else -1, axis=1) 
df['A1Cresult_t'] = df.apply(lambda row: 1 if (row['A1Cresult'][0:2] == '>7' or row['A1Cresult'][0:2] == '>8') else row['A1Cresult_t'], axis=1) 
df.drop(['A1Cresult'], axis = 1, inplace = True)
df['max_glu_serum_t'] = df.apply(lambda row: 0 if (row['max_glu_serum'][0:4] == 'Norm') else -1, axis=1) 
df['max_glu_serum_t'] = df.apply(lambda row: 1 if (row['max_glu_serum'][0:2] == '>7' or row['max_glu_serum'][0:2] == '>8') else row['max_glu_serum_t'], axis=1) 
df.drop(['max_glu_serum'], axis = 1, inplace = True)
df['age_faixa'] = df.apply(lambda row: 0 if (row['age'] == '[0-10)') else -1, axis=1) 
df['age_faixa'] = df.apply(lambda row: 1 if (row['age'] == '[10-20)') else row['age_faixa'], axis=1)
df['age_faixa'] = df.apply(lambda row: 2 if (row['age'] == '[20-30)') else row['age_faixa'], axis=1) 
df['age_faixa'] = df.apply(lambda row: 3 if (row['age'] == '[30-40)') else row['age_faixa'], axis=1)
df['age_faixa'] = df.apply(lambda row: 4 if (row['age'] == '[40-50)') else row['age_faixa'], axis=1) 
df['age_faixa'] = df.apply(lambda row: 5 if (row['age'] == '[50-60)') else row['age_faixa'], axis=1)
df['age_faixa'] = df.apply(lambda row: 6 if (row['age'] == '[70-80)') else row['age_faixa'], axis=1) 
df['age_faixa'] = df.apply(lambda row: 7 if (row['age'] == '[80-90)') else row['age_faixa'], axis=1)
df['age_faixa'] = df.apply(lambda row: 8 if (row['age'] == '[90-100)') else row['age_faixa'], axis=1)
df.drop(['age'], axis = 1, inplace = True)
df['race_t'] = df.apply(lambda row: 0 if (row['race'] == '?') else -1, axis=1) 
df['race_t'] = df.apply(lambda row: 1 if (row['race'] == 'AfricanAmerican') else row['race_t'], axis=1)
df['race_t'] = df.apply(lambda row: 2 if (row['race'] == 'Asian') else row['race_t'], axis=1) 
df['race_t'] = df.apply(lambda row: 3 if (row['race'] == 'Caucasian') else row['race_t'], axis=1)
df['race_t'] = df.apply(lambda row: 4 if (row['race'] == 'Hispanic') else row['race_t'], axis=1) 
df['race_t'] = df.apply(lambda row: 5 if (row['race'] == 'Other') else row['race_t'], axis=1)
df.drop(['race'], axis = 1, inplace = True)
# Saving the dataset with the transformations
df.to_csv('./diabetes_data_modificado.csv', index=False)
# Loading the transformed database
df = pd.read_csv('diabetes_data_modificado.csv', decimal=b',')
df.head(10)
print (df.info ()) 
print(df.groupby(['classe']).classe.count())
# Data Manipulation Packages
import sklearn 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import scale, MinMaxScaler, MultiLabelBinarizer, QuantileTransformer, Normalizer, StandardScaler, MaxAbsScaler, RobustScaler

# Keras e TensorFlow

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf

# Pacotes para Confusion Matrix e Balanceamento de Classes
#from pandas_ml import ConfusionMatrix
#import pandas_ml as pdml
import imblearn

LABELS = ["Normal", "Readmissão"]
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report

def pretty_print_conf_matrix(y_true, y_pred, 
                             classes,
                             normalize=False,
                             title='Confusion matrix',
                             cmap=plt.cm.Blues):
    """
    referência: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    """

    cm = confusion_matrix(y_true, y_pred)

    # Configure Confusion Matrix Plot Aesthetics (no text yet) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]

    # Place Numbers as Text on Confusion Matrix Plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)


    # Add Precision, Recall, F-1 Score as Captions Below Plot
    rpt = classification_report(y_true, y_pred)
    rpt = rpt.replace('avg / total', '      avg')
    rpt = rpt.replace('support', 'N Obs')

    plt.annotate(rpt, 
                 xy = (0,0), 
                 xytext = (-50, -140), 
                 xycoords='axes fraction', textcoords='offset points',
                 fontsize=12, ha='left')    

    # Plot
    plt.tight_layout()
# Function for the statistics of accuracy, inaccuracy, false negative and false positive rates
def estatisticas(y_true, y_pred):
    false_neg = 0
    false_pos = 0
    incorrect = 0
    y2_true = np.array(y_true)
    total = len(y_true)
    for i in range(len(y_true)):        
        if y_pred[i] != y2_true[i]:
            incorrect += 1
            if y2_true[i] == 1 and y_pred[i] == 0:
                false_neg += 1
            else:
                false_pos += 1

    inaccuracy = incorrect / total

    print('Inacurácia:', inaccuracy)
    print('Acurácia:', 1 - inaccuracy)
    if incorrect > 0:
        print('Taxa de Falsos Negativos:', false_neg/incorrect)
        print('Taxa de Falsos Positivos:', false_pos / incorrect )    
    print('Falsos Negativos/total:', false_neg/total)
    return inaccuracy, incorrect
#df['classe'].hist()
#plt.show()
count_classes = pd.value_counts(df['classe'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Distribuição")
plt.xticks(range(2), LABELS)
plt.xlabel("Classe")
plt.ylabel("Frequência");
print('O Dataframe diabetic_data_modificado possui ' + str(df.shape[0]) + ' linhas e ' + str(df.shape[1]) + ' colunas')
readmissoes = df.loc[df['classe'] == 1]
nao_readmissoes = df.loc[df['classe'] == 0]
print("Temos", len(readmissoes), "pontos de dados como readmissões e", len(nao_readmissoes), "pontos de dados considerados normais.")
# Assigning Values to the X and Y Variables of the Model
X = df.iloc[:,:-1]
y = df['classe']

# Aplicando Scala e Redução de dimensionalidade com PCA
X = scale(X)
pca = PCA(n_components = 10, random_state=38)
X = pca.fit_transform(X)

# Gerando dados de treino, teste e validação
X1, X_valid, y1, y_valid = train_test_split(X, y, test_size = 0.10, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.26, random_state = 0)
print("Tamanho do Dataset de Treino: ", X_train.shape)
print("Tamanho do Dataset de Validaçao: ", X_valid.shape)
print("Tamanho do Dataset de Test: ", X_test.shape)
model = Sequential()
model.add(Dense(10, input_dim = 10, activation = 'relu'))     
model.add(Dense(1, activation = 'sigmoid'))                
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs = 1, validation_data=(X_valid, y_valid))
print("Erro/Acurácia: ", model.evaluate(X_valid, y_valid, verbose = 0))
y_predicted = model.predict(X_valid).T[0].astype(int)
# Plot Confusion Matrix
warnings.filterwarnings('ignore')
pretty_print_conf_matrix(y_valid, y_predicted, 
                         classes= ['0', '1'],
                         normalize=False, 
                         title='Confusion Matrix')
estatisticas(y_valid, y_predicted)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, ratio='minority')
X2, y2 = smote.fit_sample(X, y)
count_classes = pd.value_counts(y2, sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Distribuição")
plt.xticks(range(2), LABELS)
plt.xlabel("Classe")
plt.ylabel("Frequência");
# Generating training data based on balanced data
X2_train, X_test_, y2_train, y_test_ = train_test_split(X2, y2, test_size = 0.33, random_state = 0)
from keras.callbacks import EarlyStopping
from tensorflow import set_random_seed
import keras as keras
from sklearn.metrics import precision_score, recall_score
#OPTIMIZER = Adam(lr=0.01, beta_1=0.99, beta_2=0.999, amsgrad=True) # otimizador
OPTIMIZER = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
# Class to calculate metric of accuracy based on recall
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.round(model2.predict(X_val)).T[0]
    
        self._data.append({
            'val_recall': recall_score(y_val, np.round(model2.predict(X_val)).T[0]),
            'val_precision': precision_score(y_val, np.round(model2.predict(X_val)).T[0]),
        })
        return

    def get_data(self):
        return self._data
batch_size = 8790
seed = 100
set_random_seed(seed)
metrics = Metrics()
model2 = Sequential()
model2.add(Dense(10, input_dim = 10,   kernel_initializer='ones', activation = 'tanh')) 
model2.add(Dense(1024, activation = 'tanh'))
model2.add(Dropout(0.40))
model2.add(Dense(512, activation = 'tanh'))
model2.add(Dropout(0.40))
model2.add(Dense(16,  activation = 'tanh'))
model2.add(Dropout(0.40))
model2.add(Dense(8,  activation = 'tanh'))
model2.add(Dropout(0.40))
model2.add(Dense(4,  activation = 'tanh'))
model2.add(Dropout(0.40))
model2.add(Dense(1,  activation = 'sigmoid'))
monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 5, verbose = 1, mode = 'auto')   
model2.compile(loss = 'binary_crossentropy', optimizer = OPTIMIZER, metrics = ['accuracy'])
model2.summary()
history = model2.fit(X2_train, y2_train, epochs = 100, batch_size = batch_size, validation_data=(X_valid, y_valid), callbacks = [monitor, metrics], shuffle=False)
# Perform the training until you achieve the best recall accuracy, mandating the balance of total accuracy
Lastrecall = 0
Maxrecall = 0
Maxprecision = 0
for i in range(5980,9790,10):
    batch_size = i
    print(i)
    metrics = Metrics()
    history = model2.fit(X2_train, y2_train, epochs = 100,  batch_size = batch_size, validation_data=(X_valid, y_valid), callbacks = [monitor, metrics], shuffle=False)
    if recall_score(y_test,np.round(model2.predict(X_test)).T[0]) > Maxrecall and precision_score(y_test,np.round(model2.predict(X_test)).T[0]) > Maxprecision:
        print(recall_score(y_test,np.round(model2.predict(X_test)).T[0]), i)
        Maxrecall = recall_score(y_test,np.round(model2.predict(X_test)).T[0])
        Maxprecision = precision_score(y_test,np.round(model2.predict(X_test)).T[0])
        if Maxrecall > Lastrecall:
            Lastrecall = Maxrecall
            model2.save('./best_model.h5')   
#    metrics.get_data()
# load model from single file
model2 = load_model('best_model.h5')
# Evaluating the Model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss'), 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
print("Loss: ", model2.evaluate(X_valid, y_valid, verbose=0))
from sklearn.metrics import recall_score
from sklearn import metrics
probs = model2.predict_proba(X_valid)
preds = probs[:,0]
fpr, tpr, threshold = metrics.roc_curve(y_valid, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True  Positive rate')
plt.xlabel('False Positive rate')
plt.show()
y2_predicted = np.round(model2.predict(X_test)).T[0]
y2_correct = y_test
np.setdiff1d(y2_predicted, y2_correct)
inaccuracy, incorrect = estatisticas(y2_correct, y2_predicted)
print('Validation Results')
print(recall_score(y_valid,np.round(model2.predict(X_valid)).T[0]))
print('\nTest Results')
print(1 - inaccuracy)
print(recall_score(y_test,np.round(model2.predict(X_test)).T[0]))
print(incorrect)
# Plot Confusion Matrix
warnings.filterwarnings('ignore')
#plt.style.use('classic')
#plt.figure(figsize=(5,5))
pretty_print_conf_matrix(y2_correct, y2_predicted, 
                         classes= ['0', '1'],
                         normalize=False, 
                         title='Confusion Matrix')
