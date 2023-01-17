# Necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm

from tqdm.notebook import tqdm

from scipy import stats
# Import dataframes

df_train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")

df_test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
# Convert categorical labels into numerical values

categorical_columns = ['gender', 'apache_2_bodysystem', 'ethnicity', 'apache_3j_bodysystem', 

    'icu_admit_source', 'icu_stay_type', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'icu_type']



cat_labenc_mapping = {

    col: LabelEncoder()

    for col in categorical_columns

}



for col in tqdm(categorical_columns):

    df_train[col] = df_train[col].astype('str')

    cat_labenc_mapping[col] = cat_labenc_mapping[col].fit(

        np.unique(df_train[col].unique().tolist() + df_test[col].unique().tolist())

    )

    df_train[col] = cat_labenc_mapping[col].transform(df_train[col])

    

    df_test[col] = df_test[col].astype('str')

    df_test[col] = cat_labenc_mapping[col].transform(df_test[col])
predictives = ['height','diabetes_mellitus']

dependents = ['weight']



#Load in the data columns we need and drop NA rows

test = df_train[(predictives+dependents)].dropna()



#Add the intercept to the model

X2 = sm.add_constant(test[predictives])



#create regression object and fit it

estWeight = sm.OLS(test[dependents], X2).fit()

print(estWeight.summary())
cw = estWeight.params[0] # constant

h = estWeight.params[1] # height

db = estWeight.params[2] # diabetes mellitus



index = df_train['weight'].isna() & ~df_train['height'].isna() & ~df_train['diabetes_mellitus'].isna()

n = 0

for idx,row in df_train[index].iterrows():

    df_train.loc[idx,'weight'] = cw + df_train.loc[idx,'height'] * h + df_train.loc[idx,'diabetes_mellitus'] * db

    n+=1

print('Filled up '+str(n)+' weight values')
predictives = ['weight','gender','ethnicity']

dependents = ['height']



#Load in the data columns we need and drop NA rows

test = df_train[(predictives+dependents)].dropna()



#Add the intercept to the model

X2 = sm.add_constant(test[predictives])



#create regression object and fit it

estHeight = sm.OLS(test[dependents], X2).fit()

print(estHeight.summary())
ch = estHeight.params[0] # constant

w = estHeight.params[1] # weight

g = estHeight.params[2] # gender

e = estHeight.params[3] # ethnicity



index = df_train['height'].isna() & ~df_train['weight'].isna() & ~df_train['gender'].isna() & ~df_train['ethnicity'].isna()

n = 0

for idx,row in df_train[index].iterrows():

    df_train.loc[idx,'height'] = ch + df_train.loc[idx,'weight'] * w + df_train.loc[idx,'gender'] * g + df_train.loc[idx,'ethnicity'] * e

    n+=1

print('Filled up '+str(n)+' height values')
index = df_train['bmi'].isna()

for idx,row in df_train[index].iterrows():

    df_train.loc[idx,'bmi'] = df_train.loc[idx,'weight'] / (df_train.loc[idx,'height']/100)**2



print('Calculated '+str(len(df_train[index]))+' bmi values')