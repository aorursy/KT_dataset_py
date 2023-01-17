# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer



from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

rndval = 42
df_train = pd.read_csv('/kaggle/input/loan-pred/train_ctrUa4K.csv')

df_test = pd.read_csv('/kaggle/input/loan-pred/test_lAUu6dG.csv')
df_train.isnull().sum()
df_train.nunique()
df_train.info()
df_train.columns
impute_feature_num = ['Dependents', 'LoanAmount', 'Loan_Amount_Term']

impute_feature_cat= ['Gender', 'Married', 'Self_Employed', 'Credit_History']
df_train[impute_feature_cat].head()
simple_imputer_cat = SimpleImputer(strategy='most_frequent')

simple_imputer_num = SimpleImputer(strategy='mean')



def first_stage(df):

    df = df.copy()

    df['Dependents'] = df['Dependents'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)

    return df



def second_simple_impute(df, imputer, cols, test=False):

    df = df.copy()

    if not test:

        imputer.fit(df[cols])

    df[cols] = imputer.transform(df[cols])

    return df



def third_fix_data(df):

    df = df.copy()

    df['Dependents'] = df['Dependents'].astype(int)

    return df
iterative_imputer = IterativeImputer(max_iter=10,random_state=rndval,

                                     initial_strategy='most_frequent',imputation_order='ascending')
pp_train = first_stage(df_train)

pp_train = second_simple_impute(pp_train, simple_imputer_cat, impute_feature_cat)

pp_train = second_simple_impute(pp_train, simple_imputer_num, impute_feature_num)

pp_train = third_fix_data(pp_train)
#using iterative imputer

#pp_train = first_stage(df_train)

#pp_train = second_simple_impute(pp_train, simple_imputer_cat, impute_feature_cat)

#pp_train = second_simple_impute(pp_train, iterative_imputer, impute_feature_num)

#pp_train = third_fix_data(pp_train)
#pp_test = first_stage(df_test)

#pp_test = second_simple_impute(pp_test, simple_imputer_cat, impute_feature_cat,test=True)

#pp_test = second_simple_impute(pp_test, simple_imputer_num, impute_feature_num,test=True)

#pp_test = third_fix_data(pp_test)
pp_train.isnull().sum()
pp_test.isnull().sum()
pp_train.head()
pp_test.head()
df_train.columns
encode_features_ord = ['Gender', 'Married', 'Education', 'Self_Employed']

encode_features_ohe = ['Property_Area']
orde = OrdinalEncoder()

ohe = OneHotEncoder(dtype=np.int8)



def fourth_encode(df, encoder, cols, test=False):

    df = df.copy()

    if not test:

        encoder.fit(df[cols])

    df[cols] = encoder.transform(df[cols])

    return df

    

def fourth_encode_ohe(df, encoder, cols, test=False):

    df = df.copy()

    if not test:

        encoder.fit(df[cols])

    ohe_data = encoder.transform(df[cols]).toarray()

    ohe_cols = [f"Property_Area_{item}" for item in list(encoder.categories_[0])]

    df = df.drop(cols, axis=1)

    return pd.concat([df, pd.DataFrame(ohe_data, columns=ohe_cols)], axis=1)
pp_train = fourth_encode(pp_train, orde, encode_features_ord)

pp_train = fourth_encode_ohe(pp_train, ohe, encode_features_ohe)
#pp_test = fourth_encode(pp_test,orde,encode_features_ord,test=True)

#pp_test = fourth_encode_ohe(pp_test,ohe,encode_features_ohe,test=True)
pp_train.head()

pp_train.dtypes
#pp_test.head()
featureCols = ['Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',

       'Loan_Amount_Term', 'Credit_History']

lable = 'Loan_Status'

x_train,x_valid,y_train,y_valid = train_test_split(pp_train[featureCols],pp_train[lable],test_size=0.10,random_state=rndval)
model = LogisticRegression(random_state=rndval, solver='lbfgs', max_iter=5000)

model.fit(x_train,y_train)
model.coef_
model.score(x_valid,y_valid)