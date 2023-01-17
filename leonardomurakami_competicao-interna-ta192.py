import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
file_test = '../input/ta192/test.csv'

file_train = '../input/ta192/train.csv'
def count_values(df, column, old_dict = None):

    """Dado um dataframe e a coluna desejada, retorna um dicionario com a contagem de valores unicos."""

    if old_dict == None:

        dict_return = {}

    else:

        dict_return = old_dict

    for i in df[column]:

        if(i in dict_return.keys()):

            dict_return[i] += 1

        else:

            dict_return[i] = 1

    return dict_return
train = pd.read_csv(file_train)

test = pd.read_csv(file_test)
train.head()
train.rename(columns = {'default.payment.next.month':'NEXT_MONTH'}, inplace = True)

train.head()
train['SEX'] = train['SEX'].str.replace('male','m')

train['SEX'] = train['SEX'].str.replace('fem','f')
edu = train[['EDUCATION']]

education_counts = count_values(train,'EDUCATION')

sex_counts = count_values(train, 'SEX')

marriage_counts = count_values(train, 'MARRIAGE')
print(education_counts)

print(sex_counts)

print(marriage_counts)



train['MARRIAGE'] = train.MARRIAGE.apply(lambda x: str(x).lower())



marriage_counts_fixed = count_values(train, 'MARRIAGE')

print(marriage_counts_fixed)



age_counts_invalid = count_values(train[train['AGE'] > 120], 'AGE')

age_counts_invalid = count_values(train[train['AGE'] < 0], 'AGE', old_dict = age_counts_invalid)

print(age_counts_invalid)

train.AGE.replace(list(age_counts_invalid.keys()),np.nan, inplace = True)

train.AGE.replace(np.nan,train['AGE'].quantile(0.5), inplace = True)

train.describe()



age_counts = count_values(train, 'AGE')

print(train['AGE'].max())

train.AGE.hist(bins = 79)