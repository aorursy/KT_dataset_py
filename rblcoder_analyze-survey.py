import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_codebook = pd.read_csv('/kaggle/input/developer-survey-2018/HackerRank-Developer-Survey-2018-Codebook.csv')

df_numeric_mapping = pd.read_csv('/kaggle/input/developer-survey-2018/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv')

df_values = pd.read_csv('/kaggle/input/developer-survey-2018/HackerRank-Developer-Survey-2018-Values.csv')

df_numeric = pd.read_csv('/kaggle/input/developer-survey-2018/HackerRank-Developer-Survey-2018-Numeric.csv')

pd.set_option('display.max_colwidth', -1)
#df_codebook = pd.read_csv('/kaggle/input/developer-survey-2018/HackerRank-Developer-Survey-2018-Codebook.csv')
print('df_codebook')

print(df_codebook.info())

print()

print('df_numeric_mapping')

print(df_numeric_mapping.info())

print()

print('df_numeric')

print(df_numeric.info())

print()

print('df_values')

print(df_values.info())
print('df_codebook')

print(df_codebook.sample().T)

print()

print('df_numeric_mapping')

print(df_numeric_mapping.sample().T)

print()

print('df_numeric')

print(df_numeric.sample().T)

print()

print('df_values')

print(df_values.sample().T)
df_codebook[49:70]
df_values.CountryNumeric2.value_counts()[:10]
list(df_values.columns)
df_values.q1AgeBeginCoding.value_counts()[:10]
df_values.q2Age.value_counts()
df_values.q9CurrentRole.value_counts()
pd.options.display.max_colwidth = -1

print(df_codebook[df_codebook['Data Field']=='q27EmergingTechSkill'].T)

df_values.q27EmergingTechSkill.value_counts()
print(df_codebook[df_codebook['Data Field']=='q8JobLevel'].T)

df_values.q8JobLevel.value_counts()
df_values.q9CurrentRole.value_counts()
df_values.q10Industry.value_counts()
df_numeric.loc[:,df_values.columns[25:38]].sum().sort_values(ascending=False)
df_values.q0012_other.dropna().sample(16)
# Based on your last job hunting experience, how did employers measure your skills

df_numeric.loc[:,df_values.columns[39:48]].sum().sort_values(ascending=False)
df_values.q0013_other.dropna().sample(10)
# Did you feel these were a good reflection of your abilities? 	

df_values.q14GoodReflecAbilities.value_counts()