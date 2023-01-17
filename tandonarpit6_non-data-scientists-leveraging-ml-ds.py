# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
surveySchema=pd.read_csv('../input/SurveySchema.csv')
freeFormResponses=pd.read_csv('../input/freeFormResponses.csv')
multipleChoiceResponses=pd.read_csv('../input/multipleChoiceResponses.csv')
df_temp=multipleChoiceResponses['Q2']
df_temp.value_counts(normalize=True).plot(kind='bar',figsize=(8,8))
df_temp=multipleChoiceResponses['Q3']
df_temp.value_counts(normalize=True).plot(kind='bar',figsize=(20,10))
df_temp=multipleChoiceResponses['Q4']
df_temp.value_counts(normalize=True).plot(kind='barh')
df_temp=multipleChoiceResponses['Q5']
df_temp.value_counts(normalize=True).plot(kind='barh')
df_temp=multipleChoiceResponses['Q6']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(8,8))
df_temp=multipleChoiceResponses['Q7']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(8,8))
df_temp=multipleChoiceResponses.loc[:,'Q11_Part_1':'Q11_Part_7']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Selected Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(4,4))
df_temp=multipleChoiceResponses['Q23']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses['Q26']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses.loc[:,'Q31_Part_1':'Q31_Part_12']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Selected Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses.loc[:,'Q33_Part_1':'Q33_Part_11']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Selected Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses.loc[:,'Q34_Part_1':'Q34_Part_6']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="-",expand=True)
df_temp.columns=[column_names[1]]

df_temp=df_temp.iloc[1:]
df_temp=df_temp.apply(pd.to_numeric)
df_temp.mean().plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses.loc[:,'Q35_Part_1':'Q35_Part_6']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="-",expand=True)
df_temp.columns=[column_names[1]]

df_temp=df_temp.iloc[1:]
df_temp=df_temp.apply(pd.to_numeric)
df_temp.mean().plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses['Q39_Part_1']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses['Q39_Part_2']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses['Q40']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses.loc[:,'Q42_Part_1':'Q42_Part_5']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(6,6))
df_temp=multipleChoiceResponses['Q48']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))