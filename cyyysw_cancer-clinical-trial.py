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
import matplotlib.pyplot as plt
trial_0=pd.read_csv('../input/labeledEligibilitySample1000000.csv')
trial=pd.read_table('../input/labeledEligibilitySample1000000.csv', header=None)
trial.shape
trial.sample(10)
clin_trial = pd.DataFrame(np.array(trial).reshape(1000000,2), columns=['label', 'describe'])
clin_trial['describe'].head(10)
clin_trial['label'].unique()
clin_trial['study'], clin_trial['condition'] = clin_trial['describe'].str.split('.', 1).str
clin_trial.head(10)
clin_trial=clin_trial.drop(['describe'], axis=1)
clin_trial['qualification']=clin_trial['label'].str.extract('(\d)', expand=True)
clin_trial=clin_trial.drop(['label'], axis=1)
clin_trial.sample(5)
clin_trial['study'].value_counts()
clintrial_lymphoma=clin_trial.loc[clin_trial.condition.str.contains('\w*lymphoma')]
clintrial_lymphoma.shape
clintrial_breast=clin_trial.loc[clin_trial.condition.str.contains('.*reast')] # to avoid lower case and upper case error
clintrial_breast.shape
clintrial_lymphoma['study'].value_counts()
clintrial_lymphoma['words'] = clintrial_lymphoma.condition.str.split(' ')
clintrial_lymphoma.head(3)
rows = list()
for row in clintrial_lymphoma[['study', 'words']].iterrows():
    r = row[1]
    for word in r.words:
        rows.append((r.study, word))

words = pd.DataFrame(rows, columns=['study', 'word'])
words.head(10)
words['word'] = words.word.str.lower()
words['word'].value_counts().head(50)
clintrial_lymphoma['recurrent'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*recurrent"), "recurrent","no")
clintrial_lymphoma.sample(10)
clintrial_lymphoma['stage_ii'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*stage ii"), "stage_ii","no")
clintrial_lymphoma['stage_iii'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*stage iii"), "stage_iii","no")
clintrial_lymphoma['stage_iv'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*stage iv"), "stage_iv","no")
clintrial_lymphoma['follicular'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*follicular"), "follicular","no")
clintrial_lymphoma['diffuse'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*diffuse"), "diffuse","no")
clintrial_lymphoma['hodgkin'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*hodgkin"), "hodgkin","no")
clintrial_lymphoma.sample(50)
clintrial_lymphoma_select=clintrial_lymphoma.drop([ 'words'], axis=1)
import seaborn as sns

var = clintrial_lymphoma_select.groupby(['study']).qualification.value_counts()
var.shape
var.head(10)
var_q = clintrial_lymphoma_select.groupby('study')['qualification'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_q
var_r = clintrial_lymphoma_select.groupby('study')['recurrent'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_r
var_iv = clintrial_lymphoma_select.groupby('study')['stage_iv'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_iv
var_ii = clintrial_lymphoma_select.groupby('study')['stage_ii'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_ii

var1 = clintrial_lymphoma_select.groupby(['study']).qualification.value_counts()

var1.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], figsize=(15,6))
