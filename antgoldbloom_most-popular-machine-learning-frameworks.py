# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_mc = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

df_ide = pd.DataFrame(columns=['IDE','count','percentage'])



for i in range(1,13):

    df_ide = df_ide.append({'IDE':df_mc['Q28_Part_{}'.format(i)].mode()[0],'count':df_mc['Q28_Part_{}'.format(i)].count(),'percentage':df_mc['Q28_Part_{}'.format(i)].count()/len(df_mc[['Q28_Part_{}'.format(c) for c in range(1,13)]].dropna(how='all'))},ignore_index=True)



df_ide.index = df_ide['IDE']

del(df_ide['IDE'])

df_ide[['percentage']].sort_values(by='percentage',ascending = False)