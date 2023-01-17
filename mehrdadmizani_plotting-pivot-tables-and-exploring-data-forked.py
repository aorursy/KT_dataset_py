# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from matplotlib import style

import seaborn as sns

sns.set(style='ticks', palette='RdBu')

#sns.set(style='ticks', palette='Set2')

import pandas as pd

import numpy as np

import time

import datetime 

%matplotlib inline

import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 1000

from time import gmtime, strftime

Time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

import timeit

start = timeit.default_timer()

pd.options.display.max_rows = 100



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df=pd.read_csv("../input/HR_comma_sep.csv")



# Any results you write to the current directory are saved as output.
df_jobtype = pd.pivot_table(df,

                        values = ['satisfaction_level', 'last_evaluation'],

                        index = ['sales'],

                        columns = [],aggfunc=[np.mean], 

                        margins=True).fillna('')



cm = sns.light_palette("green", as_cmap=True)

df_jobtype.style.background_gradient(cmap=cm)
df_jobtype_salary = pd.pivot_table(df,

                        values = ['satisfaction_level', 'last_evaluation'],

                        index = ['sales', 'salary'],

                        columns = [],aggfunc=[np.mean], 

                        margins=True).fillna('')

cm = sns.light_palette("green", as_cmap=True)

df_jobtype_salary.style.background_gradient(cmap=cm)
df_jobtype_salary_prom = pd.pivot_table(df,

                        values = ['satisfaction_level', 'last_evaluation'],

                        index = ['sales','promotion_last_5years', 'salary'],

                        columns = [],aggfunc=[np.mean], 

                        margins=True).fillna('')



cm = sns.light_palette("green", as_cmap=True)

df_jobtype_salary_prom.style.background_gradient(cmap=cm)
df_jobtype_prom = pd.pivot_table(df,

                        values = ['satisfaction_level', 'last_evaluation'],

                        index = ['sales','promotion_last_5years'],

                        columns = [],aggfunc=[np.mean], 

                        margins=True).fillna('')



cm = sns.light_palette("green", as_cmap=True)

df_jobtype_prom.style.background_gradient(cmap=cm)
df_jobtype_salary_time = pd.pivot_table(df,

                        values = ['satisfaction_level', 'last_evaluation'],

                        index = ['sales','time_spend_company', 'salary'],

                        columns = [],aggfunc=[np.mean], 

                        margins=True).fillna('')

cm = sns.light_palette("green", as_cmap=True)

df_jobtype_salary_time.style.background_gradient(cmap=cm)
for i in set(df['sales']):

    aa= df[df['sales'].isin([i])]

    g = sns.factorplot(x='time_spend_company', y="satisfaction_level",data=aa, 

                   saturation=1, kind="box", col = 'left', row = 'sales',

                   ci=None, aspect=1, linewidth=1) 
for i in set(df['sales']):

    aa= df[df['sales'].isin([i])]

    g = sns.factorplot(x='time_spend_company', y="satisfaction_level",data=aa, 

                   saturation=1, kind="box", col = 'salary',row='sales', 

                   ci=None, aspect=0.75, linewidth=1) 
for i in set(df['sales']):

    aa= df[df['sales'].isin([i])]

    g = sns.factorplot(x='left', y="satisfaction_level",data=aa, 

                   saturation=1, kind="box", col = 'salary',row='sales', 

                   ci=None, aspect=.75, linewidth=1) 