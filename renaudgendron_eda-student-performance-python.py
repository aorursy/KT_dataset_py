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
df=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.describe()
df_new = df.rename(columns={'race/ethnicity': 'race_ethnicity','parental level of education': 'parental','test preparation course': 'test','math score': 'math','reading score': 'reading','writing score': 'writing'})
df_new
import matplotlib.pyplot as plt 
import seaborn as sns

from scipy import stats

pd.value_counts(df_new.gender).to_frame().reset_index()

boxprops = dict(linestyle='-', linewidth=3, color='v')
medianprops = dict(linestyle='-', linewidth=3, color='v')
df_new.boxplot( by='gender',column=['reading'],grid= False, showfliers=False, showmeans=True, boxprops=boxprops,medianprops=medianprops )

sns.boxplot(x='diagnosis', y='area_mean', data=df[["gender","reading"]])

df_newf=df_new[df_new.gender != 'male']
df_newm=df_new[df_new.gender != 'female']

stats.ttest_ind(df_newf.reading, df_newm.reading,equal_var=True)

import scipy.stats as stats

pd.value_counts(df_new.race_ethnicity).to_frame().reset_index()

df_new.boxplot( by='race_ethnicity',column=['math'],grid= False )

stats.f_oneway(df_new['math'][df_new['race_ethnicity'] == 'group A'],
               df_new['math'][df_new['race_ethnicity'] == 'group B'],
               df_new['math'][df_new['race_ethnicity'] == 'group C'],
               df_new['math'][df_new['race_ethnicity'] == 'group D'],
               df_new['math'][df_new['race_ethnicity'] == 'group E'],
               )
df_new.boxplot( by='lunch',column=['writing'],grid= False )

pd.value_counts(df_new.lunch).to_frame().reset_index()

df_newf=df_new[df_new.lunch != 'standard']
df_news=df_new[df_new.lunch != 'free/reduced']

stats.ttest_ind(df_newf.reading, df_news.reading,equal_var=False)
