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
df_LABS = pd.read_csv('../input/labs.csv')
df_QUES = pd.read_csv('../input/questionnaire.csv')
df_DEMO = pd.read_csv('../input/demographic.csv')
#df_DIET = pd.read_csv('../input/diet.csv')
df_EXAM = pd.read_csv('../input/examination.csv')
labs_p = df_LABS[[not x for x in df_LABS.URXPREG.isna()]][['SEQN','URXPREG']]
labs_p['URXPREG'].describe()
exam_p = df_EXAM[[not x for x in df_EXAM.CSQ241.isna()]][['SEQN','CSQ241']]
exam_p['CSQ241'].describe()
demo_p = df_DEMO[[not x for x in df_DEMO.RIDEXPRG.isna()]][['SEQN','RIDEXPRG']]
demo_p['RIDEXPRG'].describe()
merged_df = labs_p.merge(exam_p.merge(demo_p,on='SEQN'), on='SEQN')
merged_df
import seaborn as sns
sns.pairplot(merged_df[[col for col in merged_df.columns if col != 'SEQN']])

