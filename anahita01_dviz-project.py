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
df_agr_exp = pd.read_csv("../input/agricultural-exports/Agricultural_exports.csv")
df_agr_exp.head()
df_agr_imp = pd.read_csv("../input/agricultural-imports/Agricultural_imports.csv")
df_agr_imp
df_agr_merged = df_agr_exp.merge(df_agr_imp, left_on='Country Code', right_on='Country Code', how='outer')
df_agr_merged.tail()
frames = [df_agr_exp, df_agr_imp]
df_agr_concat = pd.concat(frames)
df_agr_concat.head()
df_agr_exp_usa = df_agr_exp.loc[df_agr_exp['Country Code'] == 'USA']
df_agr_exp_usa = df_agr_exp_usa.iloc[:, 10:30]
df_agr_exp_usa.head()
df_agr_exp_usa.iloc[0].plot(kind='line', ax=None)
