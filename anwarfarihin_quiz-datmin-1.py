# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler,KBinsDiscretizer



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('../input/cardiac-arrhythmia-database/data_arrhythmia.csv',sep=';' )
df
df=df[[ 'age', 'height', 'weight','qrs_duration','p-r_interval', 'q-t_interval', 't_interval', 'p_interval','qrs']]

df
feat=df.columns
scaler=StandardScaler()

df=df.rank(method='first')

display(df)

df[df.columns] =scaler.fit_transform(df[df.columns] )

display(df)
for i in range (len(feat)):

    df[feat[i] + '_e_w'] =pd.qcut(df[feat[i]],10,duplicates='drop')
df
for i in df.columns:

    print(df[i].value_counts())
for i in range (len(feat)):

    df[feat[i] + '_e_d'] =pd.cut(df[feat[i]],10,duplicates='drop')
df_ed=df.iloc[:,18:]
for i in df_ed.columns:

    print(df_ed[i].unique())
display(df)
df.to_csv('MuhammadAnwarFarihin_1706039635.csv',index=False)