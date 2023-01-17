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
df1 = pd.read_csv('/kaggle/input/frenchpresidentialelection2017/French_Presidential_Election_2017_First_Round.csv')

df1.head()
df1.info()
df1.isnull().sum()
for col in df1.columns:

    if(df1[col].isnull().sum()!=0):

        if(df1.dtypes[col] == 'object'):

            df1[col].fillna(df1[col].value_counts().idxmax(), inplace = True)

        if(df1.dtypes[col] == 'int64'):

            df1[col].fillna(df1[col].mean(), inplace= True)

        if(df1.dtypes[col] == 'float64'):

            df1[col].fillna(df1[col].value_counts().idxmax(), inplace= True)
for col in df1.columns:

    if(df1.dtypes[col] == 'object'):

        from sklearn.preprocessing import LabelEncoder

        label=LabelEncoder()

        df1[col]=label.fit_transform(df1[col])