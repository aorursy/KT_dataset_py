# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df=pd.read_csv("../input/restaurant-scores-lives-standard.csv")

# Any results you write to the current directory are saved as output.
df.head()

df1=df.drop(['business_phone_number','business_address','business_postal_code','business_city','business_state','business_location'],axis=1)
df1.head()
df_score=df1[['business_latitude','business_longitude','inspection_score','risk_category']].copy()
df_score.isnull().sum(axis=0)
df_score = df_score[np.isfinite(df_score['business_latitude'])]
df_score = df_score[np.isfinite(df_score['inspection_score'])]
df_score= df_score[pd.notnull(df_score['risk_category'])]
df_score.isnull().sum(axis=0)
df_score['risk_category'].unique()
df_score = pd.get_dummies(df_score, prefix='risk_', columns=['risk_category']) 
df_score.head()
import seaborn as sns
import matplotlib.pyplot as pl

f, ax = pl.subplots(figsize=(10, 8))
corr = df_score.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)