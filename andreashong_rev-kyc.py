import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



d = pd.read_csv("/kaggle/input/doc-reports/doc_reports.csv")

f = pd.read_csv("/kaggle/input/doc-reports/facial_similarity_reports.csv")
d.shape
d.head()
d.head()
f.shape
d.dtypes
f.dtypes
d.describe(include='O')
f.describe(include='O')
d.isnull().sum()
f.isnull().sum()
d=d.dropna(subset=['image_integrity_result'])

d
f=f.dropna(subset=['result'])


d.duplicated().sum()
f.duplicated().sum()
fd = pd.merge( d, f, on=['attempt_id', 'user_id'], suffixes=('_d', '_f'))

fd = fd.drop(['Unnamed: 0_f','Unnamed: 0_d'], axis=1)

fd.shape
fd.isnull().sum()
fd = fd.sort_values('created_at_d', ascending=True)

fd=fd.drop_duplicates(subset=['user_id','result_d','created_at_d','sub_result','properties_d','created_at_f','result_f'],keep='first')
fd.shape
fd.isnull().sum()