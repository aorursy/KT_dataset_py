import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/us_perm_visas.csv')
df.info()
df.head()
print('Number of Entries:', len(df))

print('Number of Columns:', len(df.columns))
sum(df.count() < 150000)
df['employer_city'].value_counts().head(10)
df['employer_city'].value_counts().head(10).plot(kind='bar')
df['employer_name'].value_counts().head(10)
df['employer_name'].value_counts().head(10).plot(kind='bar')
df['case_status'].value_counts()
df.columns
df['country_of_citzenship'].value_counts().head(10)
df['country_of_citzenship'].value_counts().head(10).plot(kind='bar')
df['application_type'].value_counts()
df['application_type'].value_counts().plot(kind='bar')