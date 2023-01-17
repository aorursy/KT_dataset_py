import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/h1b_kaggle.csv')

df.dropna()

df.info()
%matplotlib inline

import matplotlib.pyplot as plt





df.groupby('YEAR').size().plot(title='Number of Applications from 2011 to 2016',kind='bar')
name_count = df.groupby('SOC_NAME').size()

top_10_title = name_count.sort_values(ascending=False)[:10]

print(top_10_title)
name_count = df.groupby('EMPLOYER_NAME').size()

top_10_EMPLOYER_NAME = name_count.sort_values(ascending=False)[:10]

print(top_10_EMPLOYER_NAME)
df[['CASE_STATUS','YEAR']].groupby(['CASE_STATUS','YEAR']).size().unstack().plot(kind='bar',title="Case Status from 2011 to 2016",figsize=(12,12))