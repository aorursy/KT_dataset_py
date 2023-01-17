import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import numpy as np 

import seaborn as sns

from matplotlib import style

style.use('seaborn')
df = pd.read_csv('../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()
df.info()
df.drop(['Uniq Id','Crawl Timestamp'],axis = 1,inplace=True)
df.head(24)
print('No. of Columns : ',df.shape[1])

print('No. of Samples : ',df.shape[0])
df.isnull().any()
df.isnull().sum()
df = df.dropna()
sns.heatmap(df.isnull())
df.rename(columns = {'Job Experience Required':'JE'},inplace=True)
df['min_exp'] = df.JE.apply(lambda x : str(x).replace(x[1:],'') if 'yrs' in str(x) else str(x))



df['max_exp'] = df.JE.str.replace("(-).*","")



'''

df['max_exp'] = df.JE.apply(lambda x : str(x).replace(x[:4],'') if 'yrs' in str(x) else str(x))

df['max_exp'] = df.max_exp.apply(lambda x : str(x).replace(' yrs','') if 'yrs' in str(x) else str(x))

'''

pd.to_numeric(df['min_exp'],errors='coerce')

pd.to_numeric(df['max_exp'],errors='coerce')
df.Location = df.Location.str.replace("(,).*","")

df.Location = df.Location.str.replace("(/).*","")
df.drop(['JE'],1,inplace=True)


fig,ax = plt.subplots(figsize = (15,120))

sns.countplot(y='Location',data = df )

plt.show()

fig,ax = plt.subplots(figsize = (15,30))

sns.countplot(y='min_exp',data = df)

plt.show()
fig,ax = plt.subplots(figsize = (15,30))

sns.countplot(y='max_exp',data = df)

plt.show()