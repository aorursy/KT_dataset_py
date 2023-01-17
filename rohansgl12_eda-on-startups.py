import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
pd.set_option('display.max_column',None)
df_ = pd.read_csv('../input/startup-investments-crunchbase/investments_VC.csv' , encoding = 'unicode_escape')
df_.shape
df_.describe()
df_.head()
df_.isnull().sum()
def year_group(row):
    if row['founded_year'] >= 1900 and row['founded_year'] <= 1925:
        row['founded_year_group'] = 'less_than_1925'
    elif row['founded_year'] > 1925 and row['founded_year'] <= 1950:
        row['founded_year_group'] = '1925_1950'
    elif row['founded_year'] > 1950 and row['founded_year'] <= 1975:
        row['founded_year_group'] = '1950_1975'
    elif row['founded_year'] > 1975 and row['founded_year'] <= 2000:
        row['founded_year_group'] = '1975_2000'
    elif row['founded_year'] > 2000:
        row['founded_year_group'] = '2000_2014'
    else:
        row['founded_year_group'] = ''
    return row['founded_year_group']

df_['founded_year_group'] =  df_.apply(year_group , axis = 1)
df_new = df_[df_['founded_year'] >= 2000]
df_new['founded_year'] = df_new['founded_year'].astype(int)
plt.figure(figsize = (16,7))
sns.countplot(x = 'founded_year', data = df_new)
plt.show()
plt.figure(figsize=(16,7))
sns.countplot(x =' market ', data = df_, order=df_[' market '].value_counts().iloc[:10].index)
plt.xticks(rotation=30)
plt.show()
plt.figure(figsize=(16,7))
g = sns.countplot(x ='country_code', data = df_, order=df_['country_code'].value_counts().iloc[:10].index)
plt.xticks(rotation=0)
plt.show()
df_USA = df_[(df_['country_code'] =='USA')]
plt.figure(figsize=(16,7))
g = sns.countplot(x ='state_code', data = df_USA, order=df_['state_code'].value_counts().iloc[:10].index)
plt.xticks(rotation=30)
plt.show()
plt.figure(figsize = (8,8))
df_.status.value_counts().plot(kind='pie',shadow=True, explode=(0, 0, 0.15), startangle=90,autopct='%1.1f%%')
plt.title('Status')
plt.show()