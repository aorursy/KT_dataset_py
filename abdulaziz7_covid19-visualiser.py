import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
%matplotlib inline
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)
sns.set_style('darkgrid')
path = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"
df_original = pd.read_csv(path)
df = df_original.copy()
df.head(5)
df.describe()
df['Country/Region'].unique()
df.dtypes

df = df.sort_values(by=['Confirmed'],ascending=False)
df.head()
italy_df = df.loc[df['Country/Region'] == 'Italy']
italy_df.tail()
df.head()
df_percountry = df.groupby(
    ['Country/Region','ObservationDate']).agg(
    {'Confirmed': 'sum','Deaths': 'sum', 'Recovered': 'sum'})
df_percountry.head()
df_percountry = df_percountry.reset_index().sort_values(by=['ObservationDate'],ascending=False)
df_percountry.head(5)
country = 'Saudi Arabia'
df_singlecountry =  df_percountry.loc[df_percountry['Country/Region'] == country]
df_singlecountry.head()
data = df_singlecountry
x = 'ObservationDate'
y = 'Confirmed'
d = 'Deaths'
r = 'Recovered'
plt.figure(figsize=(15,8))
ax = sns.lineplot(x=data[x], y=data[y], err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X')

plt.show()
plt.figure(figsize=(15,8))
ax = sns.lineplot(x=data[x], y=np.log1p(data[y]), err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X [Log+1 Transformed]')

plt.show()
ids= ['ObservationDate', 'Country/Region']
values= ['Confirmed','Deaths','Recovered']
df_melted = pd.melt(df_singlecountry, id_vars=ids, value_vars=values)
df_melted.head()
data = df_melted
x = 'ObservationDate'
y = 'value'
z = 'variable'
plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, hue=z, err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X')

plt.show()
plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=np.log1p(data[y]), hue=z, err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in X')

plt.show()
uae_df = df.loc[df['Country/Region'] == 'United Arab Emirates']
uae_df.head()
uae_df.tail()
sns.set_style('darkgrid')
data = uae_df
x = 'ObservationDate'
y = 'Confirmed'
data = uae_df
x = 'ObservationDate'
y = 'Deaths'
plt.figure(figsize=(15,8))
ax = sns.lineplot(x=data[x], y=data[y], err_style='band')
ax = sns.lineplot(x=data[x], y=data['Recovered'])
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in the UAE')

plt.show()
plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, err_style='band')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in the UAE')

plt.show()
ids= ['ObservationDate', 'Country/Region']
values= ['Confirmed','Deaths','Recovered']
df_melted = pd.melt(uae_df, id_vars=ids, value_vars=values)
df_melted.head()
data = df_melted
x = 'ObservationDate'
y = 'value'
z = 'variable'
plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, hue=z)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases in the UAE')

plt.show()
df_melted['logvalue'] = np.log1p(df_melted['value'])
df_melted.head()
data = df_melted
x = 'ObservationDate'
y = 'logvalue'
z = 'variable'
plt.figure(figsize=(15,8))
ax = sns.lineplot(data=data, x=x, y=y, hue=z)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_title('Confirmed Covid19 cases/deaths/recovered in the UAE [Log+1 Transformed]')

plt.show()



df_percountry.unstack('Country/Region')
