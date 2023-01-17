import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno
df = pd.read_csv('../input/top50spotify2019/top50.csv' , encoding="ISO-8859-1")
df.describe()
df.head()
df.tail()
df.shape
#Renaming the columns

df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness(dB)','Valence.':'Valence','Length.':'Length', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)

df.head()
numeric_feature = df.select_dtypes(include=[np.number])

numeric_feature.columns
df.info()
df.isnull().sum()
df.skew()
y = df['Energy']

plt.figure(1); plt.title('Johnson')

sns.distplot(y , kde=False , fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y , kde=False , fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y , kde=False , fit=st.lognorm)
correlation = numeric_feature.corr()

print(correlation['Energy'].sort_values(ascending =False))
f , ax = plt.subplots(figsize = (20,10))

sns.heatmap(correlation , square = True)
k  = 5

cols = correlation.nlargest(k,'Energy')['Energy'].index

print(cols)

cm = np.corrcoef(df[cols].values.T)

f ,ax = plt.subplots(figsize = (14,12))

sns.heatmap(cm , linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
sns.set()

columns = ['Energy','Loudness(dB)','Valence','Length','Liveness']

sns.pairplot(df[columns], size = 2 , kind = 'scatter' ,diag_kind ='kde')

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2 , ncols = 2 ,figsize = (15,10))

Energy_scatter_plot = pd.concat([df['Energy'],df['Loudness(dB)']],axis = 1)

sns.regplot(x='Loudness(dB)',y = 'Energy',data = Energy_scatter_plot,scatter= True, fit_reg=True, ax=ax1)



Valence_scatter_plot = pd.concat([df['Energy'],df['Valence']],axis = 1)

sns.regplot(x='Valence',y = 'Energy',data = Valence_scatter_plot,scatter= True, fit_reg=True, ax=ax2)



Length_scatter_plot = pd.concat([df['Energy'],df['Length']],axis = 1)

sns.regplot(x='Length',y = 'Energy',data = Length_scatter_plot,scatter= True, fit_reg=True, ax=ax3)



Length_scatter_plot = pd.concat([df['Energy'],df['Liveness']],axis = 1)

sns.regplot(x='Liveness',y = 'Energy',data = Length_scatter_plot,scatter= True, fit_reg=True, ax=ax4)
Artistname = df.pivot_table(index = 'Genre' , values = 'Energy' , aggfunc = np.median).sort_values('Energy' , ascending = False)

Artistname.plot(kind = 'bar' , color = 'blue')

plt.xlabel('Genre')

plt.ylabel('Energy')

plt.show()
var = 'Loudness(dB)'

data = pd.concat([df['Energy'] , df[var]], axis = 1)

f , ax = plt.subplots(figsize = (12,8))

fig = sns.boxplot(x = var ,y = 'Energy', data =data)

fig.axis (ymin = 0 , ymax = 100)