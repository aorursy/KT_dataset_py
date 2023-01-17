import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_google= pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df_google.info()
plt.figure(figsize=(20,10))

sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )

df_google=df_google.fillna(value=df_google['Rating'].mean())

plt.figure(figsize=(20,10))

sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )

df=df_google.to_csv('Goog_out1.csv')

df_google['Reviews']=df_google['Reviews'].str.replace('M','000000').astype(float)

df_google.to_csv('Goog_out1.csv')

df= pd.read_csv('Goog_out1.csv')

df

#Save it
df_google[df_google['Size']=='Varies with device']=0.0

df_google['Size']=df_google['Size'].str.replace('M','000000')

df_google['Size']=df_google['Size'].str.replace('+','')

df_google['Size']=df_google['Size'].str.replace(',','')

df_google['Size']=df_google['Size'].str.replace('k','000').astype(float)

df_google.to_csv('Goog_out1.csv')

df= pd.read_csv('Goog_out1.csv')

df

#Save it
df_google[df_google['Installs']=='Free']=0.0

df_google['Installs']=df_google['Installs'].str.replace('+','')

df_google['Installs']=df_google['Installs'].str.replace(',','')

df_google['Installs']=df_google['Installs'].astype(float)



df_google.to_csv('Goog_out1.csv')

df= pd.read_csv('Goog_out1.csv')

df.info()
