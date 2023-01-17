import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
import plotly.graph_objects as go
df=pd.read_csv('../input/mers-igs/MERS_IGS.csv')
df.head()
df['datetime'] = df['day'].map(str) + '-' + df['month'].map(str) + '-' + df['year'].map(str)
df['datetime']=pd.to_datetime(df['datetime'])
df.info()
df.head()
plt.figure(figsize=(10, 8))
plt.plot(df['year'], df['longitude'], 'b.', label = 'Longitude')
plt.plot(df['year'], df['latitude'], 'r.', label = 'Latitude')
plt.plot(df['year'],df['height'],'y.',label='Height')
plt.xlabel('Date'); plt.ylabel('Residual'); plt.title('Mers ISG Residual')
plt.legend();
cols_plot = ['longitude', 'latitude', 'height','datetime']
axes = df[cols_plot].plot(x='datetime',marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Residual')
    ax.set_xlabel('Time(Year)')
long_df=df.groupby('datetime')['longitude'].sum().reset_index()
long_df= long_df.set_index('datetime')
long_df.index
y = long_df['longitude']. resample ('MS'). mean ()
y.plot (figsize= (12, 6)) 
plt.ylabel('LONGİTUDE')
plt.xlabel('TIME')
plt.title('MERS IGS RESİDUAL')
plt.show()
lat_df=df.groupby('datetime')['latitude'].sum().reset_index()
lat_df=df.set_index('datetime')
y=lat_df['latitude'].resample('MS').mean()
y.plot(figsize=(12,6))
plt.ylabel('LATİTUDE')
plt.xlabel('TIME')
plt.title('MERS IGS RESİDUAL')
plt.show()
h_df=df.groupby('datetime')['height'].sum().reset_index()
h_df= h_df.set_index('datetime')
h_df.index
y = h_df['height']. resample ('MS'). mean ()
y.plot (figsize= (12, 6)) 
plt.ylabel('HEIGHT')
plt.xlabel('TIME')
plt.title('MERS IGS RESİDUAL')
plt.show()