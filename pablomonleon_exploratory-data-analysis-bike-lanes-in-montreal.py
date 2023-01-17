import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt  # importamos pyplot para hacer graficas
%matplotlib inline
plt.style.use(['fivethirtyeight'])
mpl.rcParams['lines.linewidth'] = 3
df = pd.read_csv('../input/comptagesvelo2015.csv', header=0,
                 sep=',', parse_dates=['Date'], dayfirst=True,
                 index_col='Date')

df.head()
df.shape
df = df.drop('Unnamed: 1', 1)
df.head(2)
df['Berri1'].plot(figsize=(12,6))
df.plot(figsize=(12,6))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
df.plot(kind='box', sym='gD', vert=False, xlim=(-100,11000))
df['Saint-Laurent U-Zelt Test'].plot(figsize=(12,6))
df[['Saint-Laurent U-Zelt Test','Maisonneuve_1','Pont_Jacques_Cartier','Parc U-Zelt Test']].plot(figsize=(12,6))
df[['Maisonneuve_3']].plot(figsize=(15,6))
df[df['Maisonneuve_3'] > 6000]
clean_df = df.dropna(axis=1, how='any')
clean_df = clean_df.drop('Maisonneuve_3',1)
