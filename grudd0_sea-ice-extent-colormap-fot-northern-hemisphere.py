%matplotlib inline
import pandas as pd       
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate,signal
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.size']=12
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.cm as cm 
df = pd.read_csv('../input/seaice.csv')
# get northern hemisphere data
df_north = df[df.hemisphere =='north']
# drop Missing and Source Data columns
df_north = df_north.drop(['Missing','Source Data'],axis=1)
# convert year, month,and day columns to datetime format
df_north['Date'] = pd.to_datetime(df_north[['Year','Month','Day']]) 
extent_meas = np.array(df_north.Extent)
df_north.index = df_north.Date
df_m = df_north.resample("M").mean()
df_m.Year = df_m.Year.astype(int)
df_m.Month = df_m.Month.astype(int)
df_m = df_m.drop('Day',axis=1)
# 1978 and 2017 are not full years
df_m = df_m[(df_m.Year > 1978) & (df_m.Year < 2017)]
# create index array to use to setup colormap
df_m.Year = df_m.Year - 1979
df_m.Month = df_m.Month - 1
z = np.zeros([12,38])
for m in range(0,12):
    ref = df_m[(df_m.Month == m) & (df_m.Year < 12)].Extent.mean()
    for y in range(0,38):
        temp = df_m[(df_m.Month == m) & (df_m.Year == y)]
        z[m,y] = temp.Extent - ref
fig,ax = plt.subplots(figsize=(16,8))
p = ax.pcolor(range(0,39), range(0,13), z, cmap=cm.RdBu, vmin=(z).min(), vmax=(z).max())
plt.title('Delta Sea Ice Extent, Reference Monthly Mean Averaged for Years 1979 to 1990')
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5],months)
years = ['','1980','','','','','1985','','','','','1990','','','','','1995','','','','','2000','','','','','2005','','','','','2010','','','','','2015']
plt.xticks(np.linspace(0.5,37.5,38),years)
cb = fig.colorbar(p)
cb.set_label('Delta Extent, 10^6 sq km', rotation=270, labelpad=25)