import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

import seaborn as sns
sns.set(style="whitegrid")
df = pd.read_csv("../input/Indicators.csv")
df_india_fc = df[(df.CountryName=='India')&(df.IndicatorCode=='AG.LND.FRST.K2')]
fig = plt.figure()
plt.plot(df_india_fc.Year,df_india_fc.Value,'o-',color='g')
plt.xlabel('Years')
plt.ylabel('forest area in sq. km')
plt.title('India forest cover area over time')
fig.savefig('forestarea.png')
df_india_fc_landperc = df[(df.CountryName=='India')&(df.IndicatorCode=='AG.LND.FRST.ZS')]
fig = plt.figure()
plt.plot(df_india_fc_landperc.Year,df_india_fc_landperc.Value,'o-',color='g')
plt.xlabel('Years')
plt.ylabel('forest area in sq. km')
plt.title('India forest cover area as percentage of land over time')
fig.savefig('forestcover_percetange_of_land.png')
