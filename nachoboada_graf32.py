# Start with loading all necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/trainparadatos/train.csv", index_col=0)
df.head(2).T
radar_promedio=df[['antiguedad','precio','habitaciones', 'metroscubiertos','metrostotales','garages','banos']].fillna(0).agg('mean').astype('int64')

type(radar_promedio)

pd_radar_prom=pd.DataFrame(radar_promedio).T

pd_radar_prom.insert(0,'provincia','Promedio Nacional')

pd_radar_prom
radar_promedio_por_provincia=df[['antiguedad','precio','habitaciones', 'metroscubiertos','metrostotales','garages','banos','provincia']].groupby('provincia').mean().fillna(0)
radar_promedio_por_provincia1=radar_promedio_por_provincia.agg('mean').astype('int64')

pd_radar_prom_pro=pd.DataFrame(radar_promedio_por_provincia1).T

pd_radar_prom_pro.insert(0,'provincia','Promedio Provincial')

pd_radar_prom_pro.head(5)
pd_radar_menor_pro=radar_promedio_por_provincia.sort_values(by='precio').astype('int64').head(1).reset_index()

pd_radar_menor_pro
radar_mayor_por_provincia=radar_promedio_por_provincia.sort_values(by='precio').astype('int64').tail(1).reset_index()

radar_mayor_por_provincia
# Libraries

import matplotlib.pyplot as plt

import pandas as pd

from math import pi

 

# Set data

df=pd.concat([pd_radar_prom,pd_radar_prom_pro,pd_radar_menor_pro,radar_mayor_por_provincia], ignore_index='True')

 

 

 



categories=list(df)[1:]

N = len(categories)

 



angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 



ax = plt.subplot(111, polar=True)

 



ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 



plt.xticks(angles[:-1], categories)

 



ax.set_rlabel_position(0)

plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)

plt.ylim(0,40)

 



values=df.loc[0].drop('provincia').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Promedio Nacional")

ax.fill(angles, values, 'b', alpha=0.1)

 



values=df.loc[1].drop('provincia').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Promedio Provincial")

ax.fill(angles, values, 'r', alpha=0.1)





values=df.loc[2].drop('provincia').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Menor Promedio Provincial")

ax.fill(angles, values, 'g', alpha=0.1)





values=df.loc[3].drop('provincia').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Mayor Promedio Provincial")

ax.fill(angles, values, 'y', alpha=0.1)

 



plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1));