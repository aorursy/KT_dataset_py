import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/ecological-footprint/countries.csv')

df.head(10)
# The target contry

target_land = 'Austria'
# Other lands

others = list(df['Country'])

others.remove(target_land)



# EDA only for numerical columns and Fillna..

LST =  [i for i in df.columns if df[i].dtype =='float64' ]

df = df[['Country'] + LST]

df.set_index('Country', inplace=True)



for i in LST:

    df[i].fillna(0, inplace =True)

df
# Set size

cols =3

rows =len(df.columns)//cols + 1



fig, axs = plt.subplots(rows,cols,subplot_kw={'aspect':'equal'})

fig.set_size_inches(20,20)

explode = (0, 0.1)



for j,i in enumerate(df.columns):

    tgt_val = df.loc[target_land,i]

    ots_val = df.loc[others,i].sum()

    ratio = tgt_val/(tgt_val +ots_val)

    ratios = [ratio,1-ratio]

    labels = [target_land, 'Others']

    

    r = j //cols

    c = j %cols

    axs[r][c].pie(ratios, labels = labels, autopct='%1.1f%%', explode = explode,shadow=True, colors = ['Red','Yellow'])

    axs[r][c].set_title('Prop of '+ i)

    

axs[-1, -1].axis('off')    

plt.show()

# 'Biocapacity Deficit or Reverse' looks strange. It is due to minus value. 