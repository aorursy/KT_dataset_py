import pandas as pd

import numpy as np

import matplotlib.pyplot as py

import seaborn as sb
df1 = pd.read_csv('../input/filles/novel_corona.csv')
y = df1[['age','recovered','death']]

y
f= y.groupby(['age']).count()

f.drop(f.index[0:9],0,inplace=True)

f
novel =pd.DataFrame(f).reset_index()

type(novel)
novel.plot(x='age',y='death')

py.show()
novel.plot(x='age',y='death',kind='box')

py.show()
df = pd.read_csv('../input/data-t/covid-19-tracker-canada.csv')

gf = df.drop(columns=['date','travel_history','id','source','province','city'])

x = gf.groupby(['age'])['confirmed_presumptive'].count()

x = gf.groupby(['age']).count()

x.drop(x.index[9:24],0,inplace=True)

b =pd.DataFrame(x).reset_index()

b.drop(b.index[9:24],0,inplace=True)
new = pd.read_csv('../input/deaths/novel_deaths.csv')

new['confirmed_cases'] = b['confirmed_presumptive']

# new.drop(b.index[:-1],0,inplace=True)

new = new[:-1]

new['Deaths'][0] = 0.40

new
new.plot(x='Age',y=['Deaths','confirmed_cases'],kind='bar')

py.show()
Image("../input/all-data/T_1.png")
Image("../input/all-data/T_2.png")
Image("../input/all-data/T_3.png")
Image("../input/all-data/T_4.png")
Image("../input/all-data/T_5.png")
Image("../input/all-data/TD_1.png")
Image("../input/all-data/TD_2.png")
Image("../input/all-data/TD_3.png")
Image("../input/all-data/TD_4.png")
Image("../input/all-data/TD_5.png")
Image("../input/all-data/TD_6.png")