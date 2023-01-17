import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

file= ('../input/battles.csv')

data = pd.read_csv(file)

data['battle_size']=data['attacker_size'].fillna(0)+data['defender_size'].fillna(0)

data['defender_king']=data['defender_king'].fillna('NA')

data['attacker_king']=data['attacker_king'].fillna('NA')

data['attacker_outcome']=data['attacker_outcome'].fillna('NA')



data_win=data[['attacker_1', 'attacker_outcome']]



data_los=data_win.loc[data_win['attacker_outcome']=='loss']

data_win=data_win.loc[data_win['attacker_outcome']=='win']

data_win=data_win.groupby(['attacker_1'],as_index=False).count()

data_los=data_los.groupby(['attacker_1'],as_index=False).count()

data_win.rename(columns={'attacker_outcome':'wins'}, inplace=True)

data_los.rename(columns={'attacker_outcome':'loss'}, inplace=True)



data_com=pd.merge(data_win,data_los, on = 'attacker_1',how='outer')

data_com['wins']=data_com['wins'].fillna(0)

data_com['loss']=data_com['loss'].fillna(0)

data_com=data_com.sort('attacker_1')

Xunique1, X1 = np.unique(data_com['attacker_1'], return_inverse=True)





ax = data_com[['wins','loss']].plot(kind='bar')

ax.set(xticks=range(len(Xunique1)),xticklabels=Xunique1)

plt.show()