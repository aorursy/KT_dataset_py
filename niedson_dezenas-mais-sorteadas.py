# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mega = pd.read_excel('/kaggle/input/lottery-br/megas.xls')

lotofacil = pd.read_excel('/kaggle/input/lottery-br/d_lotfac.xls')
mega.info()
mega.head()
mega['ball_01']
lotofacil.info()
lotofacil
# Vamos excluir da 17º coluna em diante.

lotofacil = lotofacil.drop(columns=lotofacil.columns[17:]) # love it
lotofacil
mega.head()
df = mega

count_draw1 = df['ball_01'].value_counts().to_dict()

count_draw2 = df['ball_02'].value_counts().to_dict()

count_draw3 = df['ball_03'].value_counts().to_dict()

count_draw4 = df['ball_04'].value_counts().to_dict()

count_draw5 = df['ball_05'].value_counts().to_dict()

count_draw6 = df['ball_06'].value_counts().to_dict()



powerball = pd.DataFrame([count_draw1, count_draw2, count_draw3, count_draw4, count_draw5, count_draw6]) 

powerball = powerball.sort_index(axis=1)

pw = powerball

pw = pw.T

pw.columns = ['total_draw_1','total_draw_2','total_draw_3','total_draw_4','total_draw_5','total_draw_6']

pw['SUPER_TOTAL'] = pw.sum(axis=1)

pw 
import matplotlib.pyplot as plt # https://matplotlib.org/



fig, ax = plt.subplots(figsize=(12, 24))



group_data = list(pw['SUPER_TOTAL'])

group_names = list(pw.keys())



pw['SUPER_TOTAL'].plot.barh(group_names, group_data)

plt.style.use('bmh')

ax.set(xlim=[180, 260], xlabel='Número de reptições', ylabel='Dezenas',title='Repetições')

 # Criando Dataframe com base na contagem de vezes que aparece cada número



c = pw.T

x01 = c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9] + c[10]

x11 = c[11] + c[12] + c[13] + c[14] + c[15] + c[16] + c[17] + c[18] + c[19] + c[20]

x21 = c[21] + c[22] + c[23] + c[24] + c[25] + c[26] + c[27] + c[28] + c[29] + c[30]

x31 = c[31] + c[32] + c[33] + c[34] + c[35] + c[36] + c[37] + c[38] + c[39] + c[40]

x41 = c[41] + c[42] + c[43] + c[44] + c[45] + c[46] + c[47] + c[48] + c[49] + c[50]

x51 = c[51] + c[52] + c[53] + c[54] + c[55] + c[56] + c[57] + c[58] + c[59] + c[60]



dezenas = pd.DataFrame([x01, x11, x21, x31, x41, x51])

dezenas = dezenas.T

dezenas.columns = ['n1-10','n11-20','n21-30','n31-40','n41-50','n51-60']

dezenas = dezenas.T

dezenas

fig, ax = plt.subplots(figsize=(6, 6))





group_data = list(dezenas['SUPER_TOTAL'])

group_names = list(dezenas.keys())



dezenas['SUPER_TOTAL'].plot.barh(group_names, group_data)

plt.style.use('bmh')

ax.set(xlim=[2180, 2280], xlabel='Número de reptições', ylabel='Dezenas',title='Repetições')


df1 = lotofacil

facil_01 = df1['ball_01'].value_counts().to_dict()

facil_02 = df1['ball_02'].value_counts().to_dict()

facil_03 = df1['ball_03'].value_counts().to_dict()

facil_04 = df1['ball_04'].value_counts().to_dict()

facil_05 = df1['ball_05'].value_counts().to_dict()

facil_06 = df1['ball_06'].value_counts().to_dict()

facil_07 = df1['ball_07'].value_counts().to_dict()

facil_08 = df1['ball_08'].value_counts().to_dict()

facil_09 = df1['ball_09'].value_counts().to_dict()

facil_10 = df1['ball_10'].value_counts().to_dict()

facil_11 = df1['ball_11'].value_counts().to_dict()

facil_12 = df1['ball_12'].value_counts().to_dict()

facil_13 = df1['ball_13'].value_counts().to_dict()

facil_14 = df1['ball_14'].value_counts().to_dict()

facil_15 = df1['ball_15'].value_counts().to_dict()





lotofacil = pd.DataFrame([facil_01, facil_02, facil_03, facil_04, facil_05, facil_06, facil_07, facil_08, facil_09, facil_10, facil_11, facil_12,facil_13, facil_14, facil_15]) 

lotofacil = lotofacil.sort_index(axis=1)



lotofacil = lotofacil.T

lotofacil.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# lotofacil['TOTAL'] = lotofacil.sum(axis=1)

lotofacil 
#Convertendo dataframe para matriz para que possamos ordenar os números.

lotofacil = lotofacil.to_numpy()

lotofacil.sort(axis=1)
# Agora  que ficou ordenado, vamos retornar para dataframe para que possamos manipular.

#lotofacil = pd.DataFrame({'Draw1': df2[:, 0], 'Draw2': df2[:, 1], 'Draw3': df2[:, 2], 'Draw4': df2[:, 3], 'Draw5': df2[:, 4], 'Draw6': df2[:, 5]})

lotofacil = pd.DataFrame({'Draw1': lotofacil[:, 0], 'Draw2': lotofacil[:, 1], 'Draw3': lotofacil[:, 2], 'Draw4': lotofacil[:, 3], 'Draw5': lotofacil[:, 4], 'Draw6': lotofacil[:, 5], 'Draw7': lotofacil[:, 6], 'Draw8': lotofacil[:, 6], 'Draw9': lotofacil[:, 7], 'Draw10': lotofacil[:, 9], 'Draw11': lotofacil[:, 10], 'Draw12': lotofacil[:, 11], 'Draw13': lotofacil[:, 12], 'Draw14': lotofacil[:, 13], 'Draw15': lotofacil[:, 14]})

new_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

lotofacil = lotofacil.reindex(new_index)

lotofacil['TOTAL'] = lotofacil.sum(axis=1)

lotofacil
fig, ax = plt.subplots(figsize=(12, 24))



group_data = list(lotofacil['TOTAL'])

group_names = list(lotofacil.keys())



lotofacil['TOTAL'].plot.barh(group_names, group_data)

plt.style.use('bmh')

ax.set(xlim=[1080, 1190], xlabel='Número de reptições', ylabel='Dezenas',title='Repetições')