#importando as bibiliotecas

#from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd 

import seaborn as sns

#%matplotlib inline

from math import pi
#leitura do dataset

df = pd.read_csv('../input/BlackFriday.csv', delimiter=',')

df.dataframeName = 'BlackFriday.csv'

df.head(5)
#informações das variaveis presente no dataset

df.info()
#verificando os dados faltantes

df.isnull().sum()
#correlação das variaveis presente no dataset

corr_matrix=df.corr()

sns.heatmap(corr_matrix, cmap='PuOr')
#plotando o gráfico violino em ordem vertical

fig = plt.figure(figsize=(14,16))

ax1 = fig.add_subplot(211)

sns.violinplot(x=df["Age"].sort_values(), y=df["Purchase"], data=df, ax=ax1)
#plotando o ranking dos 15 produtos mais comprados usando plot.bar

df3 = df.query('Purchase > 8')

df4 = df3['Product_ID'].value_counts().nlargest(15)

plt.figure(figsize=(12,6))

df4.sort_values().plot.bar();

plt.title('15 Produtos mais comprados')

plt.xlabel('Product')

plt.ylabel('Count')
#lst = []



#top5 = df.groupby('Occupation').size().sort_values(ascending=False).head(5).index



#for i in range(len(top5)):    

    #lst.append(df[df['Occupation'] == top5[i]].groupby(by=['Age','Occupation']).sum())



#df_o = pd.concat(lst)

#df_o = df_o.sort_values(by=['Age','Occupation'],ascending=True)

#df_o
# set width of bar

barWidth = 0.15

plt.figure(figsize=(14,8))

 

# set height of bar

bars1 = [19123048, 82053195, 307977307, 123192302, 36985625, 40729719, 15753615]

bars2 = [3386604, 32219653, 166002573, 82341827, 62357670, 40462833, 27781669]

bars3 = [1064948, 436518213, 200481491, 16207142, 1003794, 2254805, 0]

bars4 = [1413810, 18016292, 218826393, 174621962, 63176388, 52652456, 20575443]

bars5 = [364024, 39639188, 165670680, 96376007, 44714409, 27193546, 13282501]

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]

r5 = [x + barWidth for x in r4]

 

# Make the plot

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Oc. 0')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Oc. 01')

plt.bar(r3, bars3, color='#59606D', width=barWidth, edgecolor='white', label='Oc. 04')

plt.bar(r4, bars4, color='#ffcdd2', width=barWidth, edgecolor='white', label='Oc. 07')

plt.bar(r5, bars5, color='#A2D5F2', width=barWidth, edgecolor='white', label='Oc. 17')

 

# Add xticks on the middle of the group bars

plt.xlabel('Age', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])



# Create legend & Show graphicOccup. 01

plt.legend()

plt.show()

relation = df.query('Purchase > 9000')

plt.figure(figsize=(6,6))

sns.boxplot('Marital_Status', 'Occupation', data=relation);
#sumarizando os dados de acordo o estado civil e ocupação para compras maiores que nove mil.

df_r = df[df['Purchase'] > 9000].groupby(by=['Marital_Status','Occupation']).sum()

df_r
#plotando as 10 primeiras ocupações

df = pd.DataFrame({

'group': ['MS_0','MS_1'],

'oc0': [222911839, 157918745],

'oc1': [127186033, 114391594],

'oc2': [76202002, 60918947],

'oc3': [58181427, 38756954],

'oc4': [296998838, 107499096],

'oc5': [45163341, 25931250],

'oc6': [62812177, 50532147],

'oc7': [195218177, 152970433],

'oc8': [4065707, 5564040],

'oc9': [13916389, 15761645],

'oc10': [66269319, 4072157],

})

 

 

 

# ------- PARTE 1: Criando background

 

# números de variáveis

categories=list(df)[1:]

N = len(categories)

 

# dividindo o gráfico por número de variavel para visualização do ângulo de cada eixo

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

# Inicializando o spider plot

ax = plt.subplot(111, polar=True)

 

#  Plotando o primeiro eixo no topo

ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 

# Desenhando os eixos por variável + add labels também

plt.xticks(angles[:-1], categories)

 

# Desenhando ylabels

ax.set_rlabel_position(0)

plt.yticks([100000000,200000000], ["100000000","200000000"], color="grey", size=7)

plt.ylim(0,300000000)

 

 

# ------- PARTE 2: Adicionandos os plots

 

#Plot de cada indivíduo = cada linha dos dados

 

# Ind1

values=df.loc[0].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Status 0")

ax.fill(angles, values, 'b', alpha=0.1)

 

# Ind2

values=df.loc[1].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Status 1")

ax.fill(angles, values, 'r', alpha=0.1)

 

# Add legenda

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))





#plotando as ocupações de 11 a 20. 

df = pd.DataFrame({

'group': ['MS_0','MS_1'],

'occ11': [38431725, 25931580],

'occ12': [105833079, 92430705],

'oc13': [19553522, 24067236],

'occ14': [94383937, 67905380],

'occ15': [41681662, 33880011],

'occ16': [78949315, 67826420],

'occ17': [145955153, 115153896],

'occ18': [20121291, 17289691],

'occ19': [34696110, 7610250],

'occ20': [87466782, 80957748],

})

 

 

 

# ------- PARTE 1: Criando background

 

# números de variáveis

categories=list(df)[1:]

N = len(categories)

 

# dividindo o gráfico por número de variavel para visualização do ângulo de cada eixo

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

# Inicializando o spider plot

ax = plt.subplot(111, polar=True)

 

# Plotando o primeiro eixo no topo

ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 

# Desenhando os eixos por variável + add labels também

plt.xticks(angles[:-1], categories)

 

# Desenhando ylabels

ax.set_rlabel_position(0)

plt.yticks([100000000,200000000], ["100000000","200000000"], color="grey", size=7)

plt.ylim(0,300000000)



 

# ------- PARTE 2: Adicionando os plots

 

# Plot de cada indivíduo = cada linha dos dados

 

# Ind1

values=df.loc[0].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Status 0")

ax.fill(angles, values, 'b', alpha=0.1)

 

# Ind2

values=df.loc[1].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Status 1")

ax.fill(angles, values, 'r', alpha=0.1)

 

# Add legenda

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


