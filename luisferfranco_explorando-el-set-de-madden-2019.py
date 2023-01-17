import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from math import pi

import math
df = pd.read_csv('/kaggle/input/madden-2019-ultimate-team-dataset/Madden_Ultimate_Team.csv', engine='python')
df.columns
qbattr = ['SPD', 'STR', 'AGI', 'AWR', 'ELU', 'THP', 'TAS', 'TAM', 'TAD', 'TOR', 'PAC', 'TUP']
# QB                     88%

# HB                     84%

# FB                     84%

# WR                     72%

# TE                     66%

# OL (LT, LG, C, RG, RT)

# DL (LE, DT RE)

# LB (LOLB, MLB, ROLB)

# DB (CB, SS, FS)

qb538 = ['AWR', 'THP', 'TAS', 'TAM', 'TAD', 'PAC']

hb538 = ['AWR', 'SPD', 'CAR', 'BCV', 'ACC', 'AGI', 'TRK', 'ELU']

fb538 = ['AWR', 'RBK', 'IBL', 'PBK', 'CAR', 'SPD', 'ACC', 'TRK']

wr538 = ['AWR', 'SPD', 'CTH', 'SRR', 'MRR', 'DRR', 'ACC', 'CIT', 'RLS']

te538 = ['AWR', 'RBK', 'CTH', 'SRR', 'MRR', 'DRR', 'IBL', 'SPD', 'STR']

ol538 = ['PBL', 'AWR', 'RBK', 'STR', 'IBL']

dl538 = ['AWR', 'FNM', 'PWM', 'ACC', 'SPD', 'TAK', 'BKS']

ln538 = ['AWR', 'TAK', 'PRC', 'BKS', 'PUR', 'FNM', 'PWM']

db538 = ['AWR', 'TAK', 'PRC', 'SPD', 'ACC', 'ZCV', 'MCV']

pk538 = ['KAC', 'AWR', 'KPW']

kk538 = ['KAC', 'AWR', 'KPW']
qb = df[df.Position == 'QB'].sort_values('Overall', ascending=False)
qb.columns
qb.columns[:9]
plt.style.use('ggplot')
qb.columns[9:-3]
def Compare(df, attr=[], figsize=(12, 10)):

    # Tamaño de la figura

    plt.figure(figsize=figsize)



    # Necesitamos al menos 3 características para una gráfica de radar

    # si hay menos, ignoramos la lista que se pasó y usamos todos los atributos

    # para el dataframe. En todo caso mantenemos siempre el Nombre

    if (len(attr) > 2):

        categories = attr

    else:

        categories = qb.columns[9:-3]

        

    # Las categorías son los nombres de las columnas de atributos

    N = len(categories)

    

    # Calcular los ángulos para cada uno de los ejes en la gráfica

    # plot/numero de variables

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    

    # Una gráfica polar (o radial) es una gráfica la cual se dobla

    # sobre un centro, se puede mostrar si se elimina el comentario

    # en el siguiente código...

    #

    # ax = plt.subplot(111, polar=False)

    #

    # ...y se comenta el siguiente bloque:



    #

    # Se inicializa el espacio para la gráfica

    #

    ax = plt.subplot(111, polar=True)

    # Si se quiere que el primer eje esté hacia arriba

    ax.set_theta_offset(pi / 2)

    ax.set_theta_direction(-1)

    #

    ax.set_rlabel_position(0)

 



    # Se dibuja el eje de las X con los nombres de las categorías (variables)

    plt.xticks(angles[:-1], categories)

 

    # Ejes Y

    ytick = [r for r in range(10, 101, 10)]

    yticklabel = [str(r) for r in ytick]



    plt.yticks(ytick, yticklabel, color="grey", size=12)

    plt.ylim(0,100)



    # Para cada elemento del dataframe que se pasó como parámetro...

    for index, row in df.iterrows():

        # Los valores se encuentran a partir de la segunda columna

        # (después del nombre)

        values=list(row[categories])



        # Como vamos a pintar un polígono, tenemos que regresar al

        # "origen"

        values += values[:1]

        

        ax.plot(angles, values, linewidth=1, linestyle='solid', label=row.Name)

        ax.fill(angles, values, alpha=0.1)



    plt.legend(loc='upper left')
Compare(qb.iloc[[0,1]], attr=qbattr)
Compare(qb.head(1).append(qb.tail(1)))
Compare(qb.iloc[[5, 43, 95, 180]], attr=qb538)
Compare(qb.iloc[[0,1,2]], attr=qbattr)
# Redefinimos la función para que use "facets" de tal forma que podamos

# revisar varias gráficas de lado a lado para hacer mejor las comparaciones



def Compare(df, attr=[], figsize=(12, 10), cols=3, c='steelblue', faucet=True, name=True):

    # Tamaño de la figura

    plt.figure(figsize=figsize)

    

    if faucet:

        # Calculamos el grid que vamos a utilizar

        #

        # Si hay solo un registro para comparar, lo ponemos en una sola columna

        if len(qb) == 1:

            cols = 1



        rens = math.ceil(len(df) / cols)

        inci = 1

    else:

        rens = 1

        cols = 1

        inci = 0



    # Necesitamos al menos 3 características para una gráfica de radar

    # si hay menos, ignoramos la lista que se pasó y usamos todos los atributos

    # para el dataframe. En todo caso mantenemos siempre el Nombre

    if (len(attr) > 2):

        categories=attr

    else:

        categories=list(qb.columns[9:-3])

    N = len(categories)

    

    # Calcular los ángulos para cada uno de los ejes en la gráfica

    # plot/numero de variables

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    

    # Inicializamos el índice para los faucets

    i = 1



    if not faucet:

        ax = plt.subplot(rens, cols, i, polar=True)

    

    # Para cada elemento del dataframe que se pasó como parámetro...

    for index, row in df.iterrows():

        #

        # Se inicializa el espacio para la gráfica

        #

        if faucet:

            ax = plt.subplot(rens, cols, i, polar=True)

        i += inci

        # Si se quiere que el primer eje esté hacia arriba

        ax.set_theta_offset(pi / 2)

        ax.set_theta_direction(-1)

        #

        ax.set_rlabel_position(0)



        # Se dibuja el eje de las X con los nombres de las categorías (variables)

        plt.xticks(angles[:-1], categories)

        

        # Ejes Y

        ytick = [r for r in range(10, 101, 10)]

        yticklabel = [str(r) for r in ytick]



        plt.yticks(ytick, yticklabel, color="grey", size=12)

        plt.ylim(0,100)



        ax.yaxis.grid(True,color='gray',linestyle='--')

        

        # Los valores se encuentran a partir de la segunda columna

        # (después del nombre)

        values=list(row[categories])

        # Como vamos a pintar un polígono, tenemos que regresar al

        # "origen"

        values += values[:1]

        

        if len(df)>1 and not faucet:

            ax.plot(angles, values, linewidth=1, linestyle='solid', label=row.Name)

            ax.fill(angles, values, alpha=0.1)

        else:

            ax.plot(angles, values, linewidth=1, linestyle='solid', label=row.Name, c=c)

            ax.fill(angles, values, alpha=0.1, c=c)



        if name:

            plt.text(0, 0, row.Name)

        

    plt.tight_layout()
Compare(qb.head(5))
Compare(qb.head(5), cols=5)
Compare(qb.head(4), cols=2, attr=qb538)
Compare(qb.head(12), cols=4, figsize=(18, 18), attr=qb538, c='orange')
Compare(qb.head(3).append(qb.tail(10)), faucet=False, attr=qb538, name=False)
plt.figure(figsize=(10,8))

sns.boxplot(data=qb[qb538])
Compare(df.loc[df.Position=='HB'][:3].append(df.loc[df.Position=='HB'][-3:]), faucet=False, name=False, 

        attr=['AWR', 'SPD', 'CAR', 'BCV', 'ACC', 'AGI', 'TRK', 'ELU'])
Compare(qb[0:3], attr=qbattr)
plt.figure(figsize=(16, 10))

sns.scatterplot(data=qb, x='Weight', y='Height', 

                size='Overall', hue='Overall', sizes=(50, 500), legend='brief')
df.loc[df.Height < 50]
df = df.drop(df[df.Height < 50].index)
def Analisis(df, attr=[], cols=4, figsize=(16, 16), ylim=(0, 100)):

    # Tamaño del gráfico

    plt.figure(figsize=figsize)



    # Atributos base

    base = ['Height', 'Weight']

    base.extend(attr)

        

    rens = math.ceil(len(base) / cols)

   

    i = 1

    for a in base:

        plt.subplot(rens, cols, i)

        i += 1

        sns.regplot(data=df, x=a, y='Overall', 

                    color='steelblue', scatter=False)

        sns.scatterplot(data=df, x=a, y='Overall', 

                    hue='Overall', legend='brief')

        plt.ylim(ylim)

    plt.tight_layout()

Analisis(df[df.Position.isin(['HB'])], attr=hb538, figsize=(20,20), ylim=(0,100))
Analisis(df[df.Position.isin(['HB'])], attr=['STR', 'CAR'], cols=2, figsize=(15,15), ylim=(40,100))
Analisis(df[df.Position == 'K'], cols=3, attr=['STR'], figsize=(20,6), ylim=(60, 100))
plt.figure(figsize=(20, 12))

sns.scatterplot(data=df, x='Weight', y='Height', hue='Position', legend='brief')
df['grupo'] = df.Position

df.loc[df.Position.isin(['HB', 'FB']), 'grupo'] = 'RB'

df.loc[df.Position.isin(['LT', 'LG', 'C', 'RG', 'RT']), 'grupo'] = 'OL'

df.loc[df.Position.isin(['LE', 'DT', 'RE']), 'grupo'] = 'DL'

df.loc[df.Position.isin(['LOLB', 'MLB', 'ROLB']), 'grupo'] = 'LB'

df.loc[df.Position.isin(['CB', 'SS', 'FS']), 'grupo'] = 'DB'

df
plt.figure(figsize=(20, 12))

sns.scatterplot(data=df, x='Weight', y='Height', hue='grupo', legend='brief')