import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os as os
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]

gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6 , 10289.7, 14958.3]

print('years e gdp tem o mesmo tamanho ?', len(years) == len(gdp))



plt.plot(years, gdp, color='green', marker='o',linestyle='solid')

plt.title('CDP nominal')

plt.ylabel('Bilhões em $')

plt.show()
movies = ['Annie Hall', 'Ben-Hur', 'Casablanca', 'Gandhi', 'West Side Story']

num_oscars =[ 5, 11,3, 8 ,10]

# para centralizar as barras

xs = [ i + 1 for i, _ in enumerate(movies)]

plt.bar(xs, num_oscars)

plt.ylabel('# de premiações')

plt.title('Meus filmes favoritos')

plt.xticks([i + 0.5 for i, _ in enumerate(movies)], movies)

plt.show()
from collections import Counter 



grades = [83 , 95, 91, 87, 70, 0, 85,82, 100, 67, 73, 77, 0]

decile = lambda grade: grade // 10 * 10

histogram = Counter(decile(grade) for grade in grades)

plt.bar(

    histogram.keys(), 

    histogram.values(), 

    8

)

plt.axis([ -5, 105, 0, 5])

plt.xticks([ 10 * i for i in range(11) ])

plt.xlabel('Decil')

plt.ylabel('# de alunos')

plt.title('Distribuição das notas do Teste 1')

plt.show()

variance = [ 1,2,4,8,16,32,64,128, 256]

bias_squared = [256, 128,64, 32,16,8,4,2,1 ]

total_error = [ x + y  for x,y  in zip(variance, bias_squared)]



xs = [i for i, _ in enumerate(variance)]



plt.plot(xs, variance, 'g-', label='variance')

plt.plot(xs, bias_squared, 'r-', label='bias ')

plt.plot(xs, total_error, 'b:', label='total error')

plt.show()
friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]

minutes = [ 175, 170, 205, 120, 220, 130, 105, 145, 190]

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']



plt.scatter(friends, minutes)



for label, friend_count, minute_count in zip(labels, friends, minutes):

    plt.annotate(label, 

                 xy=(friend_count, minute_count),

                 xytext=(5,-5),

                 textcoords='offset points'

                )

# no livro esta escrito 'equals'    

plt.axis('equal')

plt.title('Minutos Diários vs. Números de Amigos')

plt.show()
import random as random

from collections import Counter 





num_friends = [ random.randrange(1,70) for i in list(range(0, 100)) ] 

friends_count = Counter(num_friends)

# print(friends_count)

plt.bar(

    friends_count.values(), 

    friends_count.keys()

)

plt.title('Histograma de Contagem de Amigos')

plt.xlabel('# de amigos')

plt.ylabel('# de pessoas')

# plt.axis('equal')

plt.axis([0, 10,  0, 80])

plt.show()
data = pd.read_csv(

    "../input/enem-por-escola-2005-a-2015/microdados_enem_por_escola/DADOS/MICRODADOS_ENEM_ESCOLA.csv",

    error_bad_lines=False,

    sep=';',

    engine='python',

    index_col='SG_UF_ESCOLA'

) 



data.info()

df_by_df = data.groupby('SG_UF_ESCOLA')['NU_MATRICULAS'].sum().to_frame()

df_by_df
plt.clf()





years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]



for year in years:

    plt.title('Alunos por estado em {}'.format(year))

    plt.xlabel('estados')

    plt.ylabel('# de alunos')

    data.loc[data['NU_ANO'] == year].groupby(

        ['SG_UF_ESCOLA']

    )['NU_MATRICULAS'].sum().plot(kind='bar')





    plt.show()
