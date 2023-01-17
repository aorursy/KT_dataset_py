# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/world-cup-dataset/WorldCups.csv', thousands='.')

df.rename(columns={df.columns[3]:'RunnersUp'}, inplace=True)

attendance_sum = df[df['Year'] % 10 == 0]['Attendance'].sum()



goals_scored = df[df['Year'].between(1954, 1990, inclusive=True)]['GoalsScored'].sum()



attendance_mean = df['Attendance'].mean()



goals_matches = df['GoalsScored'].sum() /  df['MatchesPlayed'].sum()



home_winner = (df['Country'] == df['Winner']).sum()



brasil_winner = df.query('Winner=="Brazil" | RunnersUp=="Brazil" | Third=="Brazil" | Fourth=="Brazil"')['Year'].count()



df_france = df[df['Third'] == "France"].copy()

df_france['Year'] = df_france['Year'].apply(str)

france_third = df_france['Year'].str.cat(sep=', ')



ranking = df.groupby(['Winner'])['Year'].count().sort_values()

with open('WorldCupsOutput.txt', 'w', encoding='utf-8') as output:

    output.write('Soma de público das copas com anos final 0 (1930, 1950, etc): ' + str(attendance_sum) + '\n')

    output.write('Quantidade total de gols entre as copas de 1954 e 1990, inclusive: ' + str(goals_scored) + '\n')

    output.write('Média de público: ' + str(round(attendance_mean,2)) + '\n')

    output.write('Média de gols por partida: ' + str(round(goals_matches,2)) + '\n')  

    output.write('Quantidade de vezes em que o país sede foi campeão: ' + str(home_winner) + '\n')  

    output.write('Quantidade de vezes em que o time do Brasil ficou entre uma das 4 primeiras posiçõe: ' + str(brasil_winner) + '\n')   

    output.write('Ano das edições em que o time da França finalizou em terceiro lugar: ' + str(france_third) + '\n')

    output.write('Quantidade de vitórias por país, classificada em ordem crescente do número de títulos: \n')

    output.write(ranking.to_string(header=False))

    

    