# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from random import randint

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df2 = pd.read_csv('../input/pokemon-types/Pokemon Type Chart.csv')
data = pd.read_csv('../input/pokemon/Pokemon.csv')
data['Mega'] = None

data['Unbound'] = None
for i in range(data.__len__()):

    if len(data['Name'][i].split('Mega')) >= 2:

        data['Name'][i] = 'Mega ' + data['Name'][i].split('Mega')[0]

        data['Mega'][i] = 1

        data['Unbound'][i] = 0

    elif len(data['Name'][i].split('Unbound')) >= 2:

        data['Name'][i] = 'Unbound' + 'Hoopa'

        data['Unbound'][i] = 1

        data['Mega'][i] = 0

    elif len(data['Name'][i].split('Confined')) >= 2:

        data['Name'][i] = 'Hoopa'

        data['Unbound'][i] = 0

        data['Mega'][i] = 0

    else:

        data['Mega'][i] = 0

        data['Unbound'][i] = 0

data
data['Type 2'].fillna('None',inplace=True)
df2['types'] = df2['Unnamed: 0']
df2.index = df2.types
df2 = df2.drop('Unnamed: 0',axis=1)
df2
def Pokemon_Battle(poke1,poke2,type_chart):

    total1=0

    total2=0

    total1 = poke1.Total

    total2 = poke2.Total

    if list(poke1.Speed.values)[0]>list(poke2.Speed.values)[0]:

        total1+=poke1['Attack']

    elif list(poke2.Speed.values)[0]>list(poke1.Speed.values)[0]:

        total2+=poke2['Attack']

    a = list((poke1[['Type 1','Type 2']].values)[0])

    c = list((poke2[['Type 1','Type 2']].values)[0])

    a.append('types')

    c.append('types')

    

    print([[a,poke1.Name.iloc[0],poke1.Total.iloc[0]],[c,poke2.Name.iloc[0],poke2.Total.iloc[0]]])

    try:

        b = type_chart[a]

        d = type_chart[c]

        e = 'try'

    except:

        if 'None' in a and not 'None' in c:

            a.remove('None')

            d = type_chart[c]

            b = type_chart[a]

        elif 'None' in c and not 'None' in a:

            c.remove('None')

            d = type_chart[c]

            b = type_chart[a]

        else:

            c.remove('None')

            a.remove('None')

            d = type_chart[c]

            b = type_chart[a]

        e = 'except'

    if e == 'try':

        poke2_buff = b[b['types']==poke2['Type 1'].iloc[0]]

        poke2_buff1 = d[d['types']==poke1['Type 2'].iloc[0]]

        poke1_buff = d[d['types']==poke1['Type 1'].iloc[0]]

        poke1_buff1 = d[d['types']==poke1['Type 2'].iloc[0]]

        total1 = (total1.iloc[0]*poke1_buff[poke1_buff.columns[0]].iloc[0])+poke1_buff1[poke1_buff1.columns[0]].iloc[0]*100

        total2 = (total2.iloc[0]*poke2_buff[poke2_buff.columns[0]].iloc[0])+poke2_buff1[poke2_buff1.columns[0]].iloc[0]*100

    else:

        poke2_buff = b[b['types']==poke2['Type 1'].iloc[0]]

        poke1_buff = d[d['types']==poke1['Type 1'].iloc[0]]

        total1 = (total1.iloc[0]*poke1_buff[poke1_buff.columns[0]].iloc[0])

        total2 = (total2.iloc[0]*poke2_buff[poke2_buff.columns[0]].iloc[0])

    



    

        

    if total1 > total2:

        return f'{poke1.Name.iloc[0]} won against {poke2.Name.iloc[0]} with {total1} - {total2}'

    if total2 > total1:

        return f'{poke2.Name.iloc[0]} won against {poke1.Name.iloc[0]} with {total2} - {total1}'

    else:

        return 'draw'
a = randint(0,800)

b = randint(0,800)

Pokemon_Battle(data.iloc[a:a+1],data.iloc[b:b+1],df2)
data
def Pokemon(name):

    global data

    return (data[data['Name']==name])
Pokemon('Arceus')
def PokeBattle(Poke1,Poke2):

    global Pokemon,Pokemon_Battle,df2

    return Pokemon_Battle(Pokemon(Poke1),Pokemon(Poke2),df2)
PokeBattle('Mega Latios','Arceus')