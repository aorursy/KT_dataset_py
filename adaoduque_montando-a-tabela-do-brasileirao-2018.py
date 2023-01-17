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
#Load dataset

dfDataChampionship            =  pd.read_csv("/kaggle/input/campeonato-brasileiro-de-futebol/campeonato-brasileiro-full.csv", delimiter=",")



#To String

dfDataChampionship            =  dfDataChampionship.applymap(str);  



#Create new column Placar

dfDataChampionship["Placar"]  =  dfDataChampionship["Clube 1 Gols"].map(str) + "x" + dfDataChampionship["Clube 2 Gols"]



#Show data

dfDataChampionship.head()
d = {'Clubes': [], 'PG': [], 'j': [], 'V': [], 'E': [], 'D': [], 'GP': [], 'GC': [], 'SG': []}

dfClassChampionship = pd.DataFrame(data=d);
filter1  =  dfDataChampionship['Data']>='2018-01-01'

filter2  =  dfDataChampionship['Data']<='2018-12-31'

brasilianLeague2018 = dfDataChampionship[filter1 & filter2]

brasilianLeague2018.head()
#Cancatenando as colunas Clube 1 e Clube 2

clubs  =  pd.concat([brasilianLeague2018['Clube 1'].str.lower(), brasilianLeague2018['Clube 2'].str.lower()], axis=1, keys=['Clubes'])

#Obtem todos os clubes 

cb  =  pd.Series(clubs['Clubes'].unique(), name="Clubes")

cb  =  cb.to_frame()

cb
def getNumberWinner(data, clube ):

    df = data.apply(lambda x: x.str.strip())

    filter = df["Vencedor"].str.lower() == clube

    return (data[filter]['Vencedor'].count()).astype(np.int64)



def getNumberDepartures(data, clube ):

    df = data.apply(lambda x: x.str.strip())

    filter1 = df["Clube 1"].str.lower() == clube

    filter2 = df["Clube 2"].str.lower() == clube

    return (data[filter1]['Clube 1'].count() + data[filter2]['Clube 2'].count()).astype(np.int64)



def getPoints(data, clube ):

    df = data.apply(lambda x: x.str.strip())

    filter1 = df["Clube 1"].str.lower() == clube

    filter2 = df["Clube 2"].str.lower() == clube

    filter3 = df["Vencedor"].str.lower() == clube

    filter4 = (df["Clube 1"].str.lower() == clube) | (df["Clube 2"].str.lower() == clube)

    filter5 = df["Vencedor"].str.lower() == '-'

    

    v1 = data[(filter1) & (filter3)]

    v1 = v1['Vencedor'].count()

    v2 = data[(filter2) & (filter3)]

    v2 = v2['Vencedor'].count() 

    v3 = data[(filter4) & (filter5)]

    v3 = v3['Vencedor'].count()

    return ((v2*3)+(v1*3)+v3).astype(np.int64)



def getDraw(data, clube ):

    df = data.apply(lambda x: x.str.strip())

    filter1 = (df["Clube 1"].str.lower() == clube) | (df["Clube 2"].str.lower() == clube)

    filter2 = df["Vencedor"].str.lower() == '-'

    df = data[(filter1) & (filter2)]

    empates = df['Vencedor'].count()

    return empates.astype(np.int64)



def getDefeats(data, clube ):

    df = data.apply(lambda x: x.str.strip())

    filter1 = (df["Clube 1"].str.lower() == clube) | (df["Clube 2"].str.lower() == clube)

    filter2 = (df["Vencedor"].str.lower() != clube) & (df["Vencedor"].str.lower() != '-')

    df = data[(filter1) & (filter2)]

    derrotas = df['Vencedor'].count()

    return derrotas.astype(np.int64)



def getGP(data, clube ):

    df = data.apply(lambda x: x.str.strip())

    filter1  =  df["Clube 1"].str.lower() == clube

    filter2  =  df["Clube 2"].str.lower() == clube

    df1      =  data[(filter1)]

    df2      =  data[(filter2)]

    placar1  =  df1['Placar'].str.split('x')

    placar2  =  df2['Placar'].str.split('x')

    

    gp       =  0

    gc       =  0

    for g1, g2 in placar1:

        gp = (gp + pd.to_numeric( g1 ))

        gc = (gc + pd.to_numeric( g2 ))

            

    for g1, g2 in placar2:

        gp = (gp + pd.to_numeric( g2 )) 

        gc = (gc + pd.to_numeric( g1 ))

    return gp, gc


dfTable = cb[["Clubes"]].copy()

for column in ["PG", "J", "V", "E", "D", "GP", "GC", "SG"]:

    dfTable[column] = 0



def ensureUtf(s, encoding='utf8'):

  if type(s) == bytes:

    return s.decode(encoding, 'ignore')

  else:

    return s



for index, row in dfTable.iterrows():

    c1  =  row['Clubes']

    c1  =  ensureUtf( c1 )

    c1  =  c1.strip()

    

    pg       =  getPoints(brasilianLeague2018, c1 )

    j        =  getNumberDepartures(brasilianLeague2018, c1 )

    v        =  getNumberWinner(brasilianLeague2018, c1 )

    e        =  getDraw(brasilianLeague2018, c1 )

    d        =  getDefeats(brasilianLeague2018, c1 )

    gp, gc   =  getGP(brasilianLeague2018, c1 )

    

    if c1 == "sport":

        pg = pg -3

    

    dfTable.at[index, 'PG'] = pg

    dfTable.at[index, 'J']  = j

    dfTable.at[index, 'V']  = v

    dfTable.at[index, 'E']  = e

    dfTable.at[index, 'D']  = d

    dfTable.at[index, 'GP'] = gp

    dfTable.at[index, 'GC'] = gc

    dfTable.at[index, 'SG'] = gp - gc



dfTable['Clubes'] = dfTable['Clubes'].str.capitalize() 

dfTable  =  dfTable.sort_values(by=['PG', 'J', 'V'], ascending=False)

dfTable  =  dfTable.reset_index(drop=True)

dfTable