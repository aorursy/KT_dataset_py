import pandas as pd
columnNames = ['YEAR','MOVIE']

MT = pd.read_csv("../input/netflix-prize-data/movie_titles.csv",names= columnNames,encoding = "ISO-8859-1")
MT.head(20)
MT.isnull().sum()
MT[pd.isnull(MT).any(axis=1)]
MT.at[4388,'YEAR'] = 2001

MT.at[4794,'YEAR'] = 2001

MT.at[7241,'YEAR'] = 2001

MT.at[10782,'YEAR'] = 1974

MT.at[15918,'YEAR'] = 1999

MT.at[16678,'YEAR'] = 1994

MT.at[17667,'YEAR'] = 1999
MT.isnull().sum()
MT['YEAR'] = MT['YEAR'].astype(int)
MT.head(5)
Cd1 = pd.read_csv("../input/netflix-prize-data/combined_data_1.txt",sep = ',',names= ['CustomerID','Rating','Date'] )
Cd2 = pd.read_csv("../input/netflix-prize-data/combined_data_2.txt",sep = ',',names= ['CustomerID','Rating','Date'] )
Cd3 = pd.read_csv("../input/netflix-prize-data/combined_data_3.txt",sep = ',',names= ['CustomerID','Rating','Date'] )
Cd4 = pd.read_csv("../input/netflix-prize-data/combined_data_4.txt",sep = ',',names= ['CustomerID','Rating','Date'] )
Cd1.head(5)
Cd2.head(5)
Cd3.head(5)
Cd4.head(5)
Cd1 = Cd1.dropna(how='any')
Cd2 = Cd2.dropna(how='any')
Cd3 = Cd3.dropna(how='any')
Cd4 = Cd4.dropna(how='any')
Cd1.isnull().sum()
Cd2.isnull().sum()
Cd3.isnull().sum()
Cd4.isnull().sum()
Cd1.shape
Cd2.shape
Cd3.shape
Cd4.shape 
MT['YEAR'].max()
MT['YEAR'].min()
import matplotlib

import matplotlib.pyplot as plt
MT['YEAR'].value_counts().plot(kind='bar', figsize=(20,5) )
