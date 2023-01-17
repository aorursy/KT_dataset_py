import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding="ISO-8859-1")
data.info()
data.head(3)
data.columns
data.iyear.plot(kind = 'hist',bins = 40, figsize = (8,8)) 

plt.xlabel('Years')              # label = name of label

plt.ylabel('Frequency')

plt.title('Attack Frequency')            # title = title of plot

plt.show()
cleandata=data[['iyear','imonth','iday','country_txt','region_txt','city','attacktype1_txt','nkill','nwound','gname','targtype1_txt','weaptype1_txt']]
cleandata.info()
cleandata.head(3)
cleandata.plot("nkill","iyear",kind = 'hist',bins = 40,figsize = (8,8),grid= True)

plt.xlabel('Years')             

plt.ylabel('Killed')

plt.title('Killed Frequency')

plt.show()
cleandata.plot("nkill","imonth",kind = 'hist',bins = 12,figsize = (8,8),grid= True)

plt.xlabel('Months')            

plt.ylabel('Killed')

plt.title('Killed Frequency')

plt.show()
cleandata.plot("nkill","iday",kind = 'hist',bins = 31, figsize = (8,8),grid= True)

plt.xlabel('Days')            

plt.ylabel('Killed')

plt.title('Killed Frequency')

plt.show()
data_tr = data[data.country_txt =='Turkey']

data_tr.plot(x="iyear",y="nkill",grid= True)

plt.xlabel('Attacks')            

plt.ylabel('Killed')

plt.title("Casulities in Turkey")

plt.show()
data_izm = data[cleandata.city =='Izmir']

data_izm.plot(x="iyear",y="nkill",color="r",grid= True)

plt.xlabel('Attacks')            

plt.ylabel('Killed')

plt.title("Casulities in İzmir")

plt.show()
data_izm = data[cleandata.city =='Izmir']

data_izm.plot(x="iyear",y="nwound",color="b",grid= True)

plt.xlabel('Attacks')            

plt.ylabel('Wounded')

plt.title("Casulities in İzmir")

plt.show()