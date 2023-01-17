import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")



print('Hecho')
#Actividad 1

USvideos.tail(10) 
#Actividad 2

USvideos.loc[USvideos.channel_title == 'The Deal Guy']
#Actividad 3

USvideos.iloc[5000]
#Actividad 4

USvideos.loc[USvideos.likes >= 5000000]
#Actividad 5

sum(USvideos.likes[USvideos.channel_title == 'iHasCupquake'])
#Actividad 6



plt.figure()

plt.title('Gráfico del canal iHasCupquake')

y = USvideos.likes[USvideos.channel_title == 'iHasCupquake']

x = USvideos.trending_date[USvideos.channel_title == 'iHasCupquake']

plt.plot(x,y)