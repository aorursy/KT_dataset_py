



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

ds=pd.read_csv("../input/youtube-new/USvideos.csv")



# Any results you write to the current directory are saved as output.
#Actividad 1

ds.tail(10)



#Actividad 2

ds.loc[ds.channel_title == 'The Deal Guy']



#Actividad 3

ds.iloc[5000]



#Actividad 4

ds.loc[ds.likes>= 5000000]



#Actividad 5

sum(ds.likes[ds.channel_title == 'iHasCupquake'])



#Actividad 6

grafico=ds.loc[ds.channel_title == 'iHasCupquake' ]

grafico.plot(kind = 'bar', x= 'trending_date' , y= 'likes')
import pandas as pd

CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")

DEvideos = pd.read_csv("../input/youtube-new/DEvideos.csv")

FRvideos = pd.read_csv("../input/youtube-new/FRvideos.csv")

GBvideos = pd.read_csv("../input/youtube-new/GBvideos.csv")

INvideos = pd.read_csv("../input/youtube-new/INvideos.csv")

JPvideos = pd.read_csv("../input/youtube-new/JPvideos.csv")

KRvideos = pd.read_csv("../input/youtube-new/KRvideos.csv")

MXvideos = pd.read_csv("../input/youtube-new/MXvideos.csv")

RUvideos = pd.read_csv("../input/youtube-new/RUvideos.csv")

USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")