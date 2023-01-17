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
# para coger los datos de youtube Estados Unidos

ds = pd.read_csv("../input/youtube-new/USvideos.csv")
# Actividad 1 para Mostrar las últimas 10 filas de youtube videos de Estados Unidos.



ds.tail(10)
# Actividad 2, como no tenemos el canal tenemos que buscarlo primero por el titulo de su video, una vez conseguido el nombre del canal buscamos todos los videos del canal

# ds.loc[ds.title == 'Top 10 Black Friday 2017 Tech Deals']

ds.loc[ds.channel_title == 'The Deal Guy']
# Actividad 3 para buscar informacion de este video cuando era trending el 17.09.12

ds.iloc[5000]
# Actividad 4 para buscar todos los videos de mas de 5 millones de likes en youtube Estados Unidos

ds.loc[ds.likes>= 5000000]
# Actividad 5



ds.loc[ds.channel_title == 'iHasCupquake']

# Actividad 5



sum(ds.likes[ds.channel_title == 'iHasCupquake'])
# actividad 6



ds.loc[ds.channel_title == 'iHasCupquake'].plot(kind = 'bar', x= 'trending_date' , y= 'likes')
# actividad 7

ds.loc[ds.channel_title == 'iHasCupquake'].plot(kind = 'bar', x= 'trending_date' , y= 'dislikes')