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

import pandas as pd



# Any results you write to the current directory are saved as output.
#Actividad 1:

#Mostrar las últimas 10 filas de youtube videos de Estados Unidos.



ds = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

ds.tail(10)
#Actividad 2:

#Mostrar todas las filas de este canal de youtube The Deal Guy



ds.loc[ds.channel_title == 'The Deal Guy' ]
#Actividad 3:

#fila 5000



ds.iloc[5000]
#Actividad 4:

#Mostrar los resultados con más de 5,000,000 LIKES.



ds.loc[ds.views > 5000000 ]
#Actividad 5:

#Mostrar el número total de LIKES que tiene el canal iHasCupquake.

sum(ds.likes[ds.channel_title == 'iHasCupquake'])

#Actividad 6:

import matplotlib.pyplot as plot

plot.hist(channel_id = 'iHasCupquake' ,density=1, bins=20) 

plot.axis([50, 110, 0, 0.06]) 

#axis([xmin,xmax,ymin,ymax])

plot.xlabel ('trending_date')

plot.ylabel('likes')
