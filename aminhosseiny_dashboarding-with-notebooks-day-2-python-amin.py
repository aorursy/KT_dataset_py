#import plotly library 
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

#import data we need
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

data = pd.read_csv("../input/nypd-motor-vehicle-collisions/nypd-motor-vehicle-collisions.csv")
#factor type accidents
Btypes = list(data['BOROUGH'].unique()) #vehicle types
Atypes = np.zeros(len(Btypes)) #set each vehicle type accidents to zero

#CONTRIBUTING FACTOR VEHICLE 1
#search over all types, sum the number up
d = data['BOROUGH'] #read the column
for itr in range(len(Btypes)): 
    ind = [i for i,x in enumerate(d) if x == Btypes[itr]] #find all occurences
    Atypes[itr] = Atypes[itr] + len(ind) #add all occurences

#mp.xticks(range(len(Btypes)),Btypes)
#mp.bar(range(len(Btypes)),Atypes)
#figData = [go.Bar(x=pd.DataFrame({'rang':range(len(Btypes))}),y=pd.DataFrame({'accicdens' : Atypes}))]
figData = [go.Bar(x=list(range(len(Btypes))),y=Atypes)]
layout = dict(title = "Number of Accidents In Different Regions", 
             xaxis=dict(title='Region Name'),yaxis=dict(title='Number of Accidents'))
fig = dict(data = figData, layout = layout)
iplot(fig)
