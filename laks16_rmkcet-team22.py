# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

df=pd.read_csv("../input/2015_16_Statewise_Elementary.csv")

df.head()

df.info()
state=df[["STATCD","STATNAME","TOT_6_10_15","SCH1","ENR1"]]

state.head()
# NUMBER OF DROPOUTS FROM PRIMARY SCHOOL IN EACH STATE

CHILDREN=list(state.TOT_6_10_15)

ENROLLMENT=list(state.ENR1)

DROP_OUTS=np.array(CHILDREN)-np.array(ENROLLMENT)

print(list(DROP_OUTS))
state.info()

#LIST CONTAINING THE DROPOUTS OF EACH STATE IS ADDED AS A COLUMN IN STATE DATAFRAME

state["DROPOUTS_PRIMARY"]=DROP_OUTS

state.head()
#BOKEY PLOT IMPORT STATEMENTS

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
#PLOTTING BOKEH GRAPH BETWEEN STATECODE AND DROPOUTS_PRIMARY

from bokeh.io import output_file,show

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

x=list(state.STATCD)

y=list(state.DROPOUTS_PRIMARY)

plot=figure()

plot.line(x,y,line_width=2,color="black")

plot.circle(x,y,fill_color="yellow",size=10)

output_file('line.html')

show(plot)

#FROM THE ABOVE BOKEH GRAPH IT IS OBSERVED THAT THE STATECODE WHICH HAS 11 HAS THE HIGHEST DROPOUTS FROM PRIMARY SCHOOLS. 

state[["STATCD","STATNAME"]]
a=state.STATNAME[:11]

plt.scatter(list(range(1,12)),list(a))

plt.show()

#OBSERVATIONS:THUS FROM THE SCATTER PLOT WE CAN PREDICT THAT THE STATE SIKKIM HAS HIGHEST NUMBER OF DROPOUTS FROM PRIMARY SCHOOLS.