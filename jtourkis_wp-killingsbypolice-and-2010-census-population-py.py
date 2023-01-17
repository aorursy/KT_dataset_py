# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/washington-post-police-killings-data-2010-census/Washington_Post_Top_Cities_with_2010_Census_Pop_Final.csv")
data['city_state'] = data['wp_City'].str.cat(data['wp_State_abv'], sep =", ") 

data.head()
import matplotlib.pyplot as plt

pd_stat=data[data['KillingsbyPolice']>20]





# DataFrame of avg deaths and avg pop density by state

pd_stat = pd_stat.groupby(['city_state'])['KillingsbyPolice','Census.2010.Population'].sum()

pd_stat['Census.2010.Population']=pd_stat['Census.2010.Population']/100000



pd_stat.plot(kind="bar")

plt.ylabel('# of People')

plt.title('2015-2020 Killings by Police Top Cities Compared to Population/100,000')

plt.show()
# Plot same data with adjusted ticks and subplots

pd_stat.plot()



# Set xticks

plt.xticks(

    np.arange(len(pd_stat.index)),

  pd_stat.index, 

    rotation='vertical')



# Show the plot

plt.show()



# Plot with subplots

pd_stat.plot(subplots=True)

plt.show()
#####Use matplotlib to create a scatter plot(x,y). Inflate dots by population density.

Deaths=np.array(pd_stat['KillingsbyPolice'])

plt.scatter(pd_stat['Census.2010.Population'], pd_stat['KillingsbyPolice'],s=Deaths , alpha=.3, c="black")



xlab = 'Population'

ylab = 'Deaths'

title = 'Killings by Police 2015-2020 Top Cities ~ Population (Dot Size Proportional to # Deaths)'



# Add axis labels



plt.xlabel(xlab) 

plt.ylabel(ylab)



# Add title

plt.title(title)

# Add grid() call

#plt.grid(True)



# Add Text

#plt.text(.3, 10, 'Big Pop Density/High Deaths')



# After customizing, display the plot

plt.show()
#import plotly.graph_objects as go

#import plotly.express as px

#import pandas as pd

#import math



     

#data = data.sort_values(['State','wp_City'])



#bubble_size = []





#for index, row in data.iterrows():

#    bubble_size.append(math.sqrt(row['Census.2010.Population']))



#data['size'] = bubble_size

#State_names = ['Alaska', 'Alabama', 'Arkansas', 'California', 'Colorado', 'District of Columbia', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois','Indiana','Kansas', 'Kentucky','Louisiana', 'Massachusetts', 'Maryland', 'Michigan', 'Minnesota', 'Missouri', 'Mississippi', 'Montana', 'North Carolina', 'Nebraska', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia', 'Washington','Wisconsin', 'West Virginia', 'Wyoming']

#State_data = {State:data.query("State == '%s'" %State)

#                              for State in State_names}



#fig = go.Figure()



#for State_name, State in State_data.items():

#    fig.add_trace(go.Scatter(

#        x=State['Census.2010.Population'],

#        y=State['KillingsbyPolice'],

#        name=State_name,

#        text=data['State'],

#        hovertemplate=

#        "<b>%{text}</b><br>" +

#        "City Population: %{x}<br>" +

#        "Killings by Police:%{y}<br>" +

#        "<extra></extra>"

#        ))



#fig.update_traces(

#    mode='markers',

#    marker={'sizemode':'area',

#            'sizeref':10})



#fig.update_layout(title_text="Killings by Police 2015-2020 Top 150 Cities ~ Population (WRONG)<br> Under Construction",

#    xaxis={'title':'Population'},

#    yaxis={'title':'Killings by Police'})



#fig.show(rendering = "kaggle")
pd_stat.head()

import plotly.express as px

#df = px.data.gapminder()



fig = px.scatter(data, x='Census.2010.Population', y='KillingsbyPolice',

	         size='KillingsbyPolice', color= 'State',

                 hover_name='city_state', size_max=60)

fig.update_layout(title_text="Killings by Police 2015-2020 Top 150 Cities ~ Population")

fig.show(rendering = "kaggle")



#plt.scatter(pd_stat['Census.2010.Population'], pd_stat['KillingsbyPolice'],s=Deaths , alpha=.3, c="black")