# Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation Libraries

import matplotlib.pyplot as plt

from matplotlib import cm

import seaborn as sns

import warnings

import re



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-darkgrid')

pd.set_option('display.max_columns', 50)

warnings.filterwarnings("ignore")

#pd.options.display.float_format = '{:.1f}'.format
url = '../input/covid19-corona-virus-india-dataset/2020_03_11.csv'

data = pd.read_csv(url, header='infer')
data.shape
data.head()
data = data.rename(columns={'Name of State / UT':'State', 'Total Confirmed cases (Indian National)':'ConfirmedCase(Indian)',

                            'Total Confirmed cases ( Foreign National )':'ConfirmedCase(Foreign)'})
# Let's construct a function that shows the summary and density distribution of numerical columns :



def summary(x):

    x_max = data[x].max()

    print(f'Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(max)   : {x_max}\n')



    fig = plt.figure(figsize=(15, 8))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('muted')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'r')

    plt.title(f'{x.capitalize()} Density Distribution')

    

#     plt.subplot(222)

#     ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)

#     plt.title(f'{x.capitalize()} Boxplot')

    

   

    plt.show()

#Displaying the summary of Confirmed Case - Indian

summary('ConfirmedCase(Indian)')
# Create a function that returns a Pie chart and a Bar Graph for the categorical variables:

def pie_chart(x):

    """

    Function to create a Bar chart and a Pie chart for categorical variables.

    """

    from matplotlib import cm

    color1 = cm.inferno(np.linspace(.4, .8, 30))

    color2 = cm.viridis(np.linspace(.4, .8, 30))

    

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    

     

    """

    Draw a Pie Chart on first subplot.

    """    

    total = data[x].sum()

    States = data.State.tolist()

    mydata_index = data.index.tolist()  

       

    percs = []  # defining array of percentages

    

    for i in mydata_index:

        val = data.iloc[i][x]

        perc = round(((val / total)*100),1)

        percs.append(perc)



    ax.pie(percs, labels=States, autopct='%1.1f%%', shadow=False, startangle=90)

    ax.set_title(f'{x.capitalize()} Distribution Per Stage as of 11 Mar 2020', fontsize=16)

    



    fig.tight_layout()

    plt.show()
pie_chart('ConfirmedCase(Indian)')
#Draw a Bar Graph



fig, ax = plt.subplots(1, 1, figsize=(20, 10))

color1 = cm.inferno(np.linspace(.4, .8, 30))

color2 = cm.viridis(np.linspace(.4, .8, 30))

    



State_Label = data.State.tolist()             # creating a list of States

xlabel = np.arange(len(State_Label))          # X Axis Label Location

width = 0.35                                  # Width of the bar

ind = list(data['ConfirmedCase(Indian)'])     # List of Indian National Confirmed Cases

frg = list(data['ConfirmedCase(Foreign)'])     # List of Foreign National Confirmed Cases



rects1 = ax.bar(xlabel - width/2, ind, width, label="Indian", color=color1)

rects2 = ax.bar(xlabel + width/2, frg, width, label="Foreigner", color=color2)



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Scores')

ax.set_title("Total Confirmed Cases per State", fontsize=16)

ax.set_xticks(xlabel)

ax.set_xticklabels(State_Label, rotation=45)

ax.legend()



data
data['PopupTxt'] = data[['ConfirmedCase(Indian)','ConfirmedCase(Foreign)']].astype(str).apply(lambda x: ' - Indians < > Foreigners - '.join(x), axis=1)
# Importing Libraries

import folium

from folium.plugins import CirclePattern, HeatMap, HeatMapWithTime, FastMarkerCluster
# Function to create a base map



def generateBaseMap(default_location=[19.7515,75.7139], default_zoom_start=5):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start,width='50%', height='50%', tiles="cartodbpositron")

    folium.Popup(max_width=600,min_width=600)

    return base_map



#Calling the function

base_map = generateBaseMap()


for i in range(0, len(data)):

    folium.CircleMarker(location=[data.iloc[i]['Latitude'], data.iloc[i]['Longitude']],

                      popup=data.iloc[i]['PopupTxt'],radius = 10, color='#2ca25f',fill=True,    #Green

                      fill_color='#2ca25f').add_to(base_map)



base_map