# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imagecodecs



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#        pd.read_csv(os.path.join(dirname, filename)).info()



# Any results you write to the current directory are saved as output.
# DEFINE DATASET

data = pd.read_csv('/kaggle/input/italy-covid19/covid19-ita-regions.csv')

latest = data[data['date']== np.max(data.date)]
#SWABS PER PROVINCE

import matplotlib

import matplotlib.pyplot as plt

import numpy as np



x = np.arange(len(latest.region))  # the label locations

width = 0.2  # the width of the bars

plt.rcParams["figure.figsize"] = [8, 8]

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, list(latest.swabs_made-latest.total_confirmed_cases), width, label='Negative')

#rects2 = ax.bar(x + width, list(latest.total_confirmed_cases), width, label='Positive')

rects3 = ax.bar(x , list(latest.hospitalized_with_symptoms), width, label='Hospitalised')

rects4 = ax.bar(x + width, list(latest.home_quarantine), width, label='Quarantine')

# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_title('COVID19 Test Results per Italian Region')

ax.set_xticks(x)

ax.set_xticklabels(list(latest.region),rotation=70)

ax.legend()

#autolabel(rects1)

fig.tight_layout()



plt.show()
#PROVINCE ANALYSIS

prov = pd.read_csv('/kaggle/input/italy-covid19/covid19-ita-province.csv')

prov = prov[prov.province_code<900]



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

n_provinces_to_plot = 15

n_to_explode = 3

latest_prov = prov[prov.date==np.max(prov.date)].sort_values('total_cases', ascending = False)

labels = np.append(np.array(latest_prov.province[:n_provinces_to_plot]), 'Altre province')

sizes = np.append(latest_prov.total_cases[:n_provinces_to_plot], np.sum(latest_prov.total_cases[n_provinces_to_plot:]))

explode = np.zeros(n_provinces_to_plot+1)

explode[:n_to_explode] = 0.1

fig1, ax1 = plt.subplots()

ax1.set_title('Distribuzione dei ' + str(int(np.sum(latest_prov.total_cases))) + ' casi confermati per provincia. [' + str(max(latest_prov.date)) + ']')

ax1.pie(sizes, 

        labels=labels, 

        explode=explode, 

        autopct='%1.0f%%', 

        shadow=True, 

        startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
import os



def save_jpg (plt_file, name):

    plt_file.savefig(name + '.png')

    im = Image.open(name + ".png")

    rgb_im = im.convert('RGB')

    rgb_im.save(name + '1.jpg')

    os.remove(name + '.png')
# TREND OF MOST AFFECTED PROVINCES

from datetime import datetime

from PIL import Image

import matplotlib.dates as mdates





provinces = latest_prov.head(n_provinces_to_plot).province

fig2, ax2 = plt.subplots()



for p in provinces:

    provdata = prov[prov.province==p]

    plt.plot(provdata.date, provdata.total_cases, '-d', label=p, c=np.random.rand(3,)) 

    plt.legend(loc = 'upper left')

    ax2.set_title('Province italiane con più casi positivi di Covid19' )

    ax2.set_xticklabels(list(provdata.date),rotation=70)



    plt.title('')

#    ax2.xaxis.set_major_formatter(dates.DateFormatter('%d\n\n%a'))



##provdata.set_index(provdata.date.dt.strftime('%Y-%m-%d')).plot(kind='bar', color='g', rot=45, title='Revenue Each Side by Date', figsize=(10,5))



plt.show()

#name = 'prov_time_series' + 'pippo'

#save_jpg(plt,name)
# TREND FOR CENTRAL ITALY

from datetime import datetime



souther_regions = {'Toscana', 'Umbria', 'Marche', 'Lazio', 'Abruzzo'}

provinces = latest_prov[latest_prov.region.isin(souther_regions)].province[:(int)(n_provinces_to_plot/2)]

fig3, ax3 = plt.subplots(figsize=(25,10))



for p in provinces:

    provdata = prov[prov.province==p]

    plt.plot(provdata.date, provdata.total_cases, '-o', label=p, c=np.random.rand(3,)) 

    plt.legend(loc = 'upper left')

    ax3.set_title('Casi positivi di COVID19 nelle province del centro')

    ax3.set_xticklabels(list(provdata.date),rotation=60)



#plt.show()

name = 'centro' + max(provdata.date)

save_jpg(plt, name)
# TREND FOR SOUTHERN ITALY

from datetime import datetime



souther_regions = {'Campania', 'Molise', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia'}

provinces = latest_prov[latest_prov.region.isin(souther_regions)].province[:(int)(n_provinces_to_plot/2)]

fig4, ax4 = plt.subplots(figsize=(25,10))



for p in provinces:

    provdata = prov[prov.province==p]

    plt.plot(provdata.date, provdata.total_cases, '-o', label=p, c=np.random.rand(3,)) 

    plt.legend(loc = 'upper left')

    plt.plot([0, 1], [0, 1],'r--')  

    #plt.ylabel('True Positive Rate')

    #plt.xlabel('False Positive Rate')

    ax4.set_title('Casi positivi di COVID19 nelle province del Sud')

    ax4.set_xticklabels(list(provdata.date),rotation=40)



#plt.show()

name = 'sud' + max(provdata.date)

save_jpg(plt, name)