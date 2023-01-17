# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import folium
corona_dataset_csv = pd.read_csv('../input/covid-19/time_series_covid_19_confirmed.csv')
corona_dataset_csv.head(10)
corona_dataset_csv.shape
corona_dataset_csv.drop(['Lat','Long'],axis=1,inplace=True)
corona_dataset_csv.head()
corona_dataset_aggregated = corona_dataset_csv.groupby("Country/Region").sum()
corona_dataset_aggregated.head(10)
corona_dataset_aggregated.loc['India'].plot()
corona_dataset_aggregated.loc['China'].plot()
corona_dataset_aggregated.loc['US'].plot()
plt.legend()
corona_dataset_aggregated.loc['India'].plot()
corona_dataset_aggregated.loc['India'].diff().plot()
plt.title('Infection_Rate')
corona_dataset_aggregated.loc['India'].diff().max()
corona_dataset_aggregated.loc['China'].diff().max()
corona_dataset_aggregated.loc['US'].diff().max()
countries = list(corona_dataset_aggregated.index)
max_infection_rates = []
for country in countries :
    max_infection_rates.append(corona_dataset_aggregated.loc[country].diff().max())
corona_dataset_aggregated['max infection rate'] = max_infection_rates
corona_dataset_aggregated.head()
corona_data = pd.DataFrame(corona_dataset_aggregated['max infection rate'])
corona_data.head()
world_happiness_report = pd.read_csv("../input/covid-19/worldwide_happiness_report.csv")
world_happiness_report.head()
world_happiness_report.shape
world_happiness_report.drop(['Overall rank','Score','Generosity','Perceptions of corruption'],axis=1 , inplace=True)
world_happiness_report.head()
world_happiness_report.set_index(['Country or region'],inplace=True)
world_happiness_report.head()
data = corona_data.join(world_happiness_report,how="inner")
data.head()
corona_dataset2_csv = pd.read_csv("../input/covid-19/time_series_covid_19_confirmed.csv")
corona_dataset_aggregated2 = corona_dataset2_csv.groupby("Country/Region").sum()
countries = list(corona_dataset_aggregated2.index)
max_infection_rates = []
for country in countries :
    max_infection_rates.append(corona_dataset_aggregated2.loc[country].diff().max())
corona_dataset_aggregated2['max infection rate'] = max_infection_rates
corona_data2 = pd.DataFrame(corona_dataset_aggregated2,columns = ['Lat','Long'])
corona_data2.head()
data2 = data.join(corona_data2,how = "inner")
data2.head()
world_map = folium.Map(location=[23, 80], zoom_start=4)
# display world map
world_map
for i in range(0,len(data2)):
    folium.Circle(
        location=[data2.iloc[i]['Lat'],data2.iloc[i]['Long']],
        tooltip = "<h5>"+data2.iloc[i].name+"</h5>"+
        "<li>max infection rate"+str(data2.iloc[i]['max infection rate'])+"</li>"+
        "<li>Healthy life expectancy"+str(data2.iloc[i]['Healthy life expectancy'])+"</li>",
        
        radius = int((data2.iloc[i]['max infection rate'])*5.0),
    ).add_to(world_map)
world_map