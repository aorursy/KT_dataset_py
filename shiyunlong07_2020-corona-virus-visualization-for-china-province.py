# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go
df = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv',

                header = 0,

                names = ['province', 'country','last_update','confirmed','suspected','recovered','death'])
df['last_update'] = pd.to_datetime(df['last_update']).dt.date
df.info()
df['country'].replace({'Hong Kong':'Mainland China',

                      'Taiwan':'Mainland China',

                      'Macau':'Mainland China'},inplace = True)
china = df[df['country'] == 'Mainland China']
from datetime import date



d = china['last_update'].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



china_update = china[china['last_update'] == pd.Timestamp(date(2020,2,5))]

china_update.head()
china_last = china_update.groupby(['province','last_update']).agg({'confirmed':sum,'recovered':sum,'death':sum}).reset_index().sort_values('confirmed',ascending = False)

china_last.head()
print("Until 05 Feb 2020, there are %s provinces infected in China."%format(len(china_update['province'].unique())))
# Top five province for confirmed

china_last.groupby('province')['confirmed'].sum().sort_values(ascending = False)[:5]
china_growth = china.groupby(['province','last_update']).max().reset_index().sort_values('last_update')
hubei = china_growth[china_growth['province'] == 'Hubei']

trace1 = go.Scatter(name = 'Hubei confimred growth', x = hubei['last_update'], y = hubei['confirmed'], line_shape = 'linear')

data = [trace1]

fig = go.Figure(data)



fig.update_layout(title = 'Confirmation growth for Hubei confirmed cases')

fig.show()





#hubei = china[china['province'] == 'Hubei'].reset_index()

zhejiang = china_growth[china_growth['province'] == 'Zhejiang']

guangdong = china_growth[china_growth['province'] == 'Guangdong']

henan = china_growth[china_growth['province'] == 'Henan']

hunan = china_growth[china_growth['province'] == 'Hunan']



#hubei_confirmed_growth = hubei['confirmed'].groupby(hubei['last_update']).max().reset_index()

#zhejiang_confirmed_growth = zhejiang.groupby(zhejiang['last_update']).agg({'confirmed':sum}).reset_index()

#guangdong_confirmed_growth = guangdong.groupby(guangdong['last_update']).agg({'confirmed':sum}).reset_index()

#henan_confirmed_growth = henan.groupby(henan['last_update']).agg({'confirmed':sum}).reset_index()

#hunan_confirmed_growth = hunan.groupby(hunan['last_update']).agg({'confirmed':sum}).reset_index()



#trace1 = go.Scatter(name = 'Hubei confimred growth', x = hubei_confirmed_growth['last_update'], y = hubei_confirmed_growth['confirmed'], line_shape = 'spline')

trace2 = go.Scatter(name = 'Zhejiang confimred growth', x = zhejiang['last_update'], y = zhejiang['confirmed'], line_shape = 'linear')

trace3 = go.Scatter(name = 'Guangdong confimred growth', x = guangdong['last_update'], y = guangdong['confirmed'], line_shape = 'linear')

trace4 = go.Scatter(name = 'Henan confimred growth', x = henan['last_update'], y = henan['confirmed'], line_shape = 'linear')

trace5 = go.Scatter(name = 'Hunan confimred growth', x = hunan['last_update'], y = hunan['confirmed'], line_shape ='linear') 



data = [trace2, trace3, trace4, trace5]

fig = go.Figure(data)



fig.update_layout(title = 'Confirmation growth for 5 most confirmed cases')

fig.show()
out_of_hubei = china[china['province'] != 'Hubei']

oohubei_confirmed_growth = out_of_hubei.groupby(out_of_hubei['last_update']).agg({'confirmed':sum}).reset_index()

china_growth_ = china.groupby(china['last_update']).agg({'confirmed':sum}).reset_index()



trace1 = go.Scatter(name = 'Hubei confirmed cases', x = hubei['last_update'], y = hubei['confirmed'],line_shape = 'spline')

trace2 = go.Scatter(name = 'Out of Hubei confirmed casses', x = oohubei_confirmed_growth['last_update'], y = oohubei_confirmed_growth['confirmed'], line_shape = 'spline')

trace3 = go.Scatter(name = 'China confirmed casses', x = china_growth_['last_update'], y = china_growth_['confirmed'], line_shape = 'spline')



data = [trace1, trace2,trace3]

fig = go.Figure(data)

fig.update_layout(title = 'Confirmed cases of Hubei VS out of Hubei')



fig.show()
# Top five province for mortality

china_last.groupby('province')['death'].sum().sort_values(ascending = False)[:5]
#Top five provinces for recovery

china_last.groupby('province')['recovered'].sum().sort_values(ascending = False)[:5]
china_recovery_growth = china.groupby(china['last_update']).agg({'recovered':sum}).reset_index()

china_death_growth = china.groupby(china['last_update']).agg({'death':sum}).reset_index()



trace1 = go.Scatter(name = 'China recovered cases trend', x = china_recovery_growth['last_update'], y = china_recovery_growth['recovered'], line_shape = 'spline')

trace2 = go.Scatter(name = 'China death cases trend', x = china_death_growth['last_update'], y = china_death_growth['death'], line_shape = 'spline')



data = [trace1, trace2]

fig = go.Figure(data)

fig.update_layout(title = 'Recovery/Death cases of China')



fig.show()
china['recovered_ratio'] = (china['recovered']/china['confirmed'])*100

china['death_ratio'] = (china['death']/china['confirmed'])*100



china_recovered_ratio = china.groupby(china['last_update']).agg({'recovered_ratio':'mean'}).reset_index()

china_death_ratio = china.groupby(china['last_update']).agg({'death_ratio':'mean'}).reset_index()



trace1 = go.Scatter(name = 'Recovered Ratio %', x = china_recovered_ratio['last_update'], y = china_recovered_ratio['recovered_ratio'])

trace2 = go.Scatter(name = 'Death Ratio %', x = china_death_ratio['last_update'], y = china_death_ratio['death_ratio'])

data = [trace1, trace2]

fig = go.Figure(data)

fig.update_layout(title = 'Recovery/Death ratio for China')

fig.show()
hubei['recovered_ratio'] = (hubei['recovered']/hubei['confirmed'])*100

hubei_recovery_ratio = hubei.groupby(hubei['last_update']).agg({'recovered_ratio':'mean'}).reset_index()

out_of_hubei['recovered_ratio'] = (out_of_hubei['recovered']/out_of_hubei['confirmed'])*100

out_of_hubei_ratio = out_of_hubei.groupby(out_of_hubei['last_update']).agg({'recovered_ratio':'mean'}).reset_index()



trace1 = go.Scatter(name = 'Hubei recovered ratio %', x = hubei_recovery_ratio['last_update'], y = hubei_recovery_ratio['recovered_ratio'], line_shape = 'spline')

trace2 = go.Scatter(name = 'Out of Hubei recovered ratio %', x = out_of_hubei_ratio['last_update'], y = out_of_hubei_ratio['recovered_ratio'], line_shape = 'spline') 

trace3 = go.Scatter(name = 'China recovered ratio %', x = china_recovered_ratio['last_update'], y = china_recovered_ratio['recovered_ratio'], line_shape = 'spline') 

data = [trace1, trace2, trace3]

fig = go.Figure(data)

fig.update_layout(title = 'Recovery ratio % of China, Hubei and Out of Hubei')



fig.show()





hubei['death_ratio'] = (hubei['death']/hubei['confirmed'])*100

hubei_death_ratio = hubei.groupby(hubei['last_update']).agg({'death_ratio':'mean'}).reset_index()

out_of_hubei['death_ratio'] = (out_of_hubei['death']/out_of_hubei['confirmed'])*100

out_of_hubei_death_ratio = out_of_hubei.groupby(out_of_hubei['last_update']).agg({'death_ratio':'mean'}).reset_index()



trace1 = go.Scatter(name = 'Hubei death ratio %', x = hubei_death_ratio['last_update'], y = hubei_death_ratio['death_ratio'], line_shape = 'spline')

trace2 = go.Scatter(name = 'Out of Hubei death ratio %', x = out_of_hubei_death_ratio['last_update'], y = out_of_hubei_death_ratio['death_ratio'], line_shape = 'spline') 

trace3 = go.Scatter(name = 'China death ratio %', x = china_death_ratio['last_update'], y = china_death_ratio['death_ratio'], line_shape = 'spline') 

data = [trace1, trace2, trace3]

fig = go.Figure(data)

fig.update_layout(title = 'Death ratio % of China, Hubei and Out of Hubei')



fig.show()