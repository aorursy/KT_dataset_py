# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import folium
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#df = pd.read_csv('/kaggle/input/uncover/covid_tracker_canada/covid-19-tracker-canada.csv')
df = pd.read_csv('../input/uncover/un_world_food_programme/un_world_food_programme/world-travel-restrictions.csv')
df.head()
#x == Long,  Y== lat
f_m = folium.Map(location=[10, 50],
                 tiles = 'Stamen Toner',
                 detect_retina = True, zoom_start=3)

for i in range(0, len(df)):
   # map(nltk.word_tokenize)
    s=tokenize.sent_tokenize(str(df.loc[i]['info']))[0:1]
    s=' '.join(map(str, s))
    folium.CircleMarker(
        location=[df.iloc[i]['y'], df.iloc[i]['x']],
        color='red', 
        popup =    '<bold>Country  '+'<font color="#a9b171">'+str(df.iloc[i]['iso3']).upper()+'</font>'+'<br>'+
                    #'<br>'+'<bold>Province  '+'<font color="#a9b171">'+str(_map.iloc[i]['Province/State']).upper()+'</font>'+'<br>'+
                    '<br>'+' '+'<font color="#c745eb">'+str(s)+'</font>'+'<br>',
                    #'<br>'+'<bold>Sources  '+'<font color="green">'+str(df.iloc[i]['sources'])+'</font>'+'<br>',
                    #'<br>'+'<bold>Deaths'+'<br>'+'<font color="red">'+str(_map.iloc[i]['Deaths'])+'</font>',
        radius=13,
        fill_color='red',                                 
        fill_opacity=0.7).add_to(f_m)
f_m
