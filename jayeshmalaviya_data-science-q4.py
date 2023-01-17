# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly_express as px

import pandas as pd

import numpy as np
data = pd.read_csv("/kaggle/input/Suicide.csv")
data.head()
df_male =data[data['sex'] == 'male']

df_female =data[data['sex'] == 'female']
df_male.head()
df_female.head()
px.scatter(df_male, x= df_male['suicides_per_100k_pop'], y= df_male['gdp_per_capita_in_doller'], animation_frame=df_male['year'], animation_group=df_male['country'],

           size=df_male['suicides_per_100k_pop'] , color=df_male['country'], hover_name=df_male['country'], 

           log_y = True, 

           size_max=45, range_x= [1,150] , range_y= [100,150000] )
px.scatter(df_female, x= df_female['suicides_per_100k_pop'], y= df_female['gdp_per_capita_in_doller'], animation_frame=df_female['year'], animation_group=df_female['country'],

           size=df_female['suicides_per_100k_pop'] , color=df_female['country'], hover_name=df_female['country'], 

           log_y = True, 

           size_max=45, range_x= [1,130] , range_y= [100,150000] )
px.scatter(data, x= data['suicides_per_100k_pop'], y= data['gdp_per_capita_in_doller'], animation_frame=data['year'], animation_group=data['country'],

           size=data['suicides_per_100k_pop'] , color=data['country'], hover_name=data['country'], 

           log_y = True, 

           size_max=45, range_x= [1,150] , range_y= [100,150000] )