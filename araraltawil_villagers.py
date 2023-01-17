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


data = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/villagers.csv')
data.head()
missing_percentage=df.isna().sum()*100/df.shape[0]
missing_percentage
The_Male=data[data['Gender']=='Male']
print(The_Male.shape)
The_Female=data[data['Gender']=='Female']
print(The_Female.shape)
import plotly.express as px
fig = px.histogram(The_Male,x='Species',color='Species')
fig.show()
import plotly.express as px
fig = px.histogram(The_Female,x='Species',color='Species')
fig.show()

import plotly.express as px
fig = px.histogram(The_Male,x='Color 1',color='Color 1')
fig.show()


fig = px.histogram(The_Male,x='Style 1',color='Style 1')
fig.show()


fig = px.histogram(The_Male,x='Color 2',color='Color 2')
fig.show()


fig = px.histogram(The_Male,x='Style 2',color='Style 2')
fig.show()


fig = px.histogram(The_Female,x='Color 1',color='Color 1')
fig.show()


fig = px.histogram(The_Female,x='Style 1',color='Style 1')
fig.show()


fig = px.histogram(The_Female,x='Color 2',color='Color 2')
fig.show()


fig = px.histogram(The_Female,x='Style 2',color='Style 2')
fig.show()

Color_1 = data['Color 1'].value_counts()
Color_1 = pd.DataFrame(Color_1)
Color_1=Color_1.reset_index()
fig = px.pie(Color_1, values='Color 1', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
Color_2 = data['Color 2'].value_counts()
Color_2 = pd.DataFrame(Color_2)
Color_2=Color_2.reset_index()
fig = px.pie(Color_2, values='Color 2', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
Style_1 = data['Style 1'].value_counts()
Style_1 = pd.DataFrame(Style_1)
Style_1=Style_1.reset_index()
fig = px.pie(Style_1, values='Style 1', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
Style_2 = data['Style 2'].value_counts()
Style_2 = pd.DataFrame(Style_2)
Style_2=Style_2.reset_index()
fig = px.pie(Style_2, values='Style 2', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()