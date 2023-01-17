# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import plotly.express as px
train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")

#train.columns
X = train.copy()

X.dropna(axis=0, subset=['Cover_Type'], inplace=True)

y = X.Cover_Type           

#X.drop(['Cover_Type'], axis=1, inplace=True)

X['Dis_To_Hy'] = ((X.Horizontal_Distance_To_Hydrology **2) + (X.Vertical_Distance_To_Hydrology **2))**0.5

X.columns
Cover_Type = {

    1 : "Spruce/Fir",

    2 : "Lodgepole Pine",

    3 : "Ponderosa Pine",

    4 : "Cottonwood/Willow",

    5 : "Aspen",

    6 : "Douglas-fir",

    7 : "Krummholz",

}
for i in range(4,5):

    fig = px.scatter_3d(X[X['Cover_Type'] == i], x='Horizontal_Distance_To_Roadways', y='Horizontal_Distance_To_Fire_Points', z='Vertical_Distance_To_Hydrology',

                  color='Aspect', size_max=8, size='Elevation', width=1000, height=800, opacity=0.9, template="plotly_dark")

    fig.update_layout(

        title=Cover_Type[i],

        font_size=16,

        legend_font_size=16,)

    fig.show()
for i in range(4,5):

    fig = px.scatter_3d(X[X['Cover_Type'] == i], x='Horizontal_Distance_To_Roadways', y='Horizontal_Distance_To_Fire_Points', z='Elevation',

                  color='Aspect', size_max=8, size='Elevation', width=1000, height=800, opacity=0.9, template="plotly_dark")

    fig.update_layout(

        title=Cover_Type[i],

        font_size=16,

        legend_font_size=16,)

    fig.show()
for i in range(4,5):

    fig = px.scatter_3d(X[X['Cover_Type'] == i], x='Horizontal_Distance_To_Roadways', y='Horizontal_Distance_To_Fire_Points', z='Horizontal_Distance_To_Hydrology',

                  color='Aspect', size_max=8, size='Elevation', width=1000, height=800, opacity=0.9, template="plotly_dark")

    fig.update_layout(

        title=Cover_Type[i],

        font_size=16,

        legend_font_size=16,)

    fig.show()
for i in range(4,5):

    fig = px.scatter_3d(X[X['Cover_Type'] == i], 

                        x='Horizontal_Distance_To_Roadways', 

                        y='Horizontal_Distance_To_Fire_Points', 

                        z='Dis_To_Hy',

                        color='Aspect', size_max=8, size='Elevation', width=800, height=800, opacity=0.9, template="plotly_dark")

    fig.update_layout(

        title=Cover_Type[i],

        font_size=16,

        legend_font_size=16,)

    fig.show()