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



# Basic Visualization

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Interactive Visualization

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



# Special Visualization

from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator # wordcloud

import missingno as msno # check missing val

from PIL import Image



# Geo Visualization

from geopy.geocoders import Nominatim

df=pd.read_csv('/kaggle/input/fifa19/data.csv')
# Get general information on the dataset

df.info()
# Display the first five instances of the dataset

df.head()
# How many rows and columns does the dataset have?

print(df.shape)

# Statistical summary of the various columns

df.describe()
# Column names

df.columns
# How many players are in each position

df['Position'].value_counts()
# Scatter Plot Comparing the Value and Potential of Players

fig = px.scatter(df, x="Value", y="Potential", color="Age",

                  hover_data=['Name','Position'])

fig.show()
# Mean of the overall rating of players

overall=df['Overall']

overall_mean=overall.mean()

print(overall_mean)
# Statistical correlation between the variables

df.corr()
# A test on the iloc function of the Pandas library

messi=df.iloc[0]

messi
# Extract data on goalkeepers

goalkeeper=df[(df["Position"]=="GK")&(df["Overall"]>=75)]

goalkeeper.columns
# Assign goalkeepers to their nationalities

german_goalies=goalkeeper[goalkeeper["Nationality"]=="Germany"]

english_goalies=goalkeeper[goalkeeper["Nationality"]=="England"]

spanish_goalies=goalkeeper[goalkeeper["Nationality"]=="Spain"]

italian_goalies=goalkeeper[goalkeeper["Nationality"]=="Italy"]
fig = px.scatter(german_goalies, x="Penalties", y="Overall",

                 size="Overall", color="Club",

                 hover_name="Name", log_x=True, size_max=60)

fig.show()
# Create a dataframe of creative players based on certain criteria

creative_players=df[

                    (df["Skill Moves"]>=4) & 

                    (df["Dribbling"]>=80) &

                    (df["FKAccuracy"]>=80) & 

                    (df["BallControl"]>=80) &

                    (df["Vision"]>=80)

]

creative_players
# Which nation has the most creative players?

creative_players['Nationality'].value_counts().plot(kind='barh',figsize=(10,10),cmap='Paired')