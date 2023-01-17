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
#Improts 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



path = "../input/"  #Insert path here

database = path + 'database.sqlite'
con = sqlite3.connect('../input/soccer/database.sqlite')
countries = pd.read_sql_query("SELECT * from Country", con)

countries
players = pd.read_sql_query("SELECT * from Player", con)

players['weight_kg'] = players['weight'] / 2.20462

players['MBI'] = players['weight_kg']**2 / players['height']

players
import plotly.express as px

fig = px.scatter(players, x="height", y="weight_kg", color="MBI")

fig.show()
players.iloc[players['MBI'].argmax()]
players.iloc[players['MBI'].argmin()]
players.iloc[players['height'].argmax()]
players.iloc[players['height'].argmin()]