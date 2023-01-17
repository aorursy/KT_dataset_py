# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')

data.head()
data.rename(columns={"Loudness..dB..":"Loudness",

                     "Speechiness.":"Speechiness",

                     "Track.Name":"Track",

                     "Artist.Name":"Artist"},inplace=True)

data.drop(["Unnamed: 0",

          "Valence.", 

          "Acousticness..",

          "Beats.Per.Minute", 

          "Length."], axis=1,inplace=True)



data.head()
data['Usability'] = np.sqrt((data['Energy'])**2 + (data['Danceability'])**2)

data.drop(["Energy", "Danceability"], axis=1,inplace=True)

data.head()
datacopy = data



for genre in data['Genre']:

    if re.search('pop', genre):

        data.loc[data.Genre == genre, 'Genre'] = 'pop'

        

data.head()
plt.figure(figsize=(15,10))

sns.scatterplot(y = "Popularity", x = 'Usability',

                hue = "Genre", data = data);
plt.figure(figsize=(15,10))

mymodel = np.poly1d(np.polyfit(x = data['Speechiness'], y = data["Popularity"], deg = 4))

myline = np.linspace(1, 50, 100)

plt.plot(myline, mymodel(myline))



sns.regplot(y = "Popularity", x = 'Speechiness', data = data.loc[data['Genre'] == 'pop'], fit_reg = True);