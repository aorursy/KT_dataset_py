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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



# Path of the file to read

spotify_filepath = "/kaggle/input/data-for-datavis/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)



# Print the first 5 rows of the data

spotify_data.tail()



# Line chart showing daily global streams of each song 

sns.lineplot(data=spotify_data)



# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Daily Global Streams of Popular Songs in 2017-2018")



# Line chart showing daily global streams of each song 

sns.lineplot(data=spotify_data)



list(spotify_data.columns)

# Set the width and height of the figure

plt.figure(figsize=(14,6))



# Add title

plt.title("Daily Global Streams of Popular Songs in 2017-2018")



# Line chart showing daily global streams of 'Shape of You'

sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")



# Line chart showing daily global streams of 'Despacito'

sns.lineplot(data=spotify_data['Despacito'], label="Despacito")



# Add label for horizontal axis

plt.xlabel("Date")