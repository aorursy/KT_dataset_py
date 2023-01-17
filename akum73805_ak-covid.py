import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
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
# Path of the file to read
coviddata_filepath = "/kaggle/input/coviddata.csv"

# Read the file into a variable spotify_data
covid_data = pd.read_csv(coviddata_filepath)
covid_data
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Deaths")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=covid_data['Place'], y=covid_data['Deaths'])

# Add label for vertical axis
plt.ylabel("Deaths")
cd2 = covid_data.copy()
cd2['adj_deaths'] = cd2.Deaths*1000000*1000/(cd2.Density*cd2.Population)
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Adjusted for density/population")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=cd2['Place'], y=cd2['adj_deaths'])

# Add label for vertical axis
plt.ylabel("Adjusted Deaths")
