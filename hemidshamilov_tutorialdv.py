# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# Path of the file to read
fifa_filepath = "../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Timestamp", parse_dates=True)
# Print the first 5 rows of the data
fifa_data.head()
# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=fifa_data)
# Path of the file to read
imdb_filepath = "../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv"

# Read the file into a variable imdb_data
imdb_data = pd.read_csv(imdb_filepath, index_col="ssc_p", parse_dates=True)
# Print the first 5 rows of the data
imdb_data.head()
# Print the first 5 rows of the data
imdb_data.tail()
list(imdb_data.columns)
# Line chart showing daily global streams of each song 
sns.lineplot(data=imdb_data)
#Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=imdb_data['specialisation'], label="specialisation")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=imdb_data['salary'], label="salary")

# Add label for horizontal axis
plt.xlabel("gender")
# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=imdb_data)
# Line chart showing daily global streams of each song 
sns.lineplot(data=imdb_data)