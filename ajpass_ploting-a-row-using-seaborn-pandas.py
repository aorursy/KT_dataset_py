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
# Import libraries

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



# Path of the file to read

ign_filepath = '/kaggle/input/data-for-datavis/ign_scores.csv'



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")

ign_data
# Transpose dataframe 

ign_data_transpose = ign_data.T

ign_data_transpose
# Change the style of the figure to the "whitegrid" theme

sns.set_style("whitegrid")



# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Playstation 4 scores by genre")





# Bar chart showing score for genre in platform PlayStation 4

chart = sns.barplot(x=ign_data_transpose.index, y=ign_data_transpose['PlayStation 4'])



# Add label for vertical axis

plt.ylabel("Score")



# X - axis labels 

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ign_data.loc['PlayStation 4'].plot.bar()

plt.show()
sns.barplot(data=ign_data.loc[['PlayStation 4']])

plt.xticks(rotation=90)

plt.show()