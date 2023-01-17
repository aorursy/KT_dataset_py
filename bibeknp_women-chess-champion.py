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
import seaborn as sns

from matplotlib import pyplot as plt

import plotly.express as px
women_chess = pd.read_csv("../input/top-women-chess-players/top_women_chess_players_aug_2020.csv") #Load data from the CSV 

women_chess #Displaying the data
women_chess.head(5) #First 5 rows of the data
women_chess.dtypes #Exlporing the data types of each columns
women_chess.isnull().sum() #Check for the missing data in each columns

percent_missing = women_chess.isnull().sum()*100/len(women_chess) 

missing_values = pd.DataFrame({'percent_missing': percent_missing})

missing_values
women_chess.drop(["Fide id"], axis =1, inplace = True) #Removing column Fide id from the Dataset
women_chess.Title.value_counts(dropna = False) #Counting the unique values of the Title column
women_chess.Title.fillna('Title Unknown', inplace = True) #Permanently changing the Nan to Title Unknown

#women_chess.Title.value_counts(dropna = False)
women_chess.rename({'Federation':'Country'}, axis = 1, inplace = True) #Rename Federation to Country

women_chess.head(4)
# ds= women_chess["Title"].value_counts().reset_index()

# ds.columns = ['Title','count']

# sns.set(style = "whitegrid")

# plt.figure(figsize = (10,8))

fig = px.histogram(women_chess,"Year_of_birth",nbins = 25, width = 700, title = "Age Distribution of Chess Players")

fig.show()
fig = px.violin(women_chess,y = "Year_of_birth", title = "Violin Plot Age Distribution of Female Chess Players")

fig.show()
fig = px.violin(women_chess,y = "Standard_Rating", title = "Violin Plot Standard Rating of Female Chess Players")

fig.show()
women_chess.Year_of_birth.fillna('1988', inplace = True)

women_chess['Year_of_birth'].isnull().sum()
import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

top_country = women_chess['Country'].value_counts().head(10)

top_country.iplot(kind = 'bar',xTitle = 'Country', yTitle = 'count', title = "Top 10 countires by number of players")

women_chess['Year_of_birth']= women_chess["Year_of_birth"].astype(float) #Convert Date as object type to Float

women_chess['Year_of_birth'].dtypes 
women_chess['Year_of_birth'].mean()
top_title = women_chess['Title'].value_counts()

top_title.iplot(kind = 'bar',xTitle = 'Title', yTitle = 'count', title = "Top Title Chart", width = 500)
grand_master = women_chess[women_chess['Title']== 'GM']

grand = grand_master.groupby('Country')['Name'].count().sort_values(ascending = False).head(10)

grand.iplot(kind = 'bar',xTitle = 'Country',yTitle = 'Count', title = "Top 10 Countries with most number of female Grand Masters")
temp = women_chess.sort_values(by = 'Rapid_rating',ascending = False).head(10)

fig = px.funnel(temp, x = 'Rapid_rating', y = 'Name')

fig.show()
temp = women_chess.sort_values(by = 'Blitz_rating',ascending = False).head(10)

fig = px.funnel(temp, x = 'Blitz_rating', y = 'Name')

fig.show()
fig = px.density_heatmap(women_chess, x ='Standard_Rating', y = 'Rapid_rating')

fig.show()