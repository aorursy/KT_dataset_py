# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_path = '../input/cricket-world-cup-2019-players-data/Batsman_Data.csv'

bd = pd.read_csv(data_path)

bd.head()
#Seeing the initial data types of all columns

bd.dtypes
#Dropping all the values with 'Did not Bat (DNB)'

bd.drop(bd[bd.Bat1 == 'DNB'].index, inplace=True)

bd.head()
#To Convert required columns into float we have to replace and remove some string elements

bd['Bat1'] = bd['Bat1'].str.replace('*', '.')

bd['SR'] = bd['SR'].str.replace('-', '0')

bd['Runs'] = bd['Runs'].str.replace('-', '0')

bd['4s'] = bd['4s'].str.replace('-', '0')

bd['6s'] = bd['6s'].str.replace('-', '0')

bd.head()
#Converting all required columns into float

bd['SR']=bd['SR'].astype(float)

bd['Runs']=bd['Runs'].astype(float)

bd['4s']=bd['4s'].astype(float)

bd['6s']=bd['6s'].astype(float)
#Checking the datatype

bd.dtypes
#Top 100 batsmen with highest macthes/data

new = bd['Batsman'].value_counts()[:100]

new
#Gathering Data of reqd Batsman induvidually

bd_induvidual = bd.query('Batsman =="Shimron Hetmyer"')

bd_induvidual
#Gathering some important data from the required Batsman

print("The highest number of score for this batsman is: ")

print(bd_induvidual['Runs'].max())

print("The highest Strike Rate for this batsman is: ")

print(bd_induvidual['SR'].max())

print("The most number of 4s for this batsman is: ")

print(bd_induvidual['4s'].max())

print("The most number of 6s score for this batsman is: ")

print(bd_induvidual['6s'].max())
#Getting the data of a particular batsman against a particular Team

againstindia = bd_induvidual.query('Opposition == "v India"')

againstindia
plt.figure(figsize=(15,7))

x = againstindia['Ground']

y = againstindia['Runs']

plt.xlabel('Ground Location', labelpad=25)

plt.ylabel('Runs Scored', labelpad=25)

plt.title('Runs vs The Ground', fontweight='bold', pad=30, fontsize=20)

plt.bar(x, y)
#strike rate vs the score of player in each match

plt.figure(figsize=(15,7))

x = bd_induvidual['SR']

y = bd_induvidual['Runs']

plt.xlabel('Strike Rate', labelpad=30)

plt.ylabel('Runs', labelpad=30)

plt.title('Runs vs Strike Rate', fontweight='bold', pad=20, fontsize=20)

plt.scatter(x, y, color='red', s=300, edgecolor='black')
bd_opposition = bd.query('Opposition =="v England"')

bd_opposition
#Gathering some important data from the required Batsman

print("The highest number of score against this team is: ")

print(bd_opposition['Runs'].max())

print("The highest Strike Rate against this team is: ")

print(bd_opposition['SR'].max())

print("The most number of 4s against this team is: ")

print(bd_opposition['4s'].max())

print("The most number of 6s against this team is: ")

print(bd_opposition['6s'].max())