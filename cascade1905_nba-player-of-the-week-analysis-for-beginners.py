# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/NBA_player_of_the_week.csv')
data.head(15)
data.info()
data.describe()
data.columns
data.corr()
import matplotlib.pyplot as plt
import seaborn as sns  

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.Age.plot(kind = 'hist',bins = 100,figsize = (15,15))
plt.show()

data.plot(kind='scatter', x='Age', y='Seasons in league',alpha = 0.5,color = 'red', figsize=(12,12))
plt.xlabel('Age')             
plt.ylabel('Seasons in league')
plt.title('Age & Seasons in league Scatter Plot')        
plt.show()
x = data['Age']>35    
data[x]
# We can see only the players older than 35 when he was chosen player of the week by using this filter.
y = data['Team']=='Los Angeles Lakers'   
data[y]
# We can see the players from only Utah Jazz Lakers by using this filter.
# We can also create new data_frames by using this data.
data_frame_specialities_of_players = data[['Player','Age','Weight','Height']]  
print(type(data_frame_specialities_of_players))
data_frame_specialities_of_players.head()


