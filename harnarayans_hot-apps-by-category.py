# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib import rcParams
import warnings 

rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'

#Import the dataframe
df = pd.read_csv('../input/googleplaystore.csv')

#Remove the rows with non-numeric values of Review field 
df= df.dropna()



#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)

#To apply sorting on Reviews and Rating fields, convert them to numeric form 
df[['Reviews','Rating']] = df[['Reviews','Rating']].apply(pd.to_numeric)

#sort the data set by Reviews and Ratings
df = df.sort_values(by=['Reviews','Rating'],ascending=[0,0])
df.head(10)
df.shape
#Delete duplicate entries for Apps
df = df.drop_duplicates(subset=['App','Category'])
df.shape

#All the categories - 
df.Category.unique()

#Distribution of apps by category
g = sns.countplot(x="Category",data=df, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Count of app in each category',size = 20)
#Top 10 Apps in Tools Category 
hot_tools = df.loc[df['Category'] == 'TOOLS']
hot_tools.head(10)
#top 10 Apps in Games category 
hot_games = df.loc[df['Category'] == 'GAME']
hot_games.head(10)

#top 10 Apps in social and communitcation category
hot_soc_com = df.loc[df['Category'].isin(['SOCIAL','COMMUNICATION']) ]
hot_soc_com.head(10)
#top 10 apps in education and productivity 
hot_edu_pro = df.loc[df['Category'].isin(['EDUCATION','PRODUCTIVITY']) ]
hot_edu_pro.head(10)
#top 10 apps in family category
hot_fam = df.loc[df['Category'].isin(['FAMILY']) ]
hot_fam.head(10)