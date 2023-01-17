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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler

from math import sqrt

sns.set(color_codes=True)
data=pd.read_csv('/kaggle/input/games-data/games.csv')
data.head()
data.shape
data.info()
data.describe()
data.isnull().sum()
#mean playing time for all the games  all together

data['playingtime'].mean()
#highest comments recieved for the game

data[data['total_comments']==data['total_comments'].max()]['type']
#which game recieved leastnumber of comments

data[data['total_comments']==data['total_comments'].min()]

#what was the avg minage of all games as per game "type"(boardgame and boardgameexpnasion)

data.groupby(data['type']).mean()['minage']
#unique games in dataset

data['id'].nunique()
data['type'].value_counts()

# here proportion of information is highly skewed
#checking the correlation between playing time and total comments in the dataset

data[['playingtime','total_comments']].corr()

# low correlation observed 
import seaborn as sns

sns.set(color_codes=True)
data.isnull().sum()
#dropping na values to nagating issues during data visualization

data.dropna(inplace=True)

data.info()
data.shape
data.duplicated().value_counts()
data['id'].nunique()
data.drop_duplicates(subset ="id", inplace = True)

data.shape
#The data set has -ve years and year =0. The data type is float for year column.
data['yearpublished'].dtype
data['yearpublished'] = data['yearpublished'].astype(int)

data['yearpublished'].dtype
data['yearpublished'].min()#-ve dates
data[data['yearpublished']>0].yearpublished.min()
data[data['yearpublished']<=0].yearpublished.count()
data[data['yearpublished']==0].yearpublished.count()
# distplot for average_rating

sns.distplot(data['average_rating'],kde=True)
# Is there arelationship between minage and average_rating

sns.jointplot(data['minage'],data['average_rating'])

#there is very low correlation since most of the points close to zero
#comparing relationship between playingtime minage and average_rating 

sns.pairplot(data[['playingtime','minage','average_rating']])

#there is no observable correlation
#compare playing time and type of game 

sns.stripplot(data['type'],data['playingtime'])
sns.distplot(data['average_rating'])