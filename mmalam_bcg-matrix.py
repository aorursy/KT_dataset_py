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

import numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

data.head()
#DELETE FROM Games

#WHERE [Average User Rating] = ''



#DELETE FROM Games

#WHERE [User Rating Count] = ''

data = pd.read_csv('../input/appstore17k/appstore_games2.csv')

data.head()
#SELECT DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day]

#,*

#FROM Games



# Number of days = Number of days between Original Release Date and Data Extraction Date

# Ratings Per Day = Total Number of Ratings / Number of days  



data_3 = pd.read_csv('../input/appstore17k/appstore_games3.csv')

data_3.head()


#SELECT DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day]

#,*

#FROM Games

#WHERE [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >0

#ORDER BY [User Rating Count] DESC

data_4 = pd.read_csv('../input/appstore17k/appstore_games4.csv')

data_4.shape
plt.figure(figsize=(20,16))

nyc_img=plt.imread('../input/appstore17k/Boston-Matrix2.jpg')

#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot

plt.imshow(nyc_img,zorder=0,extent=[1, 20000, 1,12000], alpha=0.5)

p = plt.axis('off')

title = plt.title('BCG Matrix', fontsize=20)

title.set_position([0.5, 1.05])
#SELECT DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day],[User Rating Count],

#CASE WHEN [User Rating Count] >= 10000 THEN 10000/3032734.00*[User Rating Count]+10000 ELSE [User Rating Count] END Adj_Rate_Count,

#CASE WHEN [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >= 40 THEN 6000/1414.00*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04')+6000 ELSE 6000/39.99*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') END Adj_Rating_Per_Day

#FROM Games

#WHERE [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >0

#ORDER BY [User Rating Count] DESC

data_5 = pd.read_csv('../input/appstore17k/appstore_games5.csv')

data_5.head()
plt.figure(figsize=(20,16))

nyc_img=plt.imread('../input/appstore17k/Boston-Matrix2.jpg')

#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot

plt.imshow(nyc_img,zorder=0,extent=[1, 20000, 1,12000], alpha=0.1)

title = plt.title('BCG Matrix', fontsize=20)

title.set_position([0.5, 1.05])

ax=plt.gca()

sns.scatterplot(data_5.Adj_Rate_Count, data_5.Adj_Rating_Per_Day, ax=ax)

c = ax.set_xticklabels(['1', '', '', '', '', '10000', '', '', '', '3032734'], rotation=0, horizontalalignment='center')

c = ax.set_yticklabels(['', '', '', '40', '', '', '1414'], rotation=0, horizontalalignment='right')

ax.set_xlabel('Ratings Per Day')

ax.set_ylabel('Number of Ratings')
#SELECT [Average User Rating], DATEDIFF(day, [Original Release Date], '2019-08-04') [Number_Of_Days], [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') [Ratings_Per_Day],[User Rating Count],

#CASE WHEN [User Rating Count] >= 10000 THEN 10000/300000.00*[User Rating Count]+10000 ELSE [User Rating Count] END Adj_Rate_Count,

#CASE WHEN [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >= 40 THEN 6000/259.00*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04')+6000 ELSE 6000/39.99*[User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') END Adj_Rating_Per_Day

#FROM Games

#WHERE [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') >0

#AND [User Rating Count] < 300000 AND [User Rating Count]/DATEDIFF(day, [Original Release Date], '2019-08-04') < 300

#ORDER BY [Ratings_Per_Day] DESC

data_6 = pd.read_csv('../input/appstore17k/appstore_games6.csv')

data_6.head()
plt.figure(figsize=(20,16))

nyc_img=plt.imread('../input/appstore17k/Boston-Matrix2.jpg')

#scaling the image based on the latitude and longitude max and mins for proper output specially when drawing scattter plot

plt.imshow(nyc_img,zorder=0,extent=[1, 20000, 1,12000], alpha=0.1)

title = plt.title('BCG Matrix Zoomed In', fontsize=20)

title.set_position([0.5, 1.05])

ax=plt.gca()

sns.scatterplot(data_6.Adj_Rate_Count, data_6.Adj_Rating_Per_Day, ax=ax, hue=data_6["Average User Rating"],palette="Set3", legend="full", size=data_6["Average User Rating"])

c = ax.set_xticklabels(['1', '', '', '', '', '10000', '', '', '', '300000'], rotation=0, horizontalalignment='center')

c = ax.set_yticklabels(['', '', '', '40', '', '', '259'], rotation=0, horizontalalignment='right')

ax.set_xlabel('Ratings Per Day')

ax.set_ylabel('Number of Ratings')