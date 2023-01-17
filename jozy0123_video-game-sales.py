# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
game_rating = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
game_rating.info()
ax = sns.barplot(y = 'Platform', x = 'Name', data = game_rating.groupby(['Platform']).count().reset_index().sort_values('Name', ascending= False))

fig = plt.gcf()

fig.set_size_inches(13, 12)

ax.set(xlabel = 'Game Number', ylabel = 'Platform')

plt.show()
a = game_rating['Year_of_Release'].value_counts().reset_index().sort_values('index')

plt.figure(figsize=(10,8)) 

plt.plot(a['index'], a['Year_of_Release'])

plt.xlabel('Year')

plt.ylabel('Number of Games')
game_rating.groupby('Publisher').count().sort_values('Name', ascending = False)[:20]['Name'].plot(kind = 'bar', figsize = (12,7))
game_rating.groupby('Genre').count().sort_values('Name', ascending = False)['Name'].plot(kind = 'bar', figsize = (10,7))
game_rating.sort_values('NA_Sales').iloc[:10, :5]
game_rating.sort_values('EU_Sales').iloc[:10, :5]
game_rating.sort_values('JP_Sales').iloc[:10, :5]
game_rating['Rating'].value_counts().plot(kind = 'bar', figsize = (7,6))