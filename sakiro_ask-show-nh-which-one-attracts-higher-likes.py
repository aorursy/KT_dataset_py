import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv')
# we print the header of the data

data.head()
#check the total number of entries

len(data)
ask_hn = data[data['title'].str.contains('Ask HN')]
show_hn = data[data['title'].str.contains('Show HN')]
print(str(len(show_hn)) + ' total posts showing Hacker News')

print(str(len(ask_hn)) + ' total posts asking Hacker News')
askdf = ask_hn.describe().transpose()

showdf = show_hn.describe().transpose()
print(askdf)

print(showdf)
#Check for which is bigger, if TRUE more comments go to ASK HN else more comments go to SHOW HN

print( askdf['mean'][2] > showdf['mean'][2])

print('Ask HN recieves on average ' + str(round(askdf['mean'][2] - showdf['mean'][2],2)) + ' more comments than Show HN')
