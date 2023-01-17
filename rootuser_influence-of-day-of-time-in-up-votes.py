import pandas as pd

import datetime

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn



data = pd.read_csv('../input/reddit_worldnews_start_to_2016-11-22.csv')





def getDT(timestamp):

    secondOfDay = int(timestamp)%(24*60*60)

    return secondOfDay/(60*60)



data['hours_of_day'] = data['time_created'].apply(getDT)







sortedUps = data.sort_values('up_votes',ascending=False)

bestData = sortedUps[:2000]

print(bestData['up_votes'])
plt.hist(data['hours_of_day'], 100, facecolor='blue', normed=1, alpha=0.75)

plt.hist(bestData['hours_of_day'], 100, facecolor='green', normed=1, alpha=0.75)







plt.xlabel('Time of Day')

plt.ylabel('Probability')

plt.title('At which time you should post for a successfull post (blue: all posts, green: best rated posts)')



plt.show()