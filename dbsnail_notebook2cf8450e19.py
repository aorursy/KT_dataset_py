import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv("../input/tweets_all.csv", encoding = "ISO-8859-1")



print(data.head())
data.shape
top_10_country = data.country.value_counts()[:10]

top_10_country.plot(kind='barh')

plt.show()
top_10_city = data.full_name.value_counts()[:10]

top_10_city.plot(kind='barh')

plt.show()
by_name = sns.barplot(top_10_in_reply_to_screen_name.index,top_10_in_reply_to_screen_name.values ,\

                     color = 'purple', orient='v')

for item in by_name.get_xticklabels():

    item.set_rotation(70)



by_name.set(ylabel='Number of tweets', title = 'Top 10 Names')

plt.show()
def parse_user(row):

    usr_lst = []

    for word in row.split(' '):

        if word.startswith('@'):

            word = word.strip()

            if len(word)>1:

                usr_lst.append(word[1:])

    return usr_lst



data['user'] = data.text.map(lambda row:parse_user(row))
from collections import Counter



user_cnt = Counter()

user_list = data['user'].tolist()

usr_lst = []

for user in user_list:

    if len(user) > 0:

        usr_lst.extend(user)



for usr in usr_lst:

    user_cnt[usr] +=1

    

sorted_usr = sorted(user_cnt.items(), key=lambda x: (-x[1],x[0]))

usr = [ x[0] for x in sorted_usr[:10]]

cnt = [ x[1] for x in sorted_usr[:10]]

by_user = sns.barplot(usr,cnt, color = 'purple', orient='v')

for item in by_user.get_xticklabels():

    item.set_rotation(70)



by_user.set(ylabel='Number of tweets', title = 'Top 10 Mentioned Users')

plt.show()