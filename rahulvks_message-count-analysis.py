# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import re

import numpy as np

import string

import matplotlib as plt

import seaborn as sns

from datetime import datetime

import codecs

import datetime as dt

import wordcloud

from pandas import Series, DataFrame, Panel

%pylab inline
chat = pd.read_csv("../input/messages.csv")
chat.head(5)
chat.isnull().values.any()


len(chat)
chat.columns
chat['date']=pd.to_datetime(chat['date'])

chat["new_date"] = chat["date"].dt.date

chat["time"] = chat["date"].dt.time

chat['day'] = chat['date'].dt.day

#chat['DayOfWeekNum'] = chat['new_date'].dt.dayofweek

#chat['DayOfWeekNum'] = chat['new_date'].dt.dayofweek

chat['month'] = chat['date'].dt.month

chat['year'] = chat['date'].dt.year

chat['hour'] = pd.to_datetime(chat['date'],format ='%H:%M').dt.hour

chat['minutes'] = pd.to_datetime(chat['date'],format='%H:%M').dt.hour
def removePunctuation(x):

    # Lowercasing all words

    x = x.lower()

    # Removing non ASCII chars

    #x = re.sub(r'[^\x00-\x7f]',r' ',x)

    # Removing (replacing with empty spaces actually) all the punctuations

    return re.sub("["+string.punctuation+"]", " ", x)

chat['msg'] = chat['msg'].map(removePunctuation)
countbyyear = chat.groupby('year').size()

countbyyear

countbyyear.plot(kind='bar',color=['red', 'green', 'blue', 'cyan','yellow'],title="Total Message Count In Years")
countbymonth = chat.groupby('month').size()

countbymonth
countbymonth.plot(kind='bar',title="Total Message Count In Months")
plt.figure(2,figsize=(18,12))

ax1 = plt.subplot(511)

sns.barplot(chat["year"], chat['month'])

plt.ylabel("Number of Messges By Year")





ax2 = plt.subplot(512)

sns.barplot(chat["month"], chat['year'])

plt.ylabel("Number of Messges By Month")



ax3 = plt.subplot(513)

sns.barplot(chat["day"], chat['month'])

plt.ylabel("Number of Messges By Day")





ax1.set_title('Message Count in 2013-2017')

plt.xlabel("Report Year")

plt.show()

Repated_Words = pd.Series(' '.join(chat['msg']).lower().split()).value_counts()[:10]

Repated_Words.plot(kind='bar')
allwords = ' '.join(chat['msg']).lower().replace('c', '')

cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=100,

                            width=1000,

                            height=500,

                            max_words=300,

                            relative_scaling=.5).generate(allwords)

plt.figure(figsize=(15,5))

plt.axis('off')

#plt.savefig('allword.png')

plt.imshow(cloud);