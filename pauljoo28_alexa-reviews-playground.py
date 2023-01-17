# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/amazon_alexa.tsv', sep = "\t", header = 0)
df.head()
text = ''
for i in df.verified_reviews:
    text = text + " " + i

wc = WordCloud()
image = wc.generate(text)

plt.imshow(image)
plt.axis('off')
plt.show()
good = df.loc[df.feedback==1]
bad = df.loc[df.feedback==0]

plt.figure(figsize=(20, 20))

#Good
text = ''
for i in good.verified_reviews:
    text = text + " " + i

wc = WordCloud()
image = wc.generate(text)
    
plt.subplot(1,2,1)
plt.title('good')
plt.imshow(image)
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 20))
#Bad
text = ''
for i in bad.verified_reviews:
    text = text + " " + i

wc = WordCloud()
image = wc.generate(text)
    
plt.subplot(1,2,2)
plt.title('bad')
plt.imshow(image)
plt.axis('off')
plt.show()



dates = pd.Series(df.date.unique())
dates

#Function to make dates into numbers to sort year,month,day
def date2num(dates):
    monarray = pd.Series(['01','02','03','04','05','06','07','08','09','10','11','12'],index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    num = []
    for i in range(len(dates)):
        day = dates[i].split('-')[0]
        month = dates[i].split('-')[1]
        year = dates[i].split('-')[2]
        month = monarray[month]
        if len(day) == 1:
            day = '0' + day
        num.append(int(year + month + day))
    return pd.Series(num,index=dates.values)

numdates = date2num(dates)
for i in range(len(df.index)):
    df.date[i] = numdates[df['date'].loc[i]]

df.date.unique()
df.head()
df.sort_values(by='date')
#Lets see how the reviews progressed with time although there might not be much 
#since the timespan is so short
#points = df[['rating','date']]
plt.figure(figsize=(20,20))
plt.scatter(df.date,df.rating)
#The number of reviews seems to get bigger with time so lets check that out
numrev = df.date.value_counts()
numrev = numrev.sort_index()
plt.plot(numrev)
plt.title('number of reviews per day')
numrev.idxmax()
numrev[180730]
#One thing to check out is why there are all of a sudden 1603 reviews that day.