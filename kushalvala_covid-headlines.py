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
import matplotlib.pyplot as plt

import re
#Reading the input file and dropping a redundant column 

data = pd.read_csv('/kaggle/input/times-of-india-headlines-2020/2020_headlines.csv')

data.drop(columns=['Unnamed: 0'], inplace= True)
# Lower-casing of headlines for text normalisation

data['headlines'] = data['headlines'].apply(lambda x : x.lower())
data.head()
def TextMatch(x):

    patterns = ['covid-19', 'covid', 'corona','corona virus']

    result = []

    for pattern in patterns:

        if re.search(pattern, x):

            result.append(1)

        else:

            result.append(0)

    if np.sum(result) != 0:

        return 1

    else:

        return 0
data['Flag'] = data['headlines'].apply(lambda x : TextMatch(x))
data.head()
data['Flag'].value_counts()
data['date-time'] = pd.to_datetime((data.year*10000+data.month*100+data.day).apply(str),format='%Y%m%d')
data.head()
data.drop(columns=['year','month','day'], inplace= True)
fig, ax = plt.subplots(figsize=(15,7))

data.groupby(['date-time']).sum()['Flag'].plot(ax=ax)

plt.grid()

plt.ylabel('Count of COVID Mentions')

#plt.xlabel('Date')
from wordcloud import WordCloud,STOPWORDS
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = STOPWORDS)
wordcloud.generate(' '.join(data['headlines']))
plt.figure(figsize=(20,20))

plt.imshow(wordcloud)

plt.axis('off')