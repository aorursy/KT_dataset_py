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
import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



% matplotlib inline
data = pd.read_csv('../input/world-cities.csv')
data.head()
#lets see which countries on top

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=3500,

                          height=3000

                         ).generate("".join(data['country']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#lets see which country has highest no of cities according to data

Country_city = data.groupby('country')['name'].count().reset_index().sort_values(by='name',ascending=False).reset_index(drop=True)

Country_city
#now lets find the percentage share of each country

summ = Country_city['name'].sum()
summ
a= []



for i in range(0,len(Country_city)):

    a.append((Country_city.iloc[i,1]/summ)*100)

a
Country_city['percentage_share'] = a
Country_city
#now lets check which indian states are most no countries

india = data.loc[data['country']=='India']
india
#lets check by the indian states



states_city = india.groupby('subcountry')['name'].count().reset_index().sort_values(by='name',ascending=False).reset_index(drop=True)

states_city
plt.figure(figsize=(13,13))

india.subcountry.value_counts().plot(kind='pie',autopct='%1.1f%%')

plt.title('Number of appearances in dataset')
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=2500,

                          height=2000

                         ).generate("".join(india['subcountry']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#now you see that uttar pradesh is on top

plt.figure(figsize=(13,13))

sns.barplot(x='name',y='subcountry',data=states_city)

plt.xlabel('no of cities')

plt.ylabel('states name')

plt.title('No of cities in each state')


