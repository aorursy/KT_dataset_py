# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline
crime = pd.read_csv('../input/Crime1.csv')

crime.head()
crime.Category.value_counts()
# Create a dataframe containing the Category counts

category = pd.DataFrame(list(zip(crime.Category.value_counts().index,crime.Category.value_counts())), columns=['Category','value'], index=None)
# Generating the wordcloud with the values under the category dataframe

catcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=1200,

                          height=800

                         ).generate(" ".join(category['Category'].values))
plt.imshow(catcloud, alpha=0.8)

plt.axis('off')

plt.show()
# Generating the factorplot

sns.factorplot(x='value', y = 'Category', data=category,kind="bar", size=4.25, aspect=1.9, palette="cubehelix")

plt.title('Factorplot of the category of crime and number of occurences ')
crime.Descript.value_counts()
descript = pd.DataFrame(list(zip(crime.Descript.value_counts().index,crime.Descript.value_counts())), columns=['Description','counts'], index=None)
descloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=1500,

                          height=1400

                         ).generate(" ".join(descript['Description'].values))
plt.imshow(descloud,alpha=0.8)

plt.axis('off')

plt.show()
DOW = pd.DataFrame(list(zip(crime.DayOfWeek.value_counts(),crime.DayOfWeek.value_counts().index)), columns=['count','Day'], index=None)
sns.factorplot(x="count", y="Day", data = DOW, kind="bar", size=4, aspect=1.9)
crime.PdDistrict.value_counts()
district = pd.DataFrame(list(zip(crime.PdDistrict.value_counts().index,crime.PdDistrict.value_counts())), columns=['District','count'], index=None)
sns.factorplot(x="count", y="District", data = district, kind="bar", size=4, aspect=1.9, palette='PuBuGn_d')
# Create the dataframe just for the Resolution data and the aggregation

Resolution = pd.DataFrame(list(zip(crime.Resolution.value_counts().index,crime.Resolution.value_counts())), columns=['resolution','value'], index=None)
rescloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=1500,

                          height=1400

                         ).generate(" ".join(Resolution['resolution'].values))
plt.imshow(rescloud, alpha=0.8)

plt.axis('off')

plt.show()
sns.factorplot(x='value' , y = 'resolution', data=Resolution, kind="bar", size=3.25, aspect=2.5, palette='BuGn_r')
# Importing the lag_plot plotting function

from pandas.tools.plotting import lag_plot

# Lag_plot for X coordinate

plt.figure()

lag_plot(crime.X)
lag_plot(crime.Y, c='goldenrod')
from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(crime.X, color='k', marker='.', linewidth='0.25')

autocorrelation_plot(crime.Y, color='goldenrod',marker='.', linewidth='0.15')

plt.ylim(-0.15,0.15)