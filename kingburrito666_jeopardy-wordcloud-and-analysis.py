import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import datetime as dt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/JEOPARDY_CSV.csv')


data.head()
data.columns
wrdcld = pd.Series(data[' Question'].tolist()).astype(str)

# Most frequent words in the data set. Using a beautiful wordcloud

cloud = WordCloud(width=900, height=900,

                  stopwords=('No.', 'href', 'http',

                             'www', 'com', 'target',

                             'jpg', '_blank', '<a', 'archive'),

                  colormap='rainbow').generate(''.join(wrdcld.astype(str)))

plt.figure(figsize=(15, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
data[' Round'].unique()

rounds = data.groupby(' Round')[' Question'].size().rename('Counts').sort_values(ascending=False).reset_index()

rounds
fig, ax = plt.subplots(figsize=(12, 12))

shap = rounds['Counts']

labels = 'Jeopardy!','Double Jeopardy!','Final Jeopardy!','Tiebreaker'

explode = (0.1, 0.1, 0.1, 0.3)

ax.pie(shap, explode=explode,

       labels=labels, shadow=True,

       autopct='%1.5f%%', startangle=40)



plt.title('Percentages of Round types')

plt.tight_layout()

plt.show()
dates = data.copy()

dates[' Air Date'] = pd.to_datetime(dates[' Air Date'])

dates[' Air Date'] = dates[' Air Date'].dt.year

dates = dates.sort_values(by=' Air Date')

dates.head()
dates = dates.groupby(' Air Date')[' Question'].size().rename('Counts').sort_values(ascending=True).reset_index()
fig, ax = plt.subplots(figsize=(12, 12))

shap = dates['Counts']

lbs = '1984', '1985', '1986','1987', '1988', '1989', '1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012'

ax.pie(shap,

       shadow=True, labels=lbs,

       autopct='%1.1f%%', startangle=90)



plt.title('Questions Per Year')

plt.tight_layout()

plt.show()
categories = data.groupby(' Category')[' Question'].size().rename('Q-Counts').sort_values(ascending=True).reset_index()

plt.subplots(figsize=(5, 15))

plt.title('Top 10 categories')

sns.barplot(y=' Category', x='Q-Counts',data=categories.sort_values(by='Q-Counts',ascending=False).head(10))

plt.show()