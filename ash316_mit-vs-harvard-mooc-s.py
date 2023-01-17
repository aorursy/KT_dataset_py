# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mooc=pd.read_csv('../input/appendix.csv')
mooc.head(2)
mooc.isnull().sum()
print('The Different Course Subjects Are:',mooc['Course Subject'].unique())
print('The number of different Course Topics are: ',mooc['Course Title'].nunique())
mooc['Institution'].value_counts().plot.bar()
fig=plt.gcf()

subjects=mooc['Course Subject'].value_counts().index.tolist()

size=mooc['Course Subject'].value_counts().values.tolist()

plt.pie(size,labels=subjects,explode=(0,0,0,0.1),startangle=90,shadow=True,autopct='%1.1f%%')

plt.title('Course Subjects ')

fig.set_size_inches((6,6))

plt.show()
plt.subplot(211)

mit=mooc[mooc['Institution']=='MITx']

plt.pie(mit['Course Subject'].value_counts().values.tolist(),labels=mit['Course Subject'].value_counts().index.tolist(),startangle=90,explode=(0.1,0,0,0),autopct='%1.1f%%',shadow=True,colors=['Y', '#1f2ff3', '#0fff00', 'R'])

plt.title('MITx')

plt.subplot(212)

harvard=mooc[mooc['Institution']=='HarvardX']

plt.pie(harvard['Course Subject'].value_counts().values.tolist(),labels=harvard['Course Subject'].value_counts().index.tolist(),startangle=90,explode=(0.1,0,0,0),autopct='%1.1f%%',shadow=True,colors=['Y', '#1f2ff3', '#0fff00', 'R'])

plt.title('Harvardx')

fig=plt.gcf()

fig.set_size_inches((11,11))

plt.show()
mooc['Year']=pd.DatetimeIndex(mooc['Launch Date']).year  #taking only the year from the date

sns.countplot('Year',hue='Institution',data=mooc)
a=mooc.groupby(['Institution','Year'])['Participants (Course Content Accessed)'].sum().reset_index()

a.pivot('Year','Institution','Participants (Course Content Accessed)').plot.bar()

plt.show()
abc=mooc[mooc['% Female'] >= mooc['% Male']]

from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(abc['Course Title']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()
abc=mooc[mooc['% Female'] < mooc['% Male']]



wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(abc['Course Title']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()