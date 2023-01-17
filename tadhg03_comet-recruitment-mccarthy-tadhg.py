# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ign.csv')

data.head()
data.drop(['Unnamed: 0','url'],axis=1,inplace=True)

data.head()
data['release_year'].value_counts()[:10].plot(kind='pie',autopct='%1.2f%%',shadow=False,explode=

                                              [0.1,0,0,0,0,0,0,0,0,0])

plt.title('Distribution of most games released per year')

fig=plt.gcf()

fig.set_size_inches(7,7)

plt.show()
data['genre'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%',shadow=False,explode=[0.1,0,0,0,0,0,0,0,0,0])

plt.title('Distribution of top gaming genres')

fig=plt.gcf()

fig.set_size_inches(7,7)

plt.show()
data.groupby('release_day')['genre'].count().plot(color='y')

fig=plt.gcf()

fig.set_size_inches(12,6)
masterpieceData=data[data['score_phrase']=='Masterpiece']
plt.subplots(figsize=(12,6))

sns.countplot(masterpieceData['platform'],palette='Set1')

plt.xticks(rotation=90)

plt.show()