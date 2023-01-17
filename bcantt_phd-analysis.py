# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/phd-stipends/csv')
data
import seaborn as sns

from textblob import TextBlob

import matplotlib.pyplot as plt
data = data.fillna(data.median())
data['Overall Pay'] = data['Overall Pay'].str.extract('(\d+)') + data['Overall Pay'].str.extract('(,)(\d+)')[1]
data['Overall Pay'] = data['Overall Pay'].fillna(data['Overall Pay'].median())
data['Overall Pay'] = data['Overall Pay'].astype(int) / 1000
data['12 M Gross Pay'] = data['12 M Gross Pay'].str.extract('(\d+)') + data['12 M Gross Pay'].str.extract('(,)(\d+)')[1]
data['12 M Gross Pay'] = data['12 M Gross Pay'].fillna(data['12 M Gross Pay'].median())
data['12 M Gross Pay'] = data['12 M Gross Pay'].astype(int) / 1000
data['Academic Year'] = data['Academic Year'].str.extract('(\d+)').astype(int)
ax = sns.barplot(x="Overall Pay", y="University", data=data.sort_values('Overall Pay',ascending = False).head(10))
ax = sns.barplot(x="Overall Pay", y="Department", data=data.sort_values('Overall Pay',ascending = False).head(10))
ax = sns.barplot(x="12 M Gross Pay", y="Department", data=data.sort_values('12 M Gross Pay',ascending = False).head(10))
data.groupby('University').sum().sort_values('12 M Gross Pay',ascending = False)
data['Comments'] = data['Comments'].astype(str)
uni_groups = data.groupby(['University'])['Comments'].apply(','.join).reset_index()
uni_groups['Comment Lengths'] = data.groupby(['University'])['Comments'].apply(','.join).reset_index()['Comments'].apply(len)
data = data.merge(uni_groups)
data['Comment Sentiment Polarity'] = data['Comments'].apply(lambda tweet: TextBlob(tweet).sentiment[0])

data['Comment Sentiment Subjectivity'] = data['Comments'].apply(lambda tweet: TextBlob(tweet).sentiment[1])
ax = sns.barplot(x="Comment Sentiment Polarity", y="Department", data=data.sort_values('Comment Sentiment Polarity',ascending = False).head(10))
ax = sns.barplot(x="Comment Sentiment Polarity", y="University", data=data.sort_values('Comment Sentiment Polarity',ascending = False).head(10))
ax = sns.barplot(x="Comment Sentiment Subjectivity", y="Department", data=data.sort_values('Comment Sentiment Subjectivity',ascending = False).head(10))
ax = sns.barplot(x="Comment Sentiment Subjectivity", y="University", data=data.sort_values('Comment Sentiment Subjectivity',ascending = False).head(10))
corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
corrMatrix = data.sample(100).corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
corrMatrix = data.sample(50).corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
corrMatrix = data.sample(25).corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
data = data.sort_values('Comment Sentiment Polarity',ascending = False)
corrMatrix = data.head(30).corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
corrMatrix = data.tail(30).corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()