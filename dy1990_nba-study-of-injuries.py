#Import packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.graph_objects as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#read the dataset
data = pd.read_csv("/kaggle/input/nba-injuries-2010-2018/injuries_2010-2020.csv",parse_dates=[0])

print(data.info())
print(data.columns)
data.describe()
data.head(3)
injuries = data[data.Team.notnull()]
list(set(injuries['Acquired'].values))
list(set(injuries['Relinquished'].values))
injuries.head(10)
noEmptyValuesInAcquired= injuries[injuries.Acquired.notnull()]
list(set(noEmptyValuesInAcquired['Notes'].values))
noEmptyValuesInRelinquished= injuries[injuries.Relinquished.notnull()]
list(set(noEmptyValuesInRelinquished['Notes'].values))
newDatasetInjuries = injuries[injuries.Relinquished.notnull()]
newDatasetInjuries=newDatasetInjuries.drop(['Acquired'],axis=1)
newDatasetInjuries.info()
newDatasetInjuries.head()
import datetime as dt
pastSevenSeasons=newDatasetInjuries[(newDatasetInjuries.Date.dt.year <= 2020) & (newDatasetInjuries.Date.dt.year >=2013)]
sns.countplot(y="Team",data=pastSevenSeasons,order=pastSevenSeasons.Team.value_counts().iloc[:7].index,palette="Set2")
sns.countplot(y="Relinquished",data=pastSevenSeasons,order=pastSevenSeasons.Relinquished.value_counts().iloc[:10].index,palette="Set3")
chrisPaul=newDatasetInjuries[(newDatasetInjuries.Relinquished == "Chris Paul")]
chrisPaul.head()
list(set(chrisPaul['Notes'].values))

from collections import Counter
import pandas as pd
import nltk

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('left')
stopwords.append('right')
stopwords.append('(dnp)')
stopwords.append('(dtd)')
stopwords.append('sore')
stopwords.append('sprained')
stopwords.append('injury')
stopwords.append('strained')
stopwords.append('()')
stopwords.append('rest')
stopwords.append('surgery')
stopwords.append('indefinitely')
stopwords.append('season')
stopwords.append('bruised')
stopwords.append('torn')
stopwords.append('repair')
stopwords.append('illness')
stopwords.append('fractured')
stopwords.append('for')
stopwords.append('lower')
stopwords.append('/')
# RegEx for stopwords
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
for word in stopwords:
    stopwords.remove(word)
# replace '|'-->' ' and drop all stopwords
words = (newDatasetInjuries.Notes
           .str.lower()
            .replace([r'\|', RE_stopwords], [' ', ''], regex=True)
           .str.cat(sep=' ')
         
           .split()
)

mostFrequentWordsInNotes = pd.DataFrame(Counter(words).most_common(10),
                    columns=['Word', 'Frequency']).set_index('Word')

# plot
mostFrequentWordsInNotes.plot.bar(rot=0, figsize=(10,10), width=0.8)
from collections import Counter
import nltk

pelicans=newDatasetInjuries[(newDatasetInjuries.Team=="Pelicans")]

stopwords = nltk.corpus.stopwords.words('english')
# Regular expressions for stopwords
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
# drop all stopwords
words = (pelicans.Notes
           .str.lower()
           .str.cat(sep=' ')
           .split()
)


mostFrequentWordsInNotes = pd.DataFrame(Counter(words).most_common(10),
                    columns=['Word', 'Frequency']).set_index('Word')


# plot
mostFrequentWordsInNotes.plot.bar(rot=0, figsize=(10,10), width=0.8)
newDatasetInjuries['season'] = pd.DatetimeIndex(newDatasetInjuries['Date']).year
newDatasetInjuries.head()
sns.countplot(y='season',data=newDatasetInjuries,palette='Set2',order=newDatasetInjuries.season.value_counts().iloc[:].index)
newDatasetInjuries['month'] = pd.DatetimeIndex(newDatasetInjuries['Date']).month
sns.countplot(y='month',data=newDatasetInjuries,palette='Blues_d',order=newDatasetInjuries.month.value_counts().iloc[:5].index)
season2012=newDatasetInjuries[(newDatasetInjuries.season==2012)]
sns.countplot(y='Relinquished',data=season2012,palette='Set2',order=season2012.Relinquished.value_counts().iloc[:5].index)
sns.countplot(y='Team',data=season2012,palette='Set3',order=season2012.Team.value_counts().iloc[:10].index)