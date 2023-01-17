import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/industrial-safety-and-health-analytics-database/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')

data = data.drop(['Unnamed: 0'], axis=1)## This was the original index column, there are a few missing entries which cause some issues when attempting to call a row by index, so I chose to reindex

data.tail()
crit_risk = data['Critical Risk'].value_counts()

crit_risk
## Changing the string representation of incident level to its integer value for easier plotting and comparison 

level_map = {'I': 1, 'II': 2,'III': 3 , 'IV' : 4, 'V': 5, 'VI' : 6}

data['Accident Level'] = pd.Series([level_map[x] for x in data['Accident Level']], index=data.index)

data['Potential Accident Level'] = pd.Series([level_map[x] for x in data['Potential Accident Level']], index=data.index)

data['Risk_differnce'] = data['Potential Accident Level'] - data['Accident Level']
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(data['Accident Level'])
sns.countplot(data['Risk_differnce'])
acc_level = data['Accident Level'].value_counts()

acc_level
risk_delta = data['Risk_differnce'].value_counts()

risk_delta
potential_acc_level = data['Potential Accident Level'].value_counts()

potential_acc_level
# This shows us the average value for accident level

# Notice that Accident level and Risk difference share their mean, standard deviation, 25%, 50%, and 75% values

data.describe()
## The first plot below shows the distribution of incidents

## The blue line represents the Severity of the accident

## the yellow line represents the potential severity

## The second plot we added a line to indicate the difference in risk 





plt.figure(figsize=(10,8))

sns.lineplot(y=acc_level, x=acc_level.index)

sns.lineplot(y=potential_acc_level, x=potential_acc_level.index)

plt.legend(labels=['Accident Level', 'Potential Accident Level'])





plt.figure(figsize=(10,8))

sns.lineplot(y=acc_level, x=acc_level.index)

sns.lineplot(y=potential_acc_level, x=potential_acc_level.index)

sns.lineplot(y=risk_delta, x=risk_delta.index)

plt.legend(labels=['Accident Level', 'Potential Accident Level', 'Risk Delta'])
print('Incedents by possible factor: Countries')

display(data['Countries'].value_counts())





fig, ax =plt.subplots(1,3)

ax[0].set_title('Country_01')

sns.countplot(data['Accident Level'].loc[data['Countries'] == 'Country_01'], ax=ax[0])

ax[1].set_title('Country_02')

sns.countplot(data['Accident Level'].loc[data['Countries'] == 'Country_02'], ax=ax[1])

ax[2].set_title('Country_03')

sns.countplot(data['Accident Level'].loc[data['Countries'] == 'Country_02'], ax=ax[2])
print('Incedents by possible factor: Local')

display(data['Local'].value_counts())



display(data.loc[data['Local'] == 'Local_09'])

display(data.loc[data['Local'] == 'Local_11'])



print(data.Description.iloc[119])

print('###')

print(data.Description.iloc[212])
print('Incedents by possible factor: Industry Sector')

display(data['Industry Sector'].value_counts())





fig, ax =plt.subplots(1,3)

ax[0].set_title('Mining')

sns.countplot(data['Accident Level'].loc[data['Industry Sector'] == 'Mining'], ax=ax[0])

ax[1].set_title('Metals')

sns.countplot(data['Accident Level'].loc[data['Industry Sector'] == 'Metals'], ax=ax[1])

ax[2].set_title('Others')

sns.countplot(data['Accident Level'].loc[data['Industry Sector'] == 'Others'], ax=ax[2])
print('Incedents by possible factor: Genre')

display(data['Genre'].value_counts())



fig, ax =plt.subplots(1,2)

ax[0].set_title('Male')

sns.countplot(data['Accident Level'].loc[data['Genre'] == 'Male'], ax=ax[0])

ax[1].set_title('Female')

sns.countplot(data['Accident Level'].loc[data['Genre'] == 'Female'], ax=ax[1])
print('Incedents by possible factor: Employee or Third Party')

display(data['Employee or Third Party'].value_counts())



fig, ax =plt.subplots(1,3)

ax[0].set_title('Third Party')

sns.countplot(data['Accident Level'].loc[data['Employee or Third Party'] == 'Third Party'], ax=ax[0])

ax[1].set_title('Employee')

sns.countplot(data['Accident Level'].loc[data['Employee or Third Party'] == 'Employee'], ax=ax[1])

ax[2].set_title('Third Party (Remote)')

sns.countplot(data['Accident Level'].loc[data['Employee or Third Party'] == 'Third Party (Remote)'], ax=ax[2])
import nltk

from nltk.util import ngrams

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

import re
# This creates one long string to perform n-gram operations on

super_string = data['Description'].sum()

super_string = re.sub(r'[^\w\s]','',super_string)

super_string = super_string.split(' ')

stop_words = set(nltk.corpus.stopwords.words('english'))

super_string = [word for word in super_string if word not in stop_words]

len(super_string)
# Notice how frequently hands are mentioned

word_fd = nltk.FreqDist(super_string)

bigram_fd = nltk.FreqDist(nltk.bigrams(super_string))



bigram_fd.most_common(10)
bigram_fd = nltk.FreqDist(nltk.trigrams(super_string))



bigram_fd.most_common(10)
bigram_fd = nltk.FreqDist(nltk.ngrams(super_string,7))



bigram_fd.most_common(9)
# single word counts, note how frequent the word 'hand' occurs.

bigram_fd = nltk.FreqDist(nltk.ngrams(super_string,1))



bigram_fd.most_common(9)
# Attempt with skip-grams, did not find anything signifigant



bigram_fd = nltk.FreqDist(nltk.skipgrams(super_string, n=4, k=3))



bigram_fd.most_common(9)

crit_acc = data.loc[data['Accident Level'] >= 4]

crit_acc.shape
super_hr_string = crit_acc['Description'].sum()

super_hr_string = re.sub(r'[^\w\s]','',super_hr_string)

super_hr_string = super_hr_string.split(' ')

super_hr_string = [word for word in super_hr_string if word not in stop_words]

len(super_hr_string)
bigram_fd = nltk.FreqDist(nltk.ngrams(super_hr_string,1))



bigram_fd.most_common(10)
bigram_fd = nltk.FreqDist(nltk.ngrams(super_hr_string,2))



bigram_fd.most_common(10)
print(data.Description.iloc[393])
print(data.Description.iloc[407])
print(data.Description.iloc[383])
## Now to generate the graphic that we started with



from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(super_string)

mask = np.array(Image.open("/kaggle/input/caution-hands/Hand_crush_grey_3.png"))

wordcloud_hands = WordCloud(width=1200,height=1200, prefer_horizontal=0.5,scale=2,colormap='Reds',

                            collocations=False, background_color='white', mode="RGB", min_font_size=8,

                            max_words=2500, mask=mask, contour_width=3,  contour_color='yellow').generate(text)



plt.figure(figsize=[20,20])

plt.imshow(wordcloud_hands,interpolation="bilinear")

plt.axis("off")

plt.show()