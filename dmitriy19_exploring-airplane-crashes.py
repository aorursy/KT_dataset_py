import pandas as pd

import seaborn as sns

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from gensim import corpora, models



from collections import Counter
data = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])

data['Day'] = data['Date'].map(lambda x: x.day)

data['Year'] = data['Date'].map(lambda x: x.year)

data['Month'] = data['Date'].map(lambda x: x.month)
crashes_per_year = Counter(data['Year'])

years = list(crashes_per_year.keys())

crashes_year = list(crashes_per_year.values())
crashes_per_day = Counter(data['Day'])

days = list(crashes_per_day.keys())

crashes_day = list(crashes_per_day.values())
def get_season(month):

    if month >= 3 and month <= 5:

        return 'spring'

    elif month >= 6 and month <= 8:

        return 'summer'

    elif month >= 9 and month <= 11:

        return 'autumn'

    else:

        return 'winter'



data['Season'] = data['Month'].apply(get_season)
crashes_per_season = Counter(data['Season'])

seasons = list(crashes_per_season.keys())

crashes_season = list(crashes_per_season.values())
sns.set(style="whitegrid")

sns.set_color_codes("pastel")



fig = plt.figure(figsize=(14, 10))



sub1 = fig.add_subplot(211)

sns.barplot(x=years, y=crashes_year, color='g', ax=sub1)

sub1.set(ylabel="Crashes", xlabel="Year", title="Plane crashes per year")

plt.setp(sub1.patches, linewidth=0)

plt.setp(sub1.get_xticklabels(), rotation=70, fontsize=9)



sub2 = fig.add_subplot(223)

sns.barplot(x=days, y=crashes_day, color='r', ax=sub2)

sub2.set(ylabel="Crashes", xlabel="Day", title="Plane crashes per day")



sub3 = fig.add_subplot(224)

sns.barplot(x=seasons, y=crashes_season, color='b', ax=sub3)

texts = sub3.set(ylabel="Crashes", xlabel="Season", title="Plane crashes per season")



plt.tight_layout(w_pad=4, h_pad=3)
survived = []

dead = []

for year in years:

    curr_data = data[data['Year'] == year]

    survived.append(curr_data['Aboard'].sum() - curr_data['Fatalities'].sum())

    dead.append(curr_data['Fatalities'].sum())
f, axes = plt.subplots(2, 1, figsize=(14, 10))



sns.barplot(x=years, y=survived, color='b', ax=axes[0])

axes[0].set(ylabel="Survived", xlabel="Year", title="Survived per year")

plt.setp(axes[0].patches, linewidth=0)

plt.setp(axes[0].get_xticklabels(), rotation=70, fontsize=9)



sns.barplot(x=years, y=dead, color='r', ax=axes[1])

axes[1].set(ylabel="Fatalities", xlabel="Year", title="Dead per year")

plt.setp(axes[1].patches, linewidth=0)

plt.setp(axes[1].get_xticklabels(), rotation=70, fontsize=9)



plt.tight_layout(w_pad=4, h_pad=3)
oper_list = Counter(data['Operator']).most_common(12)

operators = []

crashes = []

for tpl in oper_list:

    if 'Military' not in tpl[0]:

        operators.append(tpl[0])

        crashes.append(tpl[1])

print('Top 10 the worst operators')

pd.DataFrame({'Count of crashes' : crashes}, index=operators)
loc_list = Counter(data['Location'].dropna()).most_common(15)

locs = []

crashes = []

for loc in loc_list:

    locs.append(loc[0])

    crashes.append(loc[1])

print('Top 15 the most dangerous locations')

pd.DataFrame({'Crashes in this location' : crashes}, index=locs)
summary = data['Summary'].tolist()

punctuation = ['.', ',', ':']

texts = []



for text in summary:

    cleaned_text = str(text).lower()   

    for mark in punctuation:

        cleaned_text = cleaned_text.replace(mark, '')       

    texts.append(cleaned_text.split())
dictionary = corpora.Dictionary(texts)
word_list = []

for key, value in dictionary.dfs.items():

    if value > 100:

        word_list.append(key)
dictionary.filter_tokens(word_list)

corpus = [dictionary.doc2bow(text) for text in texts]
np.random.seed(76543)

lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=5)
topics = lda.show_topics(num_topics=10, num_words=15, formatted=False)

for topic in topics:

    num = int(topic[0]) + 1

    print('Cause %d:' % num, end=' ')

    print(', '.join([pair[0] for pair in topic[1]]))