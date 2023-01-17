# import modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import operator



%matplotlib inline



# load data

from subprocess import check_output

data = pd.read_csv('../input/un-general-debates.csv')
print('data shape:', data.shape)

data.head()
# look at first and last years in data

data.sort_values(by='year')[:3]
data.sort_values(by='year')[-3:]
# convert text data to lower case (for easier analysis)

data['text'] = data['text'].str.lower()



# remove all data before 1971 -- looks like it might be incomplete

data = data[data['year'] > 1970]
# create features from meta text data

data['char_count'] = data['text'].str.len()

data['words'] = data['text'].str.split(' ')

data['sentences'] = data['text'].str.split('.')

data['word_count'] = data['words'].str.len()

data['sentence_count'] = data['sentences'].str.len()

data['word_length'] = data['char_count'] / data['word_count']

data['sentence_length'] = data['word_count'] / data['sentence_count']



print('avg char count:', data['char_count'].mean())

print()

print('avg word count:', data['word_count'].mean())

print('avg word length:', data['word_length'].mean())

print()

print('avg sentence count:', data['sentence_count'].mean())

print('avg sentence len:', data['sentence_length'].mean())
# Look at these meta text features' distributions

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

all_data = [data]



axes[0, 0].violinplot(data['word_count'], showmeans=True, showmedians=False)

axes[0, 0].set_title('Word Count per Speech')



axes[0, 1].violinplot(data['word_length'], showmeans=True, showmedians=False)

axes[0, 1].set_title('Avg. Word Length per Speech')



axes[1, 0].violinplot(data['sentence_count'], showmeans=True, showmedians=False)

axes[1, 0].set_title('Sentence Count per Speech')



axes[1, 1].violinplot(data['sentence_length'], showmeans=True, showmedians=False)

axes[1, 1].set_title('Avg. Sentence Length per Speech')



# add x-tick labels

plt.setp(axes, xticks=[y+1 for y in range(len(all_data))], xticklabels=['Frequency'])

fig.subplots_adjust(wspace=.5, hspace=.5)

plt.show()
# show top and bottom 5 countries by avg sentence count, avg word count, and avg sentence length

a = data['sentence_count'].groupby(data['country']).mean()

print('Avg. Sentence Count')

print(pd.concat([a.sort_values(ascending=False)[:3], a.sort_values(ascending=False)[-3:]], axis=0))

print()



a = data['word_count'].groupby(data['country']).mean()

print('Avg. Word Count')

print(pd.concat([a.sort_values(ascending=False)[:3], a.sort_values(ascending=False)[-3:]], axis=0))

print()



a = data['sentence_length'].groupby(data['country']).mean()

print('Avg. Sentence Length')

print(pd.concat([a.sort_values(ascending=False)[:3], a.sort_values(ascending=False)[-3:]], axis=0))

print()
# create list of topics that UN member nations might discuss

topics = [' nuclear', ' weapons', ' nuclear weapons', ' chemical weapons', 

          ' biological weapons', ' mass destruction', ' peace', ' war',

          ' nuclear war', ' civil war', ' terror', ' genocide', ' holocaust',

          ' water', ' famine', ' disease', ' hiv', ' aids', ' malaria', ' cancer',

          ' poverty', ' human rights', ' abortion', ' refugee', ' immigration',

          ' equality', ' democracy', ' freedom', ' sovereignty', ' dictator',

          ' totalitarian', ' vote', ' energy', ' oil',  ' coal',  ' income',

          ' economy', ' growth', ' inflation', ' interest rate', ' security',

          ' cyber', ' trade', ' inequality', ' pollution', ' global warming',

          ' hunger', ' education', ' health', ' sanitation', ' infrastructure',

          ' virus', ' regulation', ' food', ' nutrition', ' transportation',

          ' violence', ' agriculture', ' diplomatic', ' drugs', ' obesity',

          ' islam', ' housing', ' sustainable', 'nuclear energy']



dictionary = {}



for i in topics:    

    dictionary[i] = data['year'][data['text'].str.contains(i)].count() / len(data) * 100



sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)



labels = [i[0] for i in sorted_dictionary]

values = [i[1] for i in sorted_dictionary]

xs = np.arange(len(labels))



width = .85

plt.figure(figsize=(18, 9))

plt.tick_params(axis='both', which='major', labelsize=12)

plt.tick_params(axis='both', which='minor', labelsize=12)

plt.xticks(rotation=80)

plt.xlabel('Topics')

plt.ylabel('% of Debates Mentioned')

plt.title('Bar Plot of Topics Mentioned')



plt.bar(xs, values, width, align='center')

plt.xticks(xs, labels)

plt.show()
# # Percentage of time mentioned in a debate

# sorted_dictionary
# # UN Members

# data['country'].unique()
# count number of debates in which a nation participates

countries = data['year'].groupby(data['country']).count()

countries = pd.DataFrame(countries.reset_index(drop=False))

countries.columns = ['country', 'num speeches']



print('Most Vocal Member Nations')

print('max number of speeches given:', countries['num speeches'].max())

print(countries[countries['num speeches'] == countries['num speeches'].max()].country.unique())

print()



countries = countries.sort_values(by='num speeches')

print('Least Vocal Member Nations')

print('min number of speeches given:', countries['num speeches'].min())

print(countries.country[:10].unique().tolist())
# Plot distribution of number of speeches per country

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

all_data = [countries['num speeches']]

axes.violinplot(countries['num speeches'], showmeans=True, showmedians=False)

axes.set_title('UN Speeches Given per Country')



plt.setp(axes, xticks=[y+1 for y in range(len(all_data))], xticklabels=['Frequency'])

plt.show()
# Plot a horizontal bar graph displaying the frequency of a given topic by country

def freqMentioned (df, country_list, topic_list, colors):

    data = df.loc[df['country'].isin(country_list)]



    for i in topic_list:

        data[i] = data['text'].str.contains(i)

        data[i].loc[data[i] == False] = np.nan



    country = country_list[0]

    data_out = pd.DataFrame(data.loc[data['country'] == country].count())

    data_out = (data_out.T)[topic_list]

    

    # sort the columns by summed occurence in countries specified

    countries = country_list.copy()

    countries.remove(country)



    for i in countries:

        a = pd.DataFrame(data.loc[data['country'] == i].count())

        a = (a.T)[topic_list].copy()

        data_out = pd.concat([data_out, a], axis=0)



    dictionary = {}

    

    for i in topic_list:

        dictionary[i] = data_out[i].sum()

        

    sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)

    data_out = data_out[[i[0] for i in sorted_dictionary]]

    data_out.index = country_list

    data_out.T.plot(kind="barh", width=.6, stacked=True, figsize = (10, len(topic_list)/3), color=colors).legend(bbox_to_anchor=(1, 1))

    

    return data_out
# Permanent Members of the UN Security Council

sec_council = ['USA', 'RUS', 'GBR', 'FRA', 'CHN']



topics = []

for i in sorted_dictionary:

    if i[1] > 1:

        topics.append(i[0])



colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'r']

freqMentioned(data, sec_council, topics, colors)
# Korea & Neighbors

countries = ['USA', 'RUS', 'KOR', 'PRK', 'JPN', 'CHN']



topics = [' united states', ' north korea', ' south korea', ' south china sea', ' asia', ' peace',

          ' navy', ' arsenal', ' aggression', ' treaty', ' dmz', ' missile',

          ' nuclear material', ' sanction', ' terror', ' border', ' seoul',

          ' growth', ' democracy', 'tokyo', ' intervention', ' human right',

          ' recognition', ' ally', ' partner', ' japan', ' china', ' proliferation',

          ' conflict', ' russia', ' rice', ' grain', ' starvation']



colors = ['dodgerblue', 'orange', 'navy', 'pink', 'lightseagreen', 'r']

freqMentioned(data, countries, topics, colors)
# Middle East

countries = ['USA', 'ISR', 'IRQ', 'EGY', 'PSE', 'JOR', 'LBN', 'SYR']



topics = [' iran', ' iraq', ' israel', ' egypt', ' palestine', ' terror', ' jordan',

          ' refugee', ' hamas', ' lebanon', ' saudi', ' oil', ' nuclear material',

          ' sanction', ' settlements', ' gaza', ' opec', ' foreign investment',

          ' desalination', ' syria', ' dictator', ' democracy', ' gender equality',

          ' islam', ' jewish', ' judaism', ' jerusalem', ' religion', ' christianity',

          ' mecca', ' sunni', ' shia', ' solar', ' instability', ' civil war', ' peace',

          ' partner', ' ally', ' arab', ' conflict', ' jew', ' free speech']



colors = ['dodgerblue', 'navy', 'r', 'green', 'black', 'orange', 'purple', 'grey']

freqMentioned(data, countries, topics, colors)
# BRINC + USA + Canada

countries = ['USA', 'RUS', 'CAN', 'NGA', 'IND', 'BRA', 'CHN']



topics = [' oil', ' water', ' crude', 'coal', ' gas', ' solar', 'wheat',

          ' tin ', ' diamond', ' wood', 'ivory', ' plutonium', ' fossil',

          ' pollution', ' carbon', ' global warming', ' climate change', 

          ' fertilizer', ' aluminum', ' steel', ' iron', ' timber', ' siicon',

          ' gold', ' silver', ' copper', ' lithium', ' salt', ' magnesium',

          ' rubber', ' paper', ' plastic', ' glass', ' nickel', ' grain',

          ' fruit', ' tariff', ' fish ', ' port ', ' leather', ' nut']



colors = ['dodgerblue', 'orange', 'navy', 'pink', 'mediumseagreen', 'silver', 'r']

freqMentioned(data, countries, topics, colors)
topics = []

for i in sorted_dictionary:

    if i[1] > 1:

        topics.append(i[0])



def textFreq(df, topic_list):

    data = df.copy()

    for i in topic_list:

        data[i] = data['text'].apply(lambda x: x.count(i))

    return data



def GroupFreq(df, topic_list, grouping_column):

    topic_freq = textFreq(df, topic_list)

    topic_list.append(grouping_column)

    topic_freq = topic_freq[topic_list]

    topic_freq_col = topic_freq.groupby(topic_freq[grouping_column]).sum()

    return topic_freq_col



topic_freq_county = GroupFreq(data, topics, 'country')

plt.matshow(topic_freq_county.corr())

topic_freq_year = GroupFreq(data, topics, 'year')
topics_keep = []

for i in topic_freq_year.columns:

    if topic_freq_year[i].sum() > 20000:

        topics_keep.append(i)



topic_freq_year[topics_keep].plot()
topic_freq_year[[' economy', ' growth', ' trade', ' inflation']].plot()
topic_freq_year[[' nuclear', ' weapons', ' nuclear weapons']].plot()
topic_freq_year[[' peace', ' war', ' terror']].plot()
topic_freq_year[[' nuclear war', ' civil war', ' genocide']].plot()
topic_freq_year[[' water', ' famine', ' disease', ' hiv']].plot()
topic_freq_year[[' human rights', ' refugee', ' equality', ' democracy']].plot()
topic_freq_year[[' freedom', ' sovereignty', ' democracy', ' equality']].plot()
topic_freq_year[[' global warming', ' energy', ' water', ' oil']].plot()
# # Top five advocate countries per cause

# for i in topic_freq_county.columns:

#     topic_freq_county = topic_freq_county.sort_values(by=i, ascending=False)

#     print(i)

#     print(topic_freq_county.index.tolist()[:5])

#     print()
# # Top seven priorities for each member country

# topic_freq_county = topic_freq_county.sort_index()

# topic_freq_county = topic_freq_county.T



# for i in topic_freq_county.columns:

#     topic_freq_county = topic_freq_county.sort_values(by=i, ascending=False)

#     print(i)

#     print(topic_freq_county.index.tolist()[:7])

#     print()