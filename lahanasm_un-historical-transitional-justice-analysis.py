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
# create list of topics that UN member nations might discuss
topics = [' amnesty', ' universal jurisdiction', ' transitional justice', ' civil war', ' truth commission', 'intervention', ' communism', 'communist', ' evil', ' unjust',
          ' peacekeeping', ' trial', 'justice cascade', ' truth seeking',' invasion', ' reparations', ' extradition', ' memorial',' dictatorship', ' fascist', ' fascism', ' injustice',
          ' prosecution', ' rule of law',' vetting', ' lustration', ' disarmament', ' demobilization', ' reintegration', ' strength', ' weakness', ' strongman', ' dictator',
          ' forgiveness', ' institutional reform',' reconciliation', ' genocide', ' hague', ' war crime', ' war crimes',' human rights', 'TRC', ' amnesties', ' democracy', ' democratic',
          ' crime against humanity', ' immunity', ' sovereign immunity',' sovereign', ' exile', ' restorative', ' tribunal', 'Rome Statute', ' illegal', ' international law',
          ' justice', ' victims', ' perpetrators', ' resistance', ' military intervention',' non-intervention', ' isolationist', 'due process', ' sovereignty', ' democratization',
          ' isolationism', ' internationalist', ' tolerance', ' nuremburg',' sanctions',' crimes against humanity',' sanction', ' backslide', ' backsliding']

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
sorted_dictionary
# # UN Members
#data['country'].unique()
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
# Oceania + Asia
oceania = ['AUS', 'NZL', 'CHN','JPN','VNM','KOR','PRK', 'IDN', 'SGP','MAL','TMP','THA']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'skyblue']
freqMentioned(data, oceania, topics, colors)
# BRICS + Regional Powers
brics = ['BRA','IND','RUS','CHN']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'olive', 'crimson']
freqMentioned(data, brics, topics, colors)
# Permanent Members of the UN Security Council
sec_council = ['USA', 'RUS', 'GBR', 'FRA', 'CHN']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'r']
freqMentioned(data, sec_council, topics, colors)
# Eastern Bloc
eastern_bloc = ['POL', 'RUS', 'HUN', 'EST', 'LVA', 'ALB', 'YUG', 'ROU', 'CSK', 'DDR']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet']
freqMentioned(data, eastern_bloc, topics, colors)
# South Korea v. North Korea
germanys = ['KOR','PRK']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'skyblue']
freqMentioned(data, germanys, topics, colors)
# West and East Germany
germanys = ['DEU','DDR']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'skyblue']
freqMentioned(data, germanys, topics, colors)
# NATO Countries
nato = ['CAN', 'GBR', 'FRA', 'DNK', 'BEL', 'ISL', 'ITA', 'LUX', 'NLD', 'PRT', 'TUR', 'GRC', 'USA', 'DEU']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'skyblue']
freqMentioned(data, nato, topics, colors)
# Permanent Members of the UN Security Council... and the Netherlands
# clearly the Netherlands likes to talk about things they host
sec_council = ['USA', 'RUS', 'GBR', 'FRA', 'CHN', 'NLD']

topics = [' international criminal court', ' international court of justice', ' hague']

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'skyblue']
freqMentioned(data, sec_council, topics, colors)
# NATO  v. Eastern Bloc Countries on TJ
nato_eastern= ['CAN', 'GBR', 'FRA', 'DNK', 'BEL', 'ISL', 'ITA', 'LUX', 'NLD', 'PRT', 'TUR', 'GRC', 'USA', 'DEU', 'POL', 'RUS', 'HUN', 'EST', 'LVA', 'ALB', 'YUG', 'ROU', 'DDR', 'CSK']
topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b','b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
freqMentioned(data, nato_eastern, topics, colors)
# Latin America
latin_america = ['COL', 'ECU', 'ARG', 'CHL', 'VEN', 'BOL', 'BRA', 'MEX', 'PAN', 'URY', 'SLV', 'HND', 'PER', 'NIC',]

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'olive', 'crimson']
freqMentioned(data, latin_america, topics, colors)
# Middle East + USA on TJ
countries = ['USA', 'ISR', 'IRQ', 'EGY', 'PSE', 'JOR', 'LBN', 'SYR']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'navy', 'r', 'green', 'black', 'orange', 'purple', 'grey']
freqMentioned(data, countries, topics, colors)
# States With Open ICC Investigations/Other TJ Mechanisms
outliers = ['COD','UGA','SDN','KEN','LBY','CIV','MLI','CAF','BDI','GEO', 'SLE', 'RWA']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'olive', 'crimson']
freqMentioned(data, outliers, topics, colors)
# Historic Outliers
outliers = ['PRK','CUB','IRN','IRQ','VEN','SOM','COD','ZAF','ZWE']

topics = []
for i in sorted_dictionary:
    if i[1] > 1:
        topics.append(i[0])

colors = ['dodgerblue', 'orange', 'navy', 'lightseagreen', 'yellow', 'green', 'blue', 'maroon', 'purple', 'peru','violet', 'tomato', 'olive', 'crimson']
freqMentioned(data, outliers, topics, colors)
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
    if topic_freq_year[i].sum() > 500:
        topics_keep.append(i)

topic_freq_year[topics_keep].plot()
topic_freq_year[[ ' war crime', ' reconciliation', ' invasion', ' tolerance']].plot()
topic_freq_year[[ ' amnesty',' trial',' rule of law' ]].plot()
topic_freq_year[[ ' victims', ' perpetrators', ]].plot()
topic_freq_year[[ ' genocide', ' war crime',' crimes against humanity', ' military intervention']].plot()
topic_freq_year[[ ' amnesty', ' civil war', ' trial']].plot()
topic_freq_year[[ ' justice', ' sovereign', ' rule of law', ' human rights']].plot()
topic_freq_year[[ ' evil', ' unjust', ' injustice']].plot()
topic_freq_year[[ ' crime against humanity', ' crimes against humanity']].plot()
topic_freq_year[[ ' strength', ' weakness']].plot()
topic_freq_year[[ ' communism', 'communist']].plot()
topic_freq_year[[ ' fascism', ' fascist', ' dictator']].plot()
topic_freq_year[[ ' democracy', ' democratization', ' democratic']].plot()
# # Top five advocate countries per cause
for i in topic_freq_county.columns:
    topic_freq_county = topic_freq_county.sort_values(by=i, ascending=False)
    print(i)
    print(topic_freq_county.index.tolist()[:5])
    print()
# # Top seven priorities for each member country
topic_freq_county = topic_freq_county.sort_index()
topic_freq_county = topic_freq_county.T

for i in topic_freq_county.columns:
    topic_freq_county = topic_freq_county.sort_values(by=i, ascending=False)
    print(i)
    print(topic_freq_county.index.tolist()[:7])
    print()