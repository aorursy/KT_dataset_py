# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# We have added our own imports as well



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json # helpful in reading in data

%matplotlib inline

import matplotlib.pyplot as plt #for visualizations

import seaborn as sns 



# Plot settings

plt.rcParams['figure.figsize'] = (12, 9)

plt.rcParams['font.size'] = 12



import glob



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input"))
metadata = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')



path = '/kaggle/input/CORD-19-research-challenge'



files = [f for f in glob.glob(path + "**/**/*.json", recursive=True)]



"""The text data for each paper is separated into multiple sections, and so we combine

it with this function for ease of analysis."""

def combine_text(text_dict):

    text_lst = [lst['section'] + ": " + lst['text'] if lst['section'] else "Unnamed: " + lst['text'] for lst in text_dict]

    return "\n".join(text_lst)   

    

data_dict = {'title' : [], 'paper_id' : [], 'text' : []} # We exclude abstract for now



for file in files:

    with open(file) as curr_json:

        curr_data = json.load(curr_json)

        # We want paper_id, metadata, and body_text as columns. We extract the title from metadata.

        # The keys body_text and abstract have values which are lists of dictionaries, with each dictionary containing the following keys:

        # text, cite_spans, ref_spans, and section. We will combine these into a single string, with each text entry separated

        # by a newline and prefaced by its section (if the section name exists). For now, we ignore cite_spans and ref_spans.

        data_dict['title'].append(curr_data['metadata']['title'])

        data_dict['paper_id'].append(curr_data['paper_id'])

        # data_dict['abstract'].append(combine_text(curr_data['abstract']))

        data_dict['text'].append(combine_text(curr_data['body_text']))



cord_data = pd.DataFrame(data_dict)

cord_data
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')

doc_term_matrix = count_vect.fit_transform(cord_data['title'].values.astype('U'))

doc_term_matrix
from sklearn.decomposition import LatentDirichletAllocation



LDA = LatentDirichletAllocation(n_components=10, random_state=42)

LDA.fit(doc_term_matrix)
# We print out LDA.components_, a 2-D array containing the probability values of each word in our dictionary for each topic

# As one can see, it is not very informative in the current format

LDA.components_
for i,topic in enumerate(LDA.components_):

    print(f'Top 20 words for topic #{i+1}:')

    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-1:-21:-1]])

    print('\n')

choice_topic = LDA.components_[3]

feature_names = count_vect.get_feature_names()    

top_150_topic = [feature_names[i] for i in choice_topic.argsort()[-1:-151:-1]]



# Citation: https://www.datacamp.com/community/tutorials/wordcloud-python

from wordcloud import WordCloud

top_150_topic_indices = choice_topic.argsort()[-1:-151:-1] # The indices of the top ten words

freqs = [int(choice_topic[i]) for i in top_150_topic_indices] # estimate of frequencies of the words

double_lst = [[top_150_topic[i]] * int(freqs[i] / 150) for i in range(150)]

multitext = " ".join([item for lst in double_lst for item in lst]) # Combine text into the string format needed for WordCloud

# Create and generate a word cloud image:

wordcloud = WordCloud(background_color=None, mode='RGBA').generate(multitext)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
top_ten = [feature_names[i] for i in choice_topic.argsort()[-1:-11:-1]]

public_health = cord_data[cord_data['title'].str.contains('|'.join(top_ten))] # This regex checks for any of the top ten words

public_health
lexicon = pd.read_csv('../input/vader-lexicon/vader_lexicon/vader_lexicon.txt', sep='\t', names = ['Words', 'polarity', 'drop1', 'drop2'], header=None)

lexicon = lexicon.drop(columns=['drop1', 'drop2']).set_index('Words').rename_axis(None)

lexicon.head()
punctuation_regex = r'[^\w\s]' # Searches the dataset for anything not a word character or a space character

public_health['raw title'] = public_health['title'].str.replace(punctuation_regex, ' ').str.lower()

public_health
tidy_format = pd.DataFrame(public_health.loc[:, ['raw title']]['raw title'].str.split(expand=True).stack()).reset_index(level=1)

tidy_format = tidy_format.rename_axis(None).rename(columns={'level_1':'num', 0:'word'})

tidy_format
public_health = public_health.sort_index()

with_pols = tidy_format.merge(lexicon, how='left', left_on='word', right_index=True).sort_index() # Merge tidy table with lexicon to match words with polarity

# The index below identifies each unique article/paper

# Thus, grouping by it gives us a total polarity value for the title of each article

# Notice we divide by the count to normalize values

polarities = with_pols.reset_index().groupby('index').agg('sum').sort_index()['polarity']

polarities = (polarities - np.mean(polarities)) / np.std(polarities)

public_health['polarity'] = polarities

public_health
print('Most positive titles:')

for t in public_health.sort_values('polarity', ascending=False).head(20)['title']:

    print('\n  ', t)
print('Most negative titles:')

for t in public_health.sort_values('polarity').head(20)['title']:

    print('\n  ', t)
text1 = public_health[public_health['title'] == 'Management of a Large Middle East Respiratory Syndrome Outbreak in a Tertiary Care Hospital: a Qualitative Case Study Management of a Large Middle East Respiratory Syndrome Outbreak Management of a Large Middle East Respiratory Syndrome Outbreak in a Tertiary Care Hospital: a Qualitative Case Study Perceptions of Management of a Large Middle East Respiratory Syndrome Outbreak in a Tertiary Care Hospital: a Qualitative Study Management of a Large Middle East Respiratory Syndrome Outbreak Perceptions of Post-break out Management by management and healthcare workers of a Middle East Respiratory Syndrome Outbreak in a Tertiary Care Hospital: a Qualitative Study Management of a Large Middle East Respiratory Syndrome Outbreak']

print(text1['text'].values[0])
virus_cases = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

virus_cases[virus_cases['Country/Region'].str.contains('hina')]
# Note that all values are cumulative

# It should be noted that values below are estimates, because it is possible that there was not an update every day in each province

# Our goal is simply to get an idea of the curve's general structure

china = virus_cases[virus_cases['Country/Region'] == 'China']

china.loc[:, 'Date'] = pd.to_datetime(china['Date']) # Will allow for ease of sorting later on, should we need it

china.loc[:, 'Date'] = china['Date'].dt.date

china = china.groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum() # Aggregate values from different provinces together

sns.lineplot(x=np.arange(china.shape[0]), y=china['Confirmed'], label = 'Confirmed')

sns.lineplot(x=np.arange(china.shape[0]), y=china['Recovered'], label = 'Recovered')

sns.lineplot(x=np.arange(china.shape[0]), y=china['Deaths'], label = 'Deaths')

plt.xlabel('Days since Jan 21st')

plt.ylabel('Count')

plt.title('Confirmed Cases, Recoveries, and Deaths for China')

plt.legend()
# Clean up format of data

virus_cases.loc[:, 'Date'] = pd.to_datetime(virus_cases['Date'])

virus_cases['Province/State'] = virus_cases.fillna('No specified province/state')

virus_cases = virus_cases.sort_values('Date', ascending=False)

by_country = virus_cases.groupby(['Country/Region', 'Province/State']).first() # Captures necessary info because values cumulative

by_country.reset_index(inplace=True)

by_country = by_country.groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered']].sum()

by_country.head(40)
top_ten_countries = by_country.sort_values('Confirmed', ascending=False).head(10)

top_ten_countries
sns.barplot(x=top_ten_countries.index, y=top_ten_countries['Confirmed'])

plt.title('Confirmed Cases for Top Ten Most Affected Countries')
united_states = pd.read_csv('../input/covid19-in-usa/us_states_covid19_daily.csv')

united_states.head()
from datetime import datetime



cases = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

us = cases[cases['Country/Region'] == 'US']

us.loc[:, 'Last Update'] = pd.to_datetime(us['Last Update'])

us.loc[:, 'Last Update'] = us['Last Update'].dt.date

us = us.groupby('Last Update')[['Confirmed', 'Deaths', 'Recovered']].sum()

#This is a bad way to group, because days that do not have an update will be inaccurate

#For now, we will drop the main outliers--March 8 and March 13

us = us.reset_index()

us = us[(us['Last Update'] != datetime.strptime('2020-03-08', '%Y-%m-%d').date())

       & (us['Last Update'] != datetime.strptime('2020-03-13', '%Y-%m-%d').date())]

sns.lineplot(x=np.arange(us.shape[0]), y=us['Confirmed'], label = 'Confirmed')

sns.lineplot(x=np.arange(us.shape[0]), y=us['Recovered'], label = 'Recovered')

sns.lineplot(x=np.arange(us.shape[0]), y=us['Deaths'], label = 'Deaths')

plt.xlabel('Days since first observation')

plt.ylabel('Count')

plt.title('Confirmed Cases, Recoveries, and Deaths for United States')

plt.legend()
us_cases = cases[cases['Country/Region'] == 'US']

us_cases['Last Update'] = pd.to_datetime(us_cases['Last Update'])



# Filter dataframe to include a single state

def get_state_df(state):

    state_df = us_cases[us_cases['Province/State'] == state]

    state_df.loc[:, 'Last Update'] = state_df['Last Update'].dt.date

    state_df = state_df.sort_values('Last Update', ascending = False)

    state_df.groupby('Last Update')[['Confirmed', 'Deaths', 'Recovered']].first() # Takes the latest reading from each day

    state_df.reset_index(inplace=True)

    state_df = state_df.sort_values('Last Update', ascending = True) # Puts data back in chronological order

    # The data has March 22nd mislabeled as March 8th. It leads to a misrepresentative visualization, so we drop it

    state_df = state_df[state_df['Last Update'] != datetime.strptime('2020-03-08', '%Y-%m-%d').date()]

    return state_df



#Draw growth curve for state

def visualize_state(state):

    state_df = get_state_df(state)

    sns.lineplot(x=np.arange(state_df.shape[0]), y=state_df['Confirmed'], label = 'Confirmed')

    sns.lineplot(x=np.arange(state_df.shape[0]), y=state_df['Recovered'], label = 'Recovered')

    sns.lineplot(x=np.arange(state_df.shape[0]), y=state_df['Deaths'], label = 'Deaths')

    plt.xlabel('Days since 1st Measurement for ' + state)

    plt.ylabel('Count')

    plt.title('Growth Curve for ' + state)

    plt.legend()



cali = get_state_df('California')

ny = get_state_df('New York')

washington = get_state_df('Washington')

sns.lineplot(x=np.arange(cali.shape[0]), y=cali['Confirmed'], label = 'California')

sns.lineplot(x=np.arange(ny.shape[0]), y=ny['Confirmed'], label = 'New York')

sns.lineplot(x=np.arange(washington.shape[0]), y=washington['Confirmed'], label = 'Washington')

plt.xlabel('Days since 1st observation')

plt.ylabel('Count')

plt.title('Growth Curve for New York, Washington, and California')

plt.legend()
visualize_state('California')
visualize_state('New York')
visualize_state('Washington')
measures = pd.read_csv('../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

measures
country_measures = pd.DataFrame(measures.groupby('Country')['Keywords'].agg(lambda x: list(x.values)))

country_measures = country_measures.reset_index()

country_measures
top_ten_country_measures = country_measures[(country_measures['Country'].isin(list(top_ten_countries.index))) | 

                (country_measures['Country'] == 'US:New York') | (country_measures['Country'] == 'US:California')]

top_ten_country_measures
top_ten_country_measures['Keywords'].values[9]
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from plotly.graph_objs import *

init_notebook_mode()

import plotly.express as px

init_notebook_mode(connected=True)



# This function creates a dataframe containing the number of confirmed cases along with the measures implemented for each date

def create_state_summary(state):

    # Isolate measures for state of choice, and extract date and keyword data

    state_measures = measures[measures['Country']=='US:{}'.format(state)][['Country', 'Date Start', 'Keywords']]

    

    state_measures['Date Start'] = pd.to_datetime(state_measures['Date Start']) # Will allow us to easily sort by the date

    state_measures = state_measures.sort_values('Date Start')

    

    # Reformat dataframe to aggregate all measures taken on the same date together into a list

    # Only one row for each date nowf

    state_measures = pd.DataFrame(state_measures.groupby('Date Start')['Keywords'].agg(lambda x: list(x.values)))

    

    # Get dataframe of cumulative confirmed cases for state

    

    state_confirmed = get_state_df(state)[['Last Update', 'Confirmed']]

    state_confirmed['Last Update'] = pd.to_datetime(state_confirmed['Last Update']) # Need to convert to matching type so that we can merge

    

    # Perform a right merge on the two dataframes

    # This enables us to keep the dates which do not have a corresponding measure implemented

    state_summary = state_measures.merge(state_confirmed, how='right', left_index=True, right_on='Last Update')

    state_summary = state_summary.fillna('N/A') # Ensures plot renders properly, just showing 'N/A' if a date has no new measure implemented

    return state_summary



# This function takes in the summary created above and created the desired plot.

def create_state_interactive_plot(state_name, state_summary):

    fig = px.line(state_summary, x="Last Update", y="Confirmed", title='{} Cases over Time'.format(state_name),

             hover_name="Last Update", hover_data=["Keywords"])

    fig.show()

    return



ny_summary = create_state_summary('New York')

create_state_interactive_plot('New York', ny_summary)
cali_summary = create_state_summary('California')

create_state_interactive_plot('California', cali_summary)
ny_summary = create_state_summary('New York')

cali_summary = create_state_summary('California')



ny_summary['State'] = np.array(['New York'] * ny_summary.shape[0])

cali_summary['State'] = np.array(['California'] * cali_summary.shape[0])



combined_summary = pd.concat([ny_summary, cali_summary], axis=0)

combined_summary



fig = px.line(combined_summary, x="Last Update", y="Confirmed", title='New York vs. California',

              hover_name="Last Update", hover_data=["Keywords"], color='State')



fig.show()
tests_done = united_states.head(55) # We only want the most recent available data

tests_done = tests_done[(tests_done['state'] == 'CA') | (tests_done['state'] == 'NY')][['state','totalTestResults']]

# Below, we add in the populations of each state 

# Source: Google, which was sourced from the United States Census Bureau

tests_done['total'] = np.array([39510000, 19450000])

# Finally, we divide to get the testing rate

tests_done['testingRate'] = tests_done['totalTestResults'] / tests_done['total']

tests_done
cases['Last Update'] = pd.to_datetime(cases['Last Update'])

# Filter dataframe to include a single country

def get_country_df(country):

    country_df = cases[cases['Country/Region'] == country]

    country_df.loc[:, 'Last Update'] = country_df['Last Update'].dt.date

    country_df = country_df.sort_values('Last Update', ascending = False)

    country_df.groupby('Last Update')[['Confirmed', 'Deaths', 'Recovered']].first() # Takes the latest reading from each day

    country_df.reset_index(inplace=True)

    country_df = country_df.sort_values('Last Update', ascending = True) # Puts data back in chronological order

    # The data has March 22nd mislabeled as March 8th. It leads to a misrepresentative visualization, so we drop it

    country_df = country_df[country_df['Last Update'] != datetime.strptime('2020-03-08', '%Y-%m-%d').date()]

    return country_df



# This function creates a dataframe containing the number of confirmed cases along with the measures implemented for each date

def create_country_summary(country):

    # Isolate measures for country of choice, and extract date and keyword data

    country_measures = measures[measures['Country']==country][['Country', 'Date Start', 'Keywords']]

    

    country_measures['Date Start'] = pd.to_datetime(country_measures['Date Start']) # Will allow us to easily sort by the date

    country_measures = country_measures.sort_values('Date Start')

    

    # Reformat dataframe to aggregate all measures taken on the same date together into a list

    # Only one row for each date now

    country_measures = pd.DataFrame(country_measures.groupby('Date Start')['Keywords'].agg(lambda x: list(x.values)))

    

    # Get dataframe of cumulative confirmed cases for country

    

    country_confirmed = get_country_df(country).loc[:, ['Last Update', 'Confirmed']].drop_duplicates(keep='first')

    country_confirmed['Last Update'] = pd.to_datetime(country_confirmed['Last Update']) # Need to convert to matching type so that we can merge

    

    # Perform a right merge on the two dataframes

    # This enables us to keep the dates which do not have a corresponding measure implemented

    country_summary = country_measures.merge(country_confirmed, how='right', left_index=True, right_on='Last Update')

    country_summary = country_summary.fillna('N/A') # Ensures plot renders properly, just showing 'N/A' if a date has no new measure implemented

    return country_summary



# This function takes in the summary created above and created the desired plot.

def create_country_interactive_plot(country_name, country_summary):

    fig = px.line(country_summary, x="Last Update", y="Confirmed", title='{} Cases over Time'.format(country_name),

             hover_name="Last Update", hover_data=["Keywords"])

    fig.show()

    return



temp = create_country_summary('Italy')

create_country_interactive_plot('Italy', temp)
temp = create_country_summary('South Korea')

create_country_interactive_plot('South Korea', temp)
italy_summary = create_country_summary('Italy')

skorea_summary = create_country_summary('South Korea')



italy_summary['Country'] = np.array(['Italy'] * italy_summary.shape[0])

skorea_summary['Country'] = np.array(['South Korea'] * skorea_summary.shape[0])



combined_summary = pd.concat([italy_summary, skorea_summary], axis=0)

combined_summary



fig = px.line(combined_summary, x="Last Update", y="Confirmed", title='Italy vs. South Korea',

              hover_name="Last Update", hover_data=["Keywords"], color='Country')





fig.show()