# Python 3 environment defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import matplotlib.dates as mdates

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
filepath = '../input/university-student-carbon-footprint-knowledge/student-responses-without-ids.csv'

data = pd.read_csv(filepath, index_col='Timestamp')
data.head()
# Only responses which agreed were collected, so this column is not needed

data.drop(columns=['Data Protection Agreement'], inplace=True)



# Rename columns from original questions to names that are easier to work with

data.rename(columns={

    'Age group:':'age_group',

    'Education status:':'education',

    'Work status:':'working',

    'How do you travel to work/university?':'travel_transport',

    'On average, how long does it take you to travel to work/university?':'average_travel_duration',

    'How engaged do you feel with the issue of climate change?':'climate_change_engagement',

    'Do you know what a carbon footprint is?':'carbon_footprint_definition_knowledge',

    'How aware are you of your own carbon footprint?':'carbon_footprint_awareness',

    'How do you currently measure your own carbon footprint?':'carbon_footprint_rating',

    'What sources of information do you use, if any, to measure your carbon footprint?':'carbon_footprint_sources',

    'Do you feel having more information about your carbon footprint would be useful?':'carbon_footprint_info_useful',

    'Would having more information about your carbon footprint be likely to change your habits?':'carbon_footprint_info_change_habits',

    'What kind of motivation, if any, would you need in order to change one of your habits for a more environmentally friendly one?':'habit_change_motivation',

    'How have you tried to reduce your carbon footprint, if at all?':'carbon_footprint_reduction_methods',

    'How much do you care about trying to reduce your carbon footprint?':'carbon_footprint_reduction_desire',

    'What do you consider to be the hardest part about trying to reduce your carbon footprint?':'carbon_footprint_reduction_difficulties',

    'Have you tried to encourage others to reduce their carbon footprint?':'encouraged_others_to_reduce_carbon'

}, inplace=True)



data.carbon_footprint_reduction_difficulties = data.carbon_footprint_reduction_difficulties.str.lower()

data.habit_change_motivation = data.habit_change_motivation.str.lower()



data.head()
print("Total Row Count: {0} \nTotal Column Count: {1}".format(data.shape[0], data.shape[1]))
data.head()
data.info()
data.isnull().sum()
data.dropna(axis=0, inplace=True);
_,ax = plt.subplots(figsize=(8,6))

data.age_group.value_counts().plot(kind='pie', ax=ax, autopct='%.0f%%')

ax.set_title('Age Group', weight='bold')

ax.set_ylabel(None);
data.education.value_counts()
_,ax = plt.subplots(figsize=(8,6))

data.working.value_counts().plot(kind='pie', autopct='%.0f%%', startangle=90, ax=ax)

ax.set_title('Work Status', weight='bold')

ax.set_ylabel(None); 
mode_of_transport = data.travel_transport.str.get_dummies(';').astype('bool').sum()



_,ax = plt.subplots(figsize=(8,4))

mode_of_transport.plot(ax=ax, kind="bar")



ax.set_title('\'How do you travel to work/university?\'', weight='bold')

ax.set_ylabel('Count')

ax.set_xlabel('Mode of Transport');
_,ax = plt.subplots(figsize=(8,6))

data.average_travel_duration.value_counts().sort_values().plot(kind='pie', autopct='%.0f%%', ax=ax)

ax.set_title('\'On average, how long does it take you to travel to work/university?\'', weight='bold')

ax.set_ylabel(None);
# Provide a general indication of the carbon impact of participants' travel.

# Assumptions:

# - Car, Motorbike, Bus, Train, all assumed to be polluting & petrol/diesel rather than electric, hydrogen, or hybrid.

# - Bus/Train less polluting than Car/Motorbike as suggested by UK Department for Business, Energy & Industrial Strategy:

# https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2019

conditions = [

    # High impact

    (data.average_travel_duration == 'Less than an hour') & (data.travel_transport.str.contains('Car / Motorbike')),

    # Moderate impact

    (data.average_travel_duration == 'Less than an hour') & (data.travel_transport.str.contains('Bus / Train')),

    # Moderate impact

    (data.average_travel_duration == 'Less than 15 minutes') & (data.travel_transport.str.contains('Car / Motorbike')),

    # Low impact

    (data.average_travel_duration == 'Less than 15 minutes') & (data.travel_transport.str.contains('Bus / Train')),

    # Zero impact

    (data.travel_transport.str.contains('Walk / Cycle')) | (data.travel_transport.str.contains('I don\'t travel'))]

choices = ['High', 'Moderate', 'Moderate', 'Low', 'Zero']

data['estimated_carbon_impact'] = np.select(conditions, choices, default='Zero')



_,ax = plt.subplots(figsize=(8,6))

data['estimated_carbon_impact'].value_counts().sort_values().plot(kind='pie', autopct='%.0f%%', ax=ax)

ax.set_title('Estimated Travel Carbon Impact', weight='bold')

ax.set_ylabel(None);

climate_change_engagement = data.climate_change_engagement.value_counts().sort_index()



_,ax = plt.subplots(figsize=(8,4))

climate_change_engagement.plot(ax=ax, kind="bar")



ax.set_title('\'How engaged do you feel with the issue of climate change?\'', weight='bold')

ax.set_ylabel('Count')

ax.set_xlabel('Engagement (\'Not Very Engaged\' – \'Very Engaged\')');
_,ax = plt.subplots(figsize=(8,6))

data.carbon_footprint_definition_knowledge.value_counts().plot(kind='pie', autopct='%.0f%%', startangle=90, ax=ax)

ax.set_title('\'Do you know what a carbon footprint is?\'', weight='bold')

ax.set_ylabel(None);
_,ax = plt.subplots(figsize=(8,6))

data.carbon_footprint_awareness.value_counts().plot(kind='bar', ax=ax)

ax.set_title('\'How aware are you of your own carbon footprint?\'', weight='bold');

ax.set_xlabel('Awareness (\'Not very aware\'–\'Very aware\')');
_,ax = plt.subplots(figsize=(8,6))

data.carbon_footprint_rating.value_counts().sort_index().plot(kind='bar', ax=ax)

ax.set_title('\'How do you currently measure your own carbon footprint?\'', weight='bold');

ax.set_xlabel('Measurement (\'Very bad\'–\'Very good\')');
print('What sources of information do you use, if any, to measure your carbon footprint?')

data.carbon_footprint_sources.value_counts()
carbon_footprint_info_useful_value_counts = data.carbon_footprint_info_useful.value_counts().sort_index().reindex(range(1,6), fill_value=0)



_,ax = plt.subplots(figsize=(8,6))

carbon_footprint_info_useful_value_counts.plot(kind='bar', ax=ax, ylim=(0, 35))



ax.set_title('\'Do you feel having more information about your carbon footprint would be useful?\'', weight='bold');

ax.set_xlabel('Usefulness (\'No, not useful at all\'–\'Yes, very useful\')');
_,ax = plt.subplots(figsize=(8,6))

data.carbon_footprint_info_change_habits.value_counts().sort_index().plot(kind='pie', autopct='%.0f%%', startangle=90, ax=ax)

ax.set_title('\'Would having more information about your carbon footprint be likely to change your habits?\'', weight='bold')

ax.set_ylabel(None);
from wordcloud import WordCloud

import nltk as nlp



habit_change_motivation_responses = data.habit_change_motivation.values

habit_change_motivation_responses = map(lambda response: response, habit_change_motivation_responses)



response_filters = ['.', 'yes', 'idk']

habit_change_motivation_responses = list(filter(lambda x: x not in response_filters, habit_change_motivation_responses))



flatten = lambda l: [item for sublist in l for item in sublist]

words = flatten([response.split() for response in habit_change_motivation_responses])



porter_stemmer = nlp.PorterStemmer()

normalised_words = [porter_stemmer.stem(word) for word in words]



stopwords = nlp.corpus.stopwords.words('english') + ['an', 'as', 'a', 'i', 'it\s', 'is', 'something', 'at', 'although', 'actually', 'bit']



wordcloud = WordCloud(

    background_color='white',

    stopwords=stopwords,

    max_words=100,

    width=960,

    height=800,

    random_state=42).generate(' '.join(words))



fig = plt.figure(1, figsize=(10, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
knowledge_of_impact_responses = list(filter(lambda response: any(word in response.lower() for word in ['impact','consequences', 'climate', 'evidence', 'effects', 'enviromint', 'education']), habit_change_motivation_responses))



print("Responses to the question: 'What kind of motivation, if any, would you need in order to change one of your habits for a more environmentally friendly one?'")

print("{} responses ({:.0f}%) mentioned knowledge of impact on the environment as a motivator.".format(len(knowledge_of_impact_responses),len(knowledge_of_impact_responses)/len(data)*100))
reduction_methods = data.carbon_footprint_reduction_methods.str.get_dummies(sep=';').sum().sort_values()

reduction_methods = reduction_methods.loc[reduction_methods > 1]



_,ax = plt.subplots(figsize=(8,6))



reduction_methods.plot(kind='pie', autopct='%.0f%%', startangle=90, ax=ax)



ax.set_title('\'How have you tried to reduce your carbon footprint, if at all?\'', weight='bold')

ax.set_ylabel(None);
carbon_footprint_reduction_desire_counts = data.carbon_footprint_reduction_desire.value_counts().sort_index()



_,ax = plt.subplots(figsize=(8,6))

carbon_footprint_reduction_desire_counts.plot(kind='bar', ax=ax, ylim=(0, 35))



ax.set_title('\'How much do you care about trying to reduce your carbon footprint?\'', weight='bold');

ax.set_xlabel('Usefulness (\'I don\'t care at all\'–\'I care very much\')');
reduction_difficulties = data.carbon_footprint_reduction_difficulties.values



def count_responses_containing(words):

    return len(list(filter(lambda r: any(w in r for w in words), reduction_difficulties)))



effort_response_count = count_responses_containing(['effort'])

habit_response_count = count_responses_containing(['habit','lifestyle','day to day','day-to-day','daily'])

money_response_count = count_responses_containing(['cost','money'])

unsure_response_count =  count_responses_containing(['not know', 'dont know', 'don\'t know', 'no idea', 'not sure', 'knowing how'])

food_response_count = len(list(filter(lambda r: any(w in r.split() for w in ['food','eat','dairy']), reduction_difficulties)))

travel_response_count =  count_responses_containing(['travel','transport'])

tech_response_count =  count_responses_containing(['energy','power','technology','devices'])



all_response_count = len(reduction_difficulties)

other_response_count = all_response_count - sum([effort_response_count,habit_response_count,money_response_count,food_response_count,travel_response_count,tech_response_count,unsure_response_count])



sizes = [effort_response_count,habit_response_count,money_response_count,food_response_count,travel_response_count,tech_response_count,unsure_response_count,other_response_count]

labels = ['Effort required', 'Changing habits/lifestyle', 'Financial concerns', 'Food concerns', 'Travel concerns', 'Technology', 'Unsure', 'Other']



_,ax = plt.subplots(figsize=(8,6))

ax.pie(sizes, labels=labels, autopct='%0.0f%%', startangle=90)



ax.set_title('\'What do you consider to be the hardest part about trying to reduce your carbon footprint?\'', weight='bold')

ax.set_ylabel(None);
_,ax = plt.subplots(figsize=(8,6))

data.encouraged_others_to_reduce_carbon.value_counts().plot(kind='pie', startangle=90, autopct='%0.0f%%', ax=ax)

ax.set_title('\'Have you tried to encourage others to reduce their carbon footprint?\'', weight='bold')

ax.set_ylabel(None);
data.corr()
# Internal Seaborn error (with Numpy) trying to subtract, when pairplot is provided with any boolean columns

sns.pairplot(data, kind="reg", diag_kind="kde");
ax = sns.regplot(data=data, x='carbon_footprint_info_useful', y='carbon_footprint_reduction_desire')
ax = sns.regplot(data=data, x='climate_change_engagement', y='carbon_footprint_reduction_desire')