from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import plotly.graph_objects as go

import plotly.express as px

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = None #1000 # specify 'None' if want to read whole file

usa_jobs = pd.read_csv('/kaggle/input/data_scientist_united_states_job_postings_jobspikr.csv', delimiter=',', nrows = nRowsRead)

usa_jobs.dataframeName = 'data_scientist_united_states_job_postings_jobspikr.csv'

nRow, nCol = usa_jobs.shape

print(f'There are {nRow} rows and {nCol} columns')
usa_jobs.head(5)
usa_jobs['crawl_timestamp'] = pd.to_datetime(usa_jobs['crawl_timestamp'])

#Finding earliest and latest posting

print(f"Earliest job post in the set: {min(usa_jobs['crawl_timestamp'])}")

print(f"Latest post in the set: {max(usa_jobs['crawl_timestamp'])}")
requirements = {"powerbi":0, " r ":0, "tableau":0, "qlikview":0, "python":0, "sql":0, "machine learning":0,'linux':0, 'c#':0, \

" ml ":0, "hive":0, "spark":0, "hadoop":0, "java":0, "scala":0, "kafka":0, "bachelor":0, "master":0, "phd":0, 'year':0, 'years':0, "c++":0}

for i in range(len(usa_jobs)):

    job_description = usa_jobs.job_description[i].lower().replace("\n", " ")

    for k in requirements:

        if k in job_description:

            requirements[k] += 1

#print(requirements['machine learning'])

#print(requirements[' ml '])

requirements['machine learning'] += requirements[' ml ']

requirements['year'] += requirements['years']

requirements['years experience'] = requirements.pop('year')

del requirements['years']

del requirements[' ml ']
from collections import OrderedDict

sorted_req = OrderedDict(sorted(requirements.items(), key=lambda x:x[1]))

plt.figure(figsize=(10, 10))

plt.bar(range(len(sorted_req)), list(sorted_req.values()), align='center')

plt.xticks(range(len(sorted_req)), list(sorted_req.keys()), rotation='vertical')

plt.xlabel("job requirement")

plt.ylabel("Number of posts")

plt.show()
#Optional cell, if interested in 

# from collections import Counter

# print(Counter(usa_jobs.inferred_state))

# print(Counter(usa_jobs.salary_offered))

# print(Counter(usa_jobs.job_board).most_common())
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

description_example = usa_jobs.job_description[2].lower()

word_tokens = word_tokenize(description_example)

filtered_description = [w for w in word_tokens if not w in stop_words]

filtered_description = " ".join(filtered_description)
from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from matplotlib.pyplot import plot

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

#sorted_req
aggregate_descriptions = " ".join(job_description.lower() 

                      for job_description in usa_jobs.job_description)

stopwords = set(STOPWORDS)
stopwords.update(['experience', 'following', 'candidates', 'big', 'background','developing', 'characteristics', 'data', 'team', 'data', 'scientist', 'strong', 'project', 

                  'solution', 'technology', 'science', 'model', 'knowledge','skill', 'work', 'build', 'will', 'knowledge', 'application','gender', 'identity', 'equal',

                  'opportunity','related','field', 'without', 'regard', 'national', 'origin', 'religion', 'sex', 'race', 'color', 'veteran', 'status','sexual',

                  'orientation','opportunity', 'employer', 'qualified','applicant','skills', 'job', 'summary', 'advanced', 'system', 'applicants', 'receive', 'large', 'best', 'practice', 'problem'

                 , 'processing', 'affirmative', 'action', 'employment', 'consideration', 'receive', 'united', 'state', 'programming', 'computer', 'working', 'saying', 

                  'preferred', 'qualification', 'disability', 'protected', 'structured', 'unstructured', 'problems', 'technical', 'internal', 'external', 'non',

                 'subject', 'matter', 'please', 'apply', 'using', 'dental', 'reasonable', 'accomodation', 'join', 'us', 'tools', 'individuals', 'disabilities'

                 , 'type', 'full', 'wide', 'range', 'duties', 'responsibilities', 'stakeholder', 'oral', 'written', 'ideal', 'candidate', 'ability', 'qualifications', 'well',

                  'must', 'able', 'unit', 'member', 'posted', 'today', 'service', 'clearance', 'days', 'ago', 'high', 'quality', 'level', 'every', 'use', 'case', 'additional'])

wordcloud = WordCloud(stopwords=stopwords, background_color='white',

                     width=1000, height=700).generate(aggregate_descriptions)

plt.figure(figsize=(15, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
states = usa_jobs.inferred_state.unique()

sum_in_states = []

for state in states:

    total_jobs_state = len(usa_jobs[usa_jobs['inferred_state']==state])

    sum_in_states.append(int(total_jobs_state))

jobs_in_states = {'state':states, 'Total jobs':sum_in_states}

jobs_in_states = pd.DataFrame(jobs_in_states)

jobs_in_states = jobs_in_states.sort_values(by='Total jobs', ascending=False)

jobs_in_states = jobs_in_states.reset_index(drop=True)

jobs_in_states = jobs_in_states.drop(jobs_in_states.index[len(jobs_in_states)-1])

jobs_in_states[:10]
latitude = [32.318231,35.20105,34.048928,36.778261,39.550051,41.603221,

38.905985,38.910832,27.664827,32.157435,19.898682,41.878003,44.068202,

40.633125,40.551217,39.011902,37.839333,31.244823,42.407211,39.045755,

45.253783,44.314844,46.729553,37.964253,32.354668,46.879682,35.759573,

47.551493,41.492537,43.193852,40.058324,34.97273,38.80261,43.299428,

40.417287,35.007752,43.804133,41.203322,41.580095,33.836081,43.969515,

35.517491,31.968599,39.32098,37.431573,44.558803,47.751074,43.78444,

38.597626,43.075968, 38.895]

longitude = [-86.902298,-91.831833,-111.093731,-119.417932,-105.782067,

-73.087749,-77.033418,-75.52767,-81.515754,-82.907123,-155.665857,-93.097702,

-114.742041,-89.398528,-85.602364,-98.484246,-84.270018,-92.145024,-71.382437,

-76.641271,-69.445469,-85.602364,-94.6859,-91.831833,-89.398528,-110.362566,

-79.0193,-101.002012,-99.901813,-71.572395,-74.405661,-105.032363,-116.419389,

-74.217933,-82.907123,-97.092877,-120.554201,-77.194525,-71.477429,-81.163725,

-99.901813,-86.580447,-99.901813,-111.093731,-78.656894,-72.577841,-120.740139,

-88.787868,-80.454903,-107.290284, -77.0366]

state_names = ['Alabama','Arkansas','Arizona','California','Colorado','Connecticut',

'District of columbia','Delaware','Florida','Georgia','Hawaii','Iowa',

'Idaho','Illinois','Indiana','Kansas','Kentucky','Louisiana','Massachusetts',

'Maryland','Maine','Michigan','Minnesota','Missouri','Mississippi',

'Montana','North carolina','North dakota','Nebraska','New hampshire',

'New jersey','New mexico','Nevada','New york','Ohio','Oklahoma','Oregon',

'Pennsylvania','Rhode island','South carolina','South dakota','Tennessee',

'Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West virginia',

'Wyoming', 'Washington d.c.']

state_dict = {'state':state_names, 'latitude':latitude, 'longitude':longitude}

state_df = pd.DataFrame(state_dict, columns=['state', 'latitude', 'longitude'])

state_coords = pd.merge(state_df, jobs_in_states, how='right', on='state')

state_coords = state_coords.sort_values(by='Total jobs', ascending=False)

state_coords = state_coords.reset_index(drop=True)
fig = px.scatter_geo(data_frame=state_coords, lat='latitude', scope='north america', hover_name='state',

                    lon='longitude', size='Total jobs', projection='hammer')

fig.show()
month_of_posting = []

for i in range(len(usa_jobs)):

    month_of_posting.append(usa_jobs['crawl_timestamp'][i].month)

usa_jobs['month'] = month_of_posting

months = [x for x in range(2, 11)]

sum_in_months = []

for month in months:

    total_jobs_in_month = len(usa_jobs[usa_jobs['month']==month])

    sum_in_months.append(total_jobs_in_month)

jobs_in_months = {'month':months, 'Total jobs':sum_in_months}

jobs_in_months = pd.DataFrame(jobs_in_months)

#dropping the last month of october, because it is not fully included in this set

jobs_in_months = jobs_in_months.drop([8])

jobs_in_months
months_plot = go.Figure()

months_plot.add_trace(go.Scatter(x=jobs_in_months.month, 

                                y=jobs_in_months['Total jobs']))

months_plot.update_layout(title='US job posts in Data science by month in 2019',

                         xaxis_title='Month', yaxis_title='Amount of job posts')

months_plot.show()
import re

from word2number import w2n

import statistics

def search_text_left_of_word(text, word, n):

    """Searches for a text and retrieves n words on left side of the text"""

    words = re.findall(r'\w+', text)

    try:

        index = words.index(word)

    except ValueError:

        return " "

    return words[index - n:index]

def search_year_word(text):

    return text.find('year')

def search_number_around_word(word_surroundings):          #this function adds all the numbers found to a list. It also converts the words to numbers, if it is the case

    word_surroundings = " ".join(word_surroundings)

    word_surroundings = word_tokenize(word_surroundings)

    pos_tags = nltk.pos_tag(word_surroundings)

    numbers_list = []

    for a in pos_tags:

        if a[1] in 'CD':

            if a[0].isalpha():       #sometimes the numbers are written as words, e.g. 'Three' instead of 3

                try:

                    numbers_list.append(w2n.word_to_num(a[0]))

                except ValueError:

                    return ""

            else:

                numbers_list.append(a[0])

    return numbers_list

years_experience_req = []



def convert_to_int(list_elem):

    try:

        converted_int = int(list_elem)

        if converted_int <= 10:

            return int(list_elem)

    except ValueError:

        return

for post_index in range (len(usa_jobs)):

    current_job = usa_jobs.job_description[post_index]

    word_surroundings = search_text_left_of_word(current_job, 'years', 2)

    if current_job.find(' year ') > -1:

        years_experience_req.append(['1'])

    years_experience_req.append(search_number_around_word(word_surroundings))

    #print(post_index, search_number_around_word(word_surroundings))

years_experience_req = [convert_to_int(item) for sublist in years_experience_req for item in sublist]

years_experience_req = [i for i in years_experience_req if i != None]

#print(years_experience_req)

print("An average of ", statistics.mean(years_experience_req), "  years is required in most job offerings. ")
#Just checking if the last 7's in the data actually correspond to 

usa_jobs.job_description[10]
statistics.mean([3, 4, None])