# import the necessary libraries

import numpy as np 

import pandas as pd 

import os

from xml.etree import ElementTree



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

!pip install chart_studio

import chart_studio.plotly as py

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

#py.init_notebook_mode(connected=True)



#Geographical Plotting

import folium

from folium import Choropleth, Circle, Marker

from folium import plugins

from folium.plugins import HeatMap, MarkerCluster



#Racing Bar Chart

!pip install bar_chart_race

import bar_chart_race as bcr

from IPython.display import HTML



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")# for pretty graphs



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
list_of_files=os.listdir('../input/covid19-clinical-trials-dataset/COVID-19 CLinical trials studies')

print('Total Researches going on: ',len(list_of_files))
path = '../input/covid19-clinical-trials-dataset/COVID-19 CLinical trials studies/'



# Read in data



df_covid = pd.DataFrame()

df = pd.DataFrame()

i=0

list_keywords=[]



for file in list_of_files:

    file_path=path+file

    #print('Processing....'+file_path)

    tree = ElementTree.parse(file_path)

    root = tree.getroot()



    trial = {}



    trial['id'] = root.find('id_info').find('nct_id').text

    trial['overall_status'] = root.find('overall_status').text

    trial['study_type'] = root.find('study_type').text

    

    if root.find('start_date') != None:

        trial['start_date'] = root.find('start_date').text

    else:

         trial['start_date'] = ''

        

    if root.find('enrollment') != None:

        trial['enrollment'] = root.find('enrollment').text

    else:

         trial['enrollment'] = ''



    trial['condition'] = root.find('condition').text.upper().replace('CORONAVIRUS INFECTIONS','CORONAVIRUS INFECTION').replace('CORONA VIRUS INFECTION','CORONAVIRUS INFECTION').replace('SARS-COV-2','SARS-COV2').replace('SARS-COV 2','SARS-COV2').replace('COVID-19','COVID').replace('COVID19','COVID').replace('COVID 19','COVID')

    if root.find('location_countries') != None:

        trial['location_countries'] = root.find('location_countries').find('country').text.upper()

    else:

        trial['location_countries'] = ''

        

    if root.find('intervention') != None:

        trial['intervention'] = root.find('intervention').find('intervention_name').text.upper()

    else:

        trial['intervention'] = ''

        

    #trial['description'] = root.find('brief_summary')[0].text

    for entry in root.findall('keyword'):

        list_keywords.append(entry.text)



    if root.find('official_title') == None:

        trial['title'] = root.find('brief_title').text

    else:

        trial['title'] = root.find('official_title').text



    date_string = root.find('required_header').find('download_date').text

    trial['date_processed'] = date_string.replace('ClinicalTrials.gov processed this data on ', '')

    

    trial['sponsors'] = root.find('sponsors').find('lead_sponsor').find('agency').text

    

    

    df  = pd.DataFrame(trial,index=[i])

    i=i+1

    

    df_covid = pd.concat([df_covid, df])

    
df_covid.shape
df_covid.head()




df_covid['overall_status'].value_counts().iplot(kind='bar',yTitle='Count',color='red')
labels = df_covid['study_type'].value_counts().index

values = df_covid['study_type'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial')])

fig.show()
interventional_studies = df_covid[df_covid['study_type']=='Interventional']

top_interventions = interventional_studies['intervention'].value_counts().sort_values(ascending=True)[-5:]

top_interventions.iplot(kind='barh', title='Interventions',color='green')
# Top 10 Countries

countries = interventional_studies[interventional_studies['location_countries']!='']

country = countries['location_countries'].value_counts().sort_values(ascending=True)[-15:]

country.iplot(kind='barh', title='Country')
lead_sponsors = interventional_studies['sponsors'].value_counts().sort_values(ascending=False)[:10]

lead_sponsors.iplot(kind='bar', title='Lead Sponsors')
# Convert to numeric

interventional_studies['enrollment'] = interventional_studies['enrollment'].astype(int)

# Remove the trials with recruitment status withdrawn and terminated

enrollment = interventional_studies.loc[

 (interventional_studies['overall_status'] != 'Withdrawn') & (interventional_studies['overall_status'] != 'Terminated')]

bins = [-1, 20, 40, 60, 100, 200, 400, 600, 1000]

group_names = ['< 20', '21-40', '41-60', '61-100', '101-200', '201-400', '401-600', '>600']

categories = pd.cut(enrollment['enrollment'], bins, labels=group_names)

# Add categories as column in dataframe

enrollment['Category'] = categories

# View value counts

enrollment_counts = enrollment['Category'].value_counts().sort_index(ascending=True)

enrollment_counts
enrollment_counts.iplot(kind='bar', title='Size of Interventional Trials')
trials = pd.read_csv('../input/covid19-clinical-trials-dataset/COVID clinical trials.csv')

trials.head()
labels = trials['Phases'].value_counts().drop('Not Applicable').index

values = trials['Phases'].value_counts().drop('Not Applicable').values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label',

                             insidetextorientation='radial')])

fig.show()
# text preprocessing helper functions



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text
# Applying the cleaning function to the dataset

phase_4_trials = trials[trials['Phases']=='Phase 4']['Title'].apply(str).apply(lambda x: text_preprocessing(x))
from wordcloud import WordCloud

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(phase_4_trials))



plt.figure(figsize=(10, 16))

plt.title('Phase 4 Trials',fontsize=30);

plt.imshow(wordcloud1, interpolation="gaussian")

plt.axis("off")

plt.show()