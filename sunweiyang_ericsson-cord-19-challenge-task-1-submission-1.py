#!pip install kaggle                   #Install the Kaggle API package (Not necessary since we are in Kaggle)

!pip install cord-19-tools             #Install the COVID-19 Data Tools package

#!pip install plotly                   #Exists in Kaggle already



#!pip install spacy                    #Exists in Kaggle already

!pip install spacy-langdetect



#!pip install pycountry                #Exists in Kaggle already

!pip install geonamescache

!pip install geopy

!pip install reverse_geocoder



#!pip install nltk                     #Exists in Kaggle already



!pip install ktrain

#!pip install --upgrade scikit-learn   #Exists in Kaggle already



#!pip install pyLDAvis                 #Exists in Kaggle already

#!pip install wordcloud                #Exists in Kaggle already



#!pip install torch                    #Exists in Kaggle already

#!pip install transformers             #Exists in Kaggle already

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
# Import the required libraries

import os

#import shutil



import pandas as pd

import numpy as np

import re



import cotools as co                   #COVID-19 Data Tools

#from pprint import pprint

import pycountry

import geonamescache

import reverse_geocoder as rg

#import sys



import plotly as py

import plotly.graph_objs as go

import plotly.express as px

from IPython.display import display, HTML

from ipywidgets import interact, interactive, fixed, interact_manual, widgets

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

#import seaborn as sns

import pyLDAvis

import pyLDAvis.sklearn



%matplotlib inline



#please make sure that sklearn version is 0.22.2.post1

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer 

from sklearn.manifold import TSNE



import spacy

from spacy_langdetect import LanguageDetector

import en_core_sci_lg



from transformers import pipeline



import gc

import nltk

#from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

#nltk.download('wordnet')

#nltk.download('averaged_perceptron_tagger')

#nltk.download('punkt')
pd.options.mode.chained_assignment = None
def log_progress(sequence, every=None, size=None, name='Items'):

    

    '''

    Use the ipwidgets package to display a progress bar to give the user an indication

    of the progress of code execution for certain portions of the notebook.

    This function is intended to be used in the definition of a for loop to indicate

    how far execution has gotten through the object being iterated over in the for loop.



    Inputs: sequence - contains the for loop iteration (e.g., list or iterator)

            every (integer) - number of steps to display



    Outputs:  displays the progress bar in the notebook

              yield record  - returns the current iteration object back to the calling for loop

    

    '''

    

    from ipywidgets import IntProgress, HTML, VBox

    from IPython.display import display



    # Determine the parameters for the progress bar based on the function inputs

    is_iterator = False

    if size is None:

        try:

            size = len(sequence)

        except TypeError:

            is_iterator = True

    if size is not None:

        if every is None:

            if size <= 200:

                every = 1

            else:

                every = int(size / 200)     # every 0.5%

    else:

        assert every is not None, 'sequence is iterator, set every'

    

    # Instantiate and display the progress bar        

    if is_iterator:

        progress = IntProgress(min=0, max=1, value=1)

        progress.bar_style = 'info'

    else:

        progress = IntProgress(min=0, max=size, value=0)

    label = HTML()

    box = VBox(children=[label, progress])

    display(box)

    

    # Update the progress bar state at each iteration of the for loop using this function

    index = 0

    try:

        for index, record in enumerate(sequence, 1):

            if index == 1 or index % every == 0:

                if is_iterator:

                    label.value = '{name}: {index} / ?'.format(

                        name=name,

                        index=index

                    )

                else:

                    progress.value = index

                    label.value = u'{name}: {index} / {size}'.format(

                        name=name,

                        index=index,

                        size=size

                    )

                    

            # return the current iteration object, preserving the state of the function

            yield record

    except:

        progress.bar_style = 'danger'

        raise

    else:

        progress.bar_style = 'success'

        progress.value = index

        label.value = "{name}: {index}".format(

            name=name,

            index=str(index or '?')

        )

        

        



def process_text(text):

    

    '''

    Use the spaCy library natural language processing capabilities to clean an input text, 

    in string format, for punctuation, stop words, and lemmatization.



    Input:  text - a string to clean and lemmatize



    Output  a modified version of the input string that has been cleaned by removing 

            punctuation, stop words, and pronouns, and has had the remaining words 

            converted into corresponding lemmas

            

    '''

    

    # Create a spaCy "Doc" object from the input text string.

    doc = nlp(text.lower())

    

    result = [] # list that will contain the lemmas for each word in the input string

    

    for token in doc:

        

        if token.text in nlp.Defaults.stop_words:   #screen out stop words

            continue

        if token.is_punct:                          #screen out punctuations

            continue

        if token.lemma_ == '-PRON-':                #screen out pronouns

            continue

        

        result.append(token.lemma_)

    

    # Return the lemmatized version of the cleaned input text string

    return " ".join(result)
# Set the path where the raw data is

data_dir = '/kaggle/input/CORD-19-research-challenge'



# Set the current working directory path to where the raw data is

os.chdir(data_dir)



# Set the path where the formatted data will be stored

output_dir = '/kaggle/working/'



# Read in the metadata.csv file as a pandas DataFrame

metadata_information = pd.read_csv('metadata.csv')
metadata_information.shape
def is_date(string, fuzzy=False):

    

    '''

    Checks if string input can be interpreted as a date

    

    Input:  string - string to check whether it is a valid date

 

    Output:  boolean - True if string is a valid date; False otherwise 

 

    '''

    

    from dateutil.parser import parse

    

    try: 

        parse(string, fuzzy=fuzzy)

        return True



    except ValueError:

        return False
#print('Please input in the earliest date to filter the research paper (yyyy-mm-dd)!')

#filter_date = str(input())

#print('Would you like to only consider research papers published in journals (Y/N)?')

#from_journal_sources_only = str(input())



# Modify this date per user requirements, or enter a non-valid date string to disable publication date filtering

filter_date = '2019-12-01'

from_journal_sources_only = 'No'



# paper_id_list is a list of the IDs for all papers published after the specified date

# (or all papers if the date filtering is disabled).



if is_date(filter_date) == True and from_journal_sources_only == 'Yes':

    paper_id_list = metadata_information[(metadata_information['publish_time'] >= filter_date) & (metadata_information['journal'].notnull())].dropna(subset=['sha'])['sha'].tolist()



elif is_date(filter_date) == True and from_journal_sources_only == 'No':

    paper_id_list = metadata_information[metadata_information['publish_time'] >= filter_date].dropna(subset=['sha'])['sha'].tolist()



elif is_date(filter_date) == False and from_journal_sources_only == 'Yes':

    paper_id_list = metadata_information[metadata_information['journal'].notnull()].dropna(subset=['sha'])['sha'].tolist()

    

else:

    paper_id_list = metadata_information['sha'].tolist()
def create_library(list_of_folders, list_of_papers = paper_id_list):



    '''

    Read JSON files for each paper from a list of subdirectories, and convert the JSON data for each paper

    into a Python dictionary.



    Inputs:  list_of_folders (list) - a list of subfolder names (strings) where the raw paper data is found

             list_of_papers (list) - list of IDs for the target paper from which we want to extract the

                                     relevant data for further analysis

    Output:  internal_library (list) - list of dictionaries, where each dictionary contains the detailed

                                       data for one paper

    

    '''

    

    import json

    

    internal_library = []     # list of dictionaries for the papers; each dictionary describes one paper 



    # Iterate through each subfolder with the raw paper data; use the log_progress()

    # function to give the user an indication of execution progress. 

    for i in log_progress(list_of_folders, every = 1):



        # Check two different sub-subfolders under each subfolder in the list of folders to find the

        # papers - ".../pdf_json" and ".../pmc_json".

        # For each paper, convert the JSON file into a Python dictionary (data{}).

        # Add this dictionary to the list of dictionaries for all papers (internal_library[])

        try:



            pdf_file_path = data_dir + '/' + i + '/' + i + '/pdf_json'

            pdf_file_list = [i for i in os.listdir(pdf_file_path) if i.split('.')[0] in list_of_papers]

            print('There are {a} papers in the {c} group after {b}.'.format(a = len(pdf_file_list), b = filter_date, c = str(i + str('_pdf'))))



            for each_file in pdf_file_list:

                file_path = data_dir + '/' + i + '/' + i + '/pdf_json/' + each_file



                with open(file_path) as f:

                    data = json.load(f)



                internal_library.append(data)



        except:

            continue



        try:



            pmc_file_path = data_dir + '/' + i + '/' + i + '/pmc_json'

            pmc_file_list = [i for i in os.listdir(pmc_file_path) if i.split('.')[0] in list_of_papers]

            print('There are {a} papers in the {c} group after {b}.'.format(a = len(pmc_file_list), b = filter_date, c = str(i + str('_pmc'))))



            for each_file in pmc_file_list:

                file_path = data_dir + '/' + i + '/' + i + '/pmc_json/' + each_file



                with open(file_path) as f:

                    data = json.load(f)



                internal_library.append(data)



        except:

            continue

            

    return internal_library







def data_creation(list_of_folders, metadata, date = filter_date, list_of_papers = paper_id_list):

    

    '''

    Extract the key data for each paper and create a combined DataFrame with this information

   

    Inputs: list_of_folders (list) - a list of subfolder names (strings) where the raw paper data is found

            metadata (DataFrame) - metadata information on all papers

            date (string) - publication date; only include papers published after this date

            list_of_papers (list) - list of IDs for the target papers from which we want to extract the

                                relevant data for further analysis

    Outputs: test_df (DataFrame) - DataFrame containing the key data for each paper  

    

    '''

    

    # Use the create_library() function above to read the raw JSON data and create a list 

    # of dictionaries (internal_library). Each dictionary describes each paper in the raw data.

    # Note that we are using the date-filtered list of paper IDs (paper_id_list) to target

    # only those papers we want based on publication date (if the filtering is enabled).

    internal_library = create_library(list_of_folders = selected_folders, list_of_papers = paper_id_list)



    title_list = []          # list of paper titles (list of strings)

    abstract_list = []       # list of paper abstracts (list of strings)

    text_list = []           # list of paper full texts (list of strings)



    # Extract title, abstract text, and body text for each paper

    for i in list(range(0, len(internal_library))):



        # Get the title from the "metadata" dictionary for each paper. The "metatdata" dictionary

        # for each paper is a dictionary that contains title and author information.        

        title_list.append(internal_library[i].get('metadata').get('title'))



        # Use cord-19-tools package functions co.abstract() and co.text() to extract

        # abstract text and body text from the paper dictionary        

        try:

            abstract_list.append(co.abstract(internal_library[i]))

        except:

            abstract_list.append('No Abstract')



        text_list.append(co.text(internal_library[i]))



        

    # Extract paper ID information

    paper_id = [i.get('paper_id') for i in internal_library]   # list of the ID for each paper



    # Extract the location and country that published the research paper

    primary_location_list = []      # list of the primary locations for the authors of each paper

    primary_country_list = []       # list of the primary countries for the authors of each paper



    

    # Extract list of "metadata" dictionaries for each paper

    internal_metadata = [i['metadata'] for i in internal_library]



    # Extract the primary location and country for the authors of each paper

    # individual_paper_metadata is the 'metadata' dictionary for one paper

    for individual_paper_metadata in internal_metadata:



        # Extract the list of author dictionaries for the current paper (one dictionary per author)

        authors_information = individual_paper_metadata.get('authors')



        if len(authors_information) == 0:

            primary_location_list.append('None')

            primary_country_list.append('None')



        else:

            location = None

            country = None

            i = 1



            # Find the first author of the paper with valid data for location and country,

            # extract this information, and add to the respective lists for all the papers

            while location == None and i <= len(authors_information):



                if bool(authors_information[i-1].get('affiliation')) == True:



                    location = authors_information[i-1].get('affiliation').get('location').get('settlement')

                    country = authors_information[i-1].get('affiliation').get('location').get('country')



                i += 1



            primary_location_list.append(location)

            primary_country_list.append(country)

                

    

    # Take all the information extracted for each paper and merge it into one combined DataFrame    

    

    # Create a DataFrame with one column - the paper ID for each paper

    index_df = pd.DataFrame(paper_id, columns =  ['paper_id'])



    # Create a DataFrame with two columns - the primary location and country of the authors for each paper

    geographical_df = pd.DataFrame(primary_location_list, columns = ['Location'])

    geographical_df['Country'] = primary_country_list



    # Create a DataFrame with three columns - title, abstract text, and body text for each paper

    paper_info_df = pd.DataFrame(title_list, columns = ['Title'])

    paper_info_df['Abstract'] = abstract_list

    paper_info_df['Text'] = text_list

    

    # Concatenate the above three DataFrames into a single combined DataFrame

    combined_df = pd.concat([index_df, geographical_df, paper_info_df], axis = 1)

    

    # Extract sha (paper ID), abstract text, url, and publication date for each paper from the metadata

    # DataFrame passed as input to this function

    part_1 = metadata[['sha', 'abstract', 'url', 'publish_time']]



    # Merge the information from the metadata DataFrame with the DataFrame with the raw paper data

    test_df = combined_df.merge(part_1, left_on = ['paper_id'], right_on = ['sha'], how = 'left')

    test_df.drop(['sha'], axis = 1,inplace = True)

    test_df = test_df[['paper_id', 'url', 'publish_time', 'Location', 'Country', 'Title', 'Abstract', 'abstract', 'Text']]

    

    # In the event where the JSON's abstract is null but there is an abstract in the 

    # metadata, make the substitution

    test_df['Abstract'] = np.where(test_df['Abstract'] == '', test_df['abstract'], test_df['Abstract'])

    test_df.drop(['abstract'], axis = 1, inplace = True)

    

    gc.collect()

    

    return test_df
# Define as a list the names of all the subdirectories under the "/kaggle/input/CORD-19-research-challenge"

# directory where the dataset files are stored

selected_folders = ['comm_use_subset', 'noncomm_use_subset', 'custom_license', 'biorxiv_medrxiv']



# Call the data_creation() function to extract the desired data from each paper and create a single,

# combined DataFrame, test_df

test_df = data_creation(list_of_folders = selected_folders, metadata = metadata_information)
test_df.to_csv(output_dir + 'Checkpoint_1.csv', index = False)
# Cleaning up after each section to save space

del paper_id_list

del metadata_information

del selected_folders



import gc

gc.collect()
def cleaning_dataset(dataset, columns_to_clean):

    

    '''

    Clean text of specified columns in DataFrame



    Inputs:  dataset (DataFrame) - Dataframe to clean

             columns to clean (list) - list of columns in the DataFrame for which we want to clean the text



    Output:  cleaned DataFrame 

    

    '''    

    

    # each_column is one of the defined columns from the DataFrame

    # Use the log_progress() helper function defined above to indicate the progress of the execution

    for each_column in log_progress(columns_to_clean, every = 1):



        # Fill in any null text items with "No Information"

        dataset[each_column] = dataset[each_column].fillna('No Information')



        # Remove square-bracketed references (i.e., [1])

        dataset[each_column] = dataset[each_column].apply(lambda x: re.sub(r'\[.*?]', r'', x))



        # Remove parenthesis references (i.e., (1))

        dataset[each_column] = dataset[each_column].apply(lambda x: re.sub(r'\((.*?)\)', r'', x))



        # Remove garbage characters

        dataset[each_column] = dataset[each_column].apply(lambda x: re.sub(r'[^a-zA-z0-9.%\s-]', r'', x))



        # Remove unnecessary white space

        dataset[each_column] = dataset[each_column].apply(lambda x: re.sub(r' +', r' ', x))



        # Remove unnecessary white space at the end of the text section

        dataset[each_column] = dataset[each_column].apply(lambda x: x.rstrip())



        # Remove white space before punctuation marks

        dataset[each_column] = dataset[each_column].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))



        

    cleaned_abstract = []     # list of cleaned abstracts for all the papers

    abstract_count = []       # list of the word counts for each paper abstract



    # Clean up abstracts as abstracts may contain unnecessary starting words like 'background' or 'abstract'

    # Count the words in each cleaned abstract and add the list of abstract word counts for each paper to

    # the test_df Data Frame

    #

    # i is the abstract text (string) for one paper

    for i in dataset['Abstract']:



        if i.split(' ')[0].lower() == 'background' or i.split(' ')[0].lower() == 'abstract':

            cleaned_abstract.append(' '.join(i.split(' ')[1:]))

            abstract_count.append(len(i.split(' ')[1:]))



        else:

            cleaned_abstract.append(i)

            abstract_count.append(len(i.split()))



    dataset['Abstract'] = cleaned_abstract

    dataset['Abstract Word Count'] = abstract_count



    # Removing the words figure X.X from the passages because it contributes no meaning

    fig_exp = re.compile(r"Fig(?:ure|.|-)\s+(?:\d*[a-zA-Z]*|[a-zA-Z]*\d*|\d*)", flags=re.IGNORECASE) 

    dataset['Text'] = [(re.sub(fig_exp, '', i)) for i in test_df['Text']]



    # Remove other instances of poor references and annotations

    poor_annotation_exp_1 = re.compile(r'(\d)\s+(\d]*)', flags = re.IGNORECASE)

    dataset['Text'] = [(re.sub(poor_annotation_exp_1, '', i)) for i in test_df['Text']]



    poor_annotation_exp_2 = re.compile(r'(\d])*', flags = re.IGNORECASE)

    dataset['Text'] = [(re.sub(poor_annotation_exp_2, '', i)) for i in test_df['Text']]

    

    gc.collect()

    

    return dataset
text_columns = ['Title', 'Abstract', 'Text']

test_df = cleaning_dataset(dataset = test_df, columns_to_clean = text_columns)
test_df['Abstract'].describe(include='all')
test_df.drop_duplicates(['Abstract', 'Text'], inplace = True)
test_df['Text'].describe(include = 'all')
# Make sure the country name entries in the DataFrame are strings

test_df['Country'] = test_df['Country'].astype(str)



# Extract the country name list from the DataFrame so we can work with it

country_list = test_df['Country'].tolist()



# Clean up the country names



# Substitute "None" for any null entries

# new_items is now a temporary list of the country names

new_items = ['None' if x == 'nan' else x for x in country_list]



# Remove garbage characters, brackets and parentheses, extra white space, etc.

new_items = [re.sub(r'\[.*?]', r'', x) for x in new_items]

new_items = [re.sub(r'\((.*?)\)', r'', x) for x in new_items]

new_items = [x.split(',')[0] for x in new_items]

new_items = [re.sub(r'[^a-zA-z\s-]', r'', x) for x in new_items]

new_items = [re.sub(r' +', r' ', x) for x in new_items]

new_items = [re.sub(r'\[\[|\]\]', '', x) for x in new_items]

new_items = [x.strip('[').strip(']') for x in new_items]



# Ensure that the country names are the uniform, proper English names for each country

new_items = ['mexico' if x.lower() == 'mxico' else x for x in new_items]

new_items = ['brazil' if x.lower().rstrip() == 'brasil' else x for x in new_items]

new_items = ['china' if x.lower() == 'china-japan' or x.lower() == 'prchina' or x.lower() == 'prc' or x.lower() == 'china-australia' else x for x in new_items]

new_items = ['united kingdom' if x.lower() == 'united-kingdom' else x for x in new_items]

new_items = ['tunisia' if x.lower() == 'tunisie' else x for x in new_items]

new_items = ['russia' if x.lower() == 'runion' or x.lower() == 'ussr' else x for x in new_items]

new_items = ['senegal' if x.lower() == 'sngal' else x for x in new_items]

new_items = ['spain' if x.lower() == 'espaa' else x for x in new_items]

new_items = ['slovakia' if x.lower() == 'czechoslovakia' else x for x in new_items]

new_items = ['usa' if x.lower() == 'ljsa' else x for x in new_items]

new_items = ['germany' if x.lower() == 'w-germany' or x.lower() == 'deutschland' else x for x in new_items]

new_items = ['belgium' if x.lower() == 'belgique' else x for x in new_items]

new_items = ['slovenia' if x.lower() == 'yugoslavia' else x for x in new_items]

new_items = ['italy' if x.lower() == 'italien' else x for x in new_items]

new_items = ['emirates' if x.lower() == 'uae' else x for x in new_items]

new_items = ['india' if x.lower() == 'india-' else x for x in new_items]



# Change to "None" any entries that are not actually names of a country

new_items = ['None' if x.lower() == 'umrs' or x.lower() == 'frg' or x.lower() == 'university' or x.lower() == 'maroc' or x.lower() == 'universidade' or x.lower() == 'ucbl' or x.lower() == 'telephone' or x.lower() == 'mcgovern' or x.lower() == 'school' or x.lower() == 'professor' else x for x in new_items]
country_list = []    # list of cleaned and ISO standard country names for all the papers



# Use the pycountry package to find the official ISO standard country names for each

# entry in the temporary list of cleaned country names (new_items). Once the list

# is updated with the standard names, replace this column in the text_df Data Frame

#

# new_items is a temporary list of the cleaned country names

# Use the log_progress() helper function defined above to indicate the progress of the execution

for i in log_progress(new_items, every = 1):

    

    try:

        if len(i.split()) > 1:

            list_to_try = i.split()



            for x in list_to_try:

                try:

                    country = pycountry.countries.search_fuzzy(x)[0].name



                except:

                    continue



            country_list.append(country)



        else:

            country = pycountry.countries.search_fuzzy(i)[0].name

            country_list.append(country)

            

    except:

        country_list.append('None')

        

test_df['Country'] = country_list
test_df.to_csv(output_dir + 'Checkpoint_2.csv', index = False)
# Cleaning up after each section to save space

del text_columns

del country_list

del new_items



gc.collect()
graphing_data = pd.DataFrame(test_df.groupby(['Country']).count()['paper_id']).reset_index()



data = dict (type = 'choropleth',

             locations = graphing_data['Country'],

             locationmode = 'country names',

             colorscale = 'viridis', reversescale = True,

             z = graphing_data['paper_id'])



map = go.Figure(data=[data])



map.update_layout(

    title_text = 'Break down of Research Papers by Countries',

)



map.show()
# Use geonamescache Python library to get city names, their longitude and latitude coordinates, and map

# the cities to their respective states.

#

geoname = geonamescache.GeonamesCache()

global_city_dictionary = geoname.get_cities()



# List of US city names

us_cities_list = [global_city_dictionary[code]['name'] for code in list(global_city_dictionary.keys()) if global_city_dictionary[code]['countrycode'] == 'US']



# List of longitudes for each city

us_cities_lng_list = [global_city_dictionary[code]['longitude'] for code in list(global_city_dictionary.keys()) if global_city_dictionary[code]['countrycode'] == 'US']



# List of latitudes for each city

us_cities_lat_list = [global_city_dictionary[code]['latitude'] for code in list(global_city_dictionary.keys()) if global_city_dictionary[code]['countrycode'] == 'US']







def get_states(longitude, latitude):

    

    '''

    Return a list of US states corresponding to input lists of longitude and latitude coordinates



    Inputs:  longitude (list) - list of longitude coordinates

             latitude (list) - list of corresponding latitude coordinates



    Outputs: us_states (list) - list of US states corresponding to longitude and latitude coordinates



    '''

    

    coord_list = list(zip(latitude, longitude))     # list of longitude-latitude coordinate pairs

    

    # Use reverse_geocoder Python package to get location information (closest city/town, 

    # country, US state, etc.) from longitude and latitude coordinates.

    # The search() method returns a dictionary with the location information for each set of 

    # coordinates in the list. If the location is in a US state, the "admin1" dictionary

    # item should have the state name.

    results = rg.search(coord_list)

    

    us_states = []

            

    for i in results:

        try:

            state = i['admin1']     # retrieving state information

        except:

            state = ''              # return empty string if there is no information

        

        us_states.append(state)

    

    return us_states





# Call the above get_states() function to get the list of states corresponding to the list

# of US city longitude and latitude coordinates

us_states_list = get_states(us_cities_lng_list, us_cities_lat_list)



# Create dictionary mapping for US city names to states

city_to_state = {}

for city, state in zip(us_cities_list, us_states_list):

    if state:

        city_to_state[city] = state
del geoname; del global_city_dictionary

del us_cities_list; del us_cities_lng_list; del us_cities_lat_list; del us_states_list
# Extract rows from test_df for which the paper "Country" is the US

graphing_data_2 = test_df[test_df['Country'] == 'United States']



# For a paper where the primary author country is in the US, the "Location" column should have 

# the city name and state name or abbreviation

location_list = [city_to_state.get(i) for i in graphing_data_2['Location']]

graphing_data_2.loc[:,'Location'] = location_list



# Count the papers by location (US state)

graphing_data_2 = graphing_data_2.groupby(['Location']).count()['paper_id'].reset_index()



# Create a dictionary mapping of US state names to their postal abbreviations

state_dict = {'Alabama': 'AL', 'Alaska': 'AK', 'American Samoa': 'AS', 'Arizona': 'AZ', 'Arkansas': 'AR',

              'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',

              'Delaware': 'DE', 'District of Columbia': 'DC',

              'Florida': 'FL',

              'Georgia': 'GA', 'Guam': 'GU',

              'Hawaii': 'HI',

              'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',

              'Kansas': 'KS', 'Kentucky': 'KY',

              'Louisiana': 'LA',

              'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',

              'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Northern Mariana Islands':'MP',

              'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',

              'Pennsylvania': 'PA', 'Puerto Rico': 'PR',

              'Rhode Island': 'RI',

              'South Carolina': 'SC', 'South Dakota': 'SD',

              'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',

              'Vermont': 'VT', 'Virgin Islands': 'VI', 'Virginia': 'VA',

              'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'

}



# Create the graph that shows paper counts by US state

graphing_data_2['Code'] = graphing_data_2['Location'].map(state_dict)



fig = go.Figure(data=go.Choropleth(

    locations=graphing_data_2['Code'],     # spatial coordinates

    z = graphing_data_2['paper_id'],       # data to be color-coded

    locationmode = 'USA-states',           # set of locations match entries in `locations`

    colorscale = 'viridis', reversescale = True,

))



fig.update_layout(

    title_text = 'Break down of Research Papers by US States',

    geo_scope='usa',                       # limited map scope to USA

)



fig.show()
#Cleaning up after each section to save space

del state_dict

del city_to_state

del location_list

del map; del graphing_data

del fig; del graphing_data_2



gc.collect()
test_df.dropna(subset = ['Text'], inplace = True)
def dimension_reduction(dataset, n = 3, n_components = 3, use_hashing_vectorizer = False):



    '''

    Obtain TF-IDF scores based on n-grams (features) for text in the input dataset. Then use

    t-SNE dimensionality reduction to project the TF-IDF scores on a lower-dimensional space

    that can be visualized



    Inputs:  dataset (DataFrame) - a dataset with text for multiple documents

             n (integer) - defines the n-gram dimension to use in the TF-IDF analysis

             n_components (integer) - defines the dimensionality of the space to project to using t-SNE

             use_hashing_vectorizor (bool) - gives the option to use a different text vectorizing method



    Outputs:  tsne_df (DataFrame) - contains t-SNE output data that can be visualized

 

    '''

    

    dataset = dataset.reset_index().drop(['index'], axis = 1)

    

    # Extract Trigram vectors for all papers in our set and obtain TDF-IDF scores 

    # using the scikit-learn TfidfVectorizer class

    if use_hashing_vectorizer == False:

    

        vectorizer=TfidfVectorizer(ngram_range=(n,n))

        vectorized_vectors=vectorizer.fit_transform(dataset['Text'].tolist())

        

    else:

        

        vectorizer=HashingVectorizer(ngram_range=(n,n))

        vectorized_vectors=vectorizer.fit_transform(dataset['Text'].tolist())



    # Use t-SNE dimensionality reduction (reduce to three dimensions) to identify outliers

    tsne_reduction = TSNE(n_components = 3, perplexity = 10, learning_rate = 100, random_state = 777)

    tsne_data = tsne_reduction.fit_transform(vectorized_vectors)



    # Convert first 3 components of T-SNE into DataFrame for 3-D visualization

    tsne_df = pd.DataFrame(tsne_data, columns = [i for i in range(0, tsne_data.shape[1])])

    

    gc.collect()

    

    return tsne_df







def visualizing_dimensions(dataset):



    '''

    Create a 3-D plot using the data in the input dataset DataFrame



    Input:  dataset (DataFrame) - contains the data to visualize



    Output:  the function does not return anything per se, but generates a 3-D plot based on the input data

   

    '''



    fig = plt.figure(1, figsize=(7, 5))

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



    ax.scatter(dataset[0], dataset[1], dataset[2], c=dataset[2], cmap='viridis', linewidth=0.5)



    ax.set_xlabel('Component A')

    ax.set_ylabel('Component B')

    ax.set_zlabel('Component C')



    plt.show()

    gc.collect()

    

    

    

def outlier_removals(dim_reduced_dataset, dataset, n_components = 3, number_std_dev = 2.5, verbose = 1):

    

    '''

    Remove outlier items from the larger input dataset based on the the metrics in the 

    input dim_reduced_dataset (implicitly this is assumed to be scores for the text in 

    dataset). The criteria for identifying outliers is an item whose score in 

    dim_reduced_dataset is more than number_std_dev away from the mean score in

    any of the axes of dim_reduced_dataset



    Inputs:  dim_reduced_dataset (DataFrame) - contains metrics or scores in n_components 

                                               dimensions (reduced) for items in dataset

             dataset (DataFrame) - larger dataset for which we want to remove outliers 

                                   based on the scores in dim_reduced_dataset

             n_components (integer) - number of dimensions in dim_reduced_dataset

             number_std_dev (float) - criteria for identifying outlier scores

             verbose (int) - if = 1, function prints out additional information



    Output:  cleaned_df (DataFrame) - based on input dataset, but with the identified outlier items removed

  

    '''

    

    outlier_papers = []

    print('{a} standard deviation is being used to clean the dataset.'.format(a = number_std_dev))

    print()

    

    # Identify outlier text based on dim_reduced_dataset scores

    for i in range(0, n_components):

        

        # Create upper and lower bounds for outliers as the mean +/- number_std_dev for each dimension

        upper = dim_reduced_dataset[i].mean() + number_std_dev*dim_reduced_dataset[i].std()

        lower = dim_reduced_dataset[i].mean() - number_std_dev*dim_reduced_dataset[i].std()



        # Extract the outlier items based on the upper and lower bounds and get the index values

        outlier_df = dim_reduced_dataset[(dim_reduced_dataset[i] >= upper) | (dim_reduced_dataset[i] <= lower)]

        outlier_list = outlier_df.reset_index()['index'].tolist()

        

        outlier_papers += outlier_list

    

    # List of outlier item index values

    outlier_papers = list(set(outlier_papers))

    

    # Report how many outliers are identified

    if verbose == 1:

        print('There are {a} outlier papers identified.'.format(a = len(outlier_papers)))

        print()

        

    # Extract the outlier items from the input dataset (all columns)     

    outlier_papers_df = dataset.iloc[outlier_papers,:]

    

    # Display the text of the outlier items

    if verbose == 1:

        print('These are the texts that are determined as abnormal.')

        print()

        for i in outlier_papers_df['Text']:

            print(i)

            print()

    

    # Remove the outliers from the input dataset

    cleaned_df = dataset.drop(outlier_papers, axis = 0)

    cleaned_df.reset_index().drop(columns = ['index'], axis = 1)

    

    gc.collect()

    

    return cleaned_df







def full_cleaning_process(dataset, n = 3, n_components = 3, use_hashing_vectorizer = False, std_dev = 3, verbose = 1):

    

    '''

    Execute a cleaning process on the input dataset, where outliers are 

    identified based on TF-IDF analysis and t-SNE dimensionality reduction, 

    and then removed from the dataset



    Inputs:  dataset (DataFrame) - set of text items for which we want to 

                                   remove outliers

             n (integer) - defines the n-gram dimension to use in the TF-IDF 

                           analysis

             n_components (integer) - defines the dimensionality of the space 

                                      to project to using t-SNE

             use_hashing_vectorizor (bool) - gives the option to use a different 

                                             text vectorizing method

             number_std_dev (float) - criteria for identifying outlier scores

             verbose (int) - if = 1, function prints out additional information



    Output:  output_df (DataFrame) - based on input dataset, but with the identified 

             outlier items removed the function also displays 3-D plots before 

            and after outlier removal

         

    '''

    

    starting_datashape = dataset.shape[0]     # number of items before cleaning

    

    # Complete TF-IDF analysis and t-SNE dimensionality reduction on the input dataset

    dim_reduced_dataset = dimension_reduction(dataset, n = n, n_components = n_components, use_hashing_vectorizer = use_hashing_vectorizer)

    print('Before Cleaning Up -')

    

    # Show the 3-D plot of scores before removing outliers

    visualizing_dimensions(dim_reduced_dataset)

    

    # Identify and remove outliers 

    output_df = outlier_removals(dim_reduced_dataset, dataset, n_components = n_components, number_std_dev = std_dev, verbose = verbose)

    

    ending_datashape = output_df.shape[0]     # number of items after cleaning

    

    print('{a} rows were dropped in this cleaning process.'.format(a = starting_datashape - ending_datashape))

    print()

    

    # Show the 3-D plot of scores after removing outliers

    print('After Cleaning Up -')

    visualizing_dimensions(dimension_reduction(output_df, n = 3, n_components = 3, use_hashing_vectorizer = False))

    

    gc.collect()

    

    return output_df     # cleaned DataFrame with outlier items removed
test_df = full_cleaning_process(test_df, std_dev = 2.5)
# Set the minimum word count for text of papers we want to keep

minimum_word_count = 150



# Reset the index for the DataFrame and drop the "index" column that gets created as part of the reset

test_df = test_df.reset_index().drop(['index'], axis = 1)



# Create a new column that contains the word count for each paper

test_df['Text Word Count'] = [len(i.split()) for i in test_df['Text']]



dirty_list = []     # list of paper indexes that have a word count <= minimum_word_count



for index, value in test_df.iterrows():

    

    if (value['Text Word Count'] <= minimum_word_count):

        dirty_list.append(index)



# Extract the papers that contain less than the minimum word count        

weird_papers_df = test_df.iloc[dirty_list,:]



# Print the text of those papers

for index, value in weird_papers_df.iterrows():

    print(value['Text Word Count'], value['Text'])

    print()
test_df = test_df.drop(dirty_list, axis = 0)

test_df = test_df.reset_index().drop(['index'], axis = 1)
visualizing_dimensions(dimension_reduction(test_df, n = 3, n_components = 3, use_hashing_vectorizer = False))
test_df.to_csv(output_dir + 'Checkpoint_3.csv', index = False)
#Cleaning up after each section to save space

gc.collect()
# Load the spaCy en_core_sci_lg English biomedical language model

nlp = en_core_sci_lg.load()



# Add language detection to the spaCy Natural Language Processing pipeline

nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
language_list = []     # list of primary languages and match scores for each paper



# For each paper in the cleaned dataset, run the text through the spaCy NLP pipeline, and

# add the detected language and match score to language_list[] (use the log_progress() helper

# function to provide the user with a progress bar).

for i in log_progress(test_df['Text'], every = 1):

    

    # To avoid exceeding memory allocation limits, we limit the text set to the pipeline

    # to 1,000,000 characters. This should be more than sufficient to get a good

    # language match.    

    if len(i) <= 1000000:

    

        doc = nlp(i)

        language_list.append(doc._.language)

        

    else:

        

        cut_off_index = i[:1000000].rfind('.')

        focus_i = i[:cut_off_index + 1]

        

        doc = nlp(focus_i)

        language_list.append(doc._.language)
# Convert the language name/abbreviation into upper case

filtered_language_list = [i['language'].upper() for i in language_list]



# Add the language for each paper to the test_df DataFrame

test_df['Language'] = filtered_language_list



# Filter out only research papers in English to perform topic modelling.

english_df = test_df[test_df['Language'] == 'EN']

print('There are {a} research papers in English out of {b} research papers.'.format(a = english_df.shape[0], b = test_df.shape[0]))
cleaned_lemma_list = []     # list of lemmas for the text of each paper; each list item is a string with the

                            # lemmas for the text of the paper, separated by a blank space



# Use the spaCy NLP library to lemmatize the paper text.

# Use the log_progress() helper function defined above to indicate the progress of the execution.

for i in log_progress(english_df['Text'], every = 1):

    

    # Variable length for efficiency

    nlp.max_length = len(i)



    # Create spaCy "Doc" for the text of each paper

    doc = nlp(i)

    

    # Extract the lemma for each token and join in one text string

    cleaned_lemma_list.append(" ".join([token.lemma_ for token in doc]))

    

english_df['Lem Text'] = cleaned_lemma_list
import ktrain

ktrain.text.preprocessor.detect_lang = ktrain.text.textutils.detect_lang



max_vocab = int(english_df['Text Word Count'].describe().max())     # set the maximum vocabulary size

num_of_topics = int((len(english_df['Text Word Count']) / 2)**0.5)  # set the number of topics

#max_vocab = 10000



# Run topic modeling using Keras wrappers on LDA

tm = ktrain.text.get_topic_model(english_df['Lem Text'], n_topics = num_of_topics, n_features = max_vocab, lda_max_iter = 25, lda_mode = 'batch')
tm.print_topics()
%%time

threshold_value = 0.25

tm.build(english_df['Lem Text'], threshold = threshold_value)
topic_list = []              # list of topic numbers for the primary topic for each paper

topic_words_list = []        # list of the words for the primary topic of each paper



# Iterate through the lemmatized paper texts

# Use the log_progress() helper function to give the user a progress indication on the execution

for i in log_progress(english_df['Lem Text'], every = 1):

    

    # Identify the most likely topic or primary topic for the paper text

    topic_list.append(np.argmax(tm.predict([i])))

    

    # Identify the string of words for the most likely topic for the paper text

    topic_words_list.append(tm.topics[np.argmax(tm.predict([i]))])



# Add the list of most likely topics for each paper, and the corresponding topic word string to english_df     

english_df['Topic Number'] = topic_list

english_df['List of Topics'] = topic_words_list
tm.save(output_dir + 'Topic Model')
def get_random_colors(n, name='hsv', hex_format=True):



    '''

    Returns an array that maps each index in 0, 1, ..., n-1 to a distinct

    RGB color; the keyword argument name must be a standard mpl colormap name.

    

    '''



    from matplotlib.colors import rgb2hex



    cmap = plt.cm.get_cmap(name, n)

    result = []

    for i in range(n):

        color = cmap(i)

        if hex_format: color = rgb2hex(color)

        result.append(color)

    return np.array(result)







def visualize_documents(texts=None, doc_topics=None, 

                        width=700, height=700, point_size=5, title='Document Visualization',

                        extra_info={},

                        colors=None,

                        filepath=None,):

    '''

    Generates a visualization of a set of documents based on a topic model.

    If <texts> is supplied, raw documents will be first transformed into 

    document-topic matrix.  If <doc_topics> is supplied, then this will be 

    used for visualization instead.

        

    Inputs:  texts(list of str) - list of document texts.  

                                  Mutually-exclusive with <doc_topics>

             doc_topics(ndarray) - pre-computed topic distribution for each 

                                   document in texts.

                                   Mutually-exclusive with <texts>.

             width(int) - width of image

             height(int) - height of image

             point_size(int) - size of circles in plot

             title(str) - title of visualization

             extra_info(dict of lists) - A user-supplied information for each  

                                             datapoint (attributes of the datapoint).

                                             The keys are field names.  The values are 

                                             lists - each of which must be the same 

                                             number of elements as <texts> or <doc_topics>. 

                                             These fields are displayed when hovering over 

                                             datapoints in the visualization.

            colors(list of str) - list of Hex color codes for each datapoint.

                                  Length of list must match either len(texts) or doc_topics.shape[0].

            filepath(str) - Optional filepath to save the interactive visualization

            

    Output:  visualization displayed to stdout

        

    '''

    

    # error-checking

    if texts is not None: 

        length = len(texts)

    else: 

        length = doc_topics.shape[0]



    if colors is not None and len(colors) != length:

        raise ValueError('length of colors is not consistent with length of texts or doctopics')

    if texts is not None and doc_topics is not None:

        raise ValueError('texts is mutually-exclusive with doc_topics')

    if texts is None and doc_topics is None:

        raise ValueError('One of texts or doc_topics is required.')

    if extra_info:

        invalid_keys = ['x', 'y', 'topic', 'fill_color']

        for k in extra_info.keys():

            if k in invalid_keys:

                raise ValueError('cannot use "%s" as key in extra_info' %(k))

            lst = extra_info[k]

            if len(lst) != length:

                raise ValueError('texts and extra_info lists must be same size')



    # check fo bokeh

    try:

        import bokeh.plotting as bp

        from bokeh.plotting import save

        from bokeh.models import HoverTool

        from bokeh.io import output_notebook

    except:

        warnings.warn('visualize_documents method requires bokeh package: pip3 install bokeh')

        return



    # prepare data

    if doc_topics is not None:

        X_topics = doc_topics

    else:

        if tm.verbose:  print('transforming texts...', end='')

        X_topics = tm.predict(texts, harden=False)

        if tm.verbose: print('done.')



    # reduce to 2-D

    if tm.verbose:  print('reducing to 2 dimensions...', end='')

    tsne_model = TSNE(n_components=2, verbose=tm.verbose, random_state=777, angle=.99, init='pca')

    tsne_lda = tsne_model.fit_transform(X_topics)

    print('done.')



    # get random colormap

    colormap = get_random_colors(tm.n_topics)



    # generate inline visualization in Jupyter notebook

    lda_keys = tm._harden_topics(X_topics)

    if colors is None: colors = colormap[lda_keys]

    topic_summaries = tm.get_topics(n_words=5)

    os.environ["BOKEH_RESOURCES"]="inline"

    output_notebook()

    dct = { 

            'x':tsne_lda[:,0],

            'y':tsne_lda[:, 1],

            'topic':[topic_summaries[tid] for tid in lda_keys],

            'fill_color':colors,}

    tool_tups = [('index', '$index'),

                 ('(x,y)','($x,$y)'),

                 ('topic', '@topic')]

    for k in extra_info.keys():

        dct[k] = extra_info[k]

        tool_tups.append((k, '@'+k))



    source = bp.ColumnDataSource(data=dct)

    hover = HoverTool( tooltips=tool_tups)

    p = bp.figure(plot_width=width, plot_height=height, 

                  tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],

                  title=title)

    p.circle('x', 'y', size=point_size, source=source, fill_color= 'fill_color')

    bp.show(p)

    if filepath is not None:

        bp.output_file(filepath)

        bp.save(p)

    return
text_information = {'Paper': english_df['paper_id'].astype(str).tolist(), 'Topic': english_df['Topic Number'].astype(str).tolist(),'Title': english_df['Title'].tolist()}

visualize_documents(texts = english_df['Lem Text'].tolist(), extra_info = text_information)
dtm = tm.vectorizer.fit_transform(english_df['Lem Text'].tolist()) 

LDAvis_prepared = pyLDAvis.sklearn.prepare(tm.model, dtm, tm.vectorizer)

#pyLDAvis.enable_notebook()

#pyLDAvis.display(LDAvis_prepared, template_type='notebook')

#display(pyLDAvis.display(LDAvis_prepared))

pyLDAvis.save_html(LDAvis_prepared, output_dir + 'lda_visualization.html')
os.chdir('/kaggle/working')
graphing_data_3 = english_df[['Topic Number', 'List of Topics']] 

graphing_data_3 = graphing_data_3.assign(Keywords=graphing_data_3['List of Topics'].str.split(' ')).explode('List of Topics')

graphing_data_3 = graphing_data_3.explode('Keywords')

graphing_data_3 = graphing_data_3[['Topic Number', 'Keywords']]



graphing_data_3 = graphing_data_3.groupby(['Topic Number', 'Keywords']).size().reset_index(name='Count')

graphing_data_3 = graphing_data_3.sort_values(by='Count', ascending=False)



fig = px.bar(graphing_data_3, y="Topic Number", x="Count", color='Keywords', orientation = 'h', height = 1000)

fig.show()
test_df = test_df.reset_index()

english_df = english_df[['Lem Text', 'Topic Number', 'List of Topics']].reset_index() #This works because english_df is a subset of the main df (Lem Text)



test_df = test_df.merge(english_df, on = 'index', how = 'left')
test_df.to_csv(output_dir + 'Checkpoint_4.csv', index = False)
final_value_list = []     # list of the occurrence count for the most frequent topic word for each paper

final_topic_list = []     # list of the topic word with the maximum occurrence count for each paper



# Iterate over the rows of test_df (i.e., each paper)

for index, value in test_df.iterrows():

    

    count_value_list = []     # list where each item is the number of times each topic word for a paper

                              # appears in the paper lemmatized text

    

    count_topic_list = []     # list of the topic words for a paper

    

    # The "List of Topics" column contains the most likely topic word string for each paper

    if pd.isnull(value['List of Topics']) == True:

        

        final_value_list.append(None)

        final_topic_list.append(None)

        

    else:

    

        # Split the topic word string for each paper into a list of the individual words

        list_of_topics = value['List of Topics'].split()



        # each_topic is one of the topic words of the topic for the paper

        for each_topic in list_of_topics:



            # Count the number of times the each topic word appears in the paper lemmatized text string

            count_value = value['Lem Text'].count(each_topic)

            

            # Add the count for the current topic word to the list

            count_value_list.append(count_value)

            

            # Add the current topic word to the list

            count_topic_list.append(each_topic)



        # Get the highest occurrence count for all the topic words for the current paper   

        max_value = max(count_value_list)

        

        # Get the index of the max_value in the count value list (maps to the topic word with the highest

        # occurrence in the paper text)

        max_index = count_value_list.index(max_value)

        

        # Get the topic word that has the highest occurrence count in the paper text

        max_topic = count_topic_list[max_index]



        # Add the topic word that occurs most frequently in the paper text, and the corresponding

        # occurrence count, to the appropriate list

        final_value_list.append(max_value)

        final_topic_list.append(max_topic)

        

# Add a column to test_df that has the single topic word that occurs most frequently in the paper

# text; this is classified as the sub-topic for the paper

test_df['Sub-Topic'] = final_topic_list
top_N = 50

a = test_df['List of Topics'].str.cat(sep=' ')

words = nltk.tokenize.word_tokenize(a)

word_dist = nltk.FreqDist(words)

rslt = pd.DataFrame(word_dist.most_common(top_N), columns=['Keywords', 'Frequency'])
del a; del words; del word_dist
d = {}

for a, x in rslt.values:

    d[a] = x



def f(word_count):

    

    '''

    Generate a WordCloud visualization with a specified 

    number of words

    

    '''

    

    wordcloud = WordCloud(width=1600,height=800,background_color="white",max_words=word_count)

    wordcloud.generate_from_frequencies(frequencies=d)

    plt.figure(figsize=(20,8), facecolor='k')

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
widgets.interact_manual(f, word_count=widgets.IntSlider(min=10, max=50, step=5, value=20))#This is purely for publishing purpose to showcase interactivity

widgets.interact(f, word_count=widgets.IntSlider(min=10, max=50, step=5, value=20))
fig = px.bar(rslt, y='Keywords', x='Frequency', orientation = 'h',

             color='Frequency', color_continuous_scale='viridis',

             labels={'pop':'Count of Keywords'}, height=600)

fig.show()
import plotly.offline as py_offline



py_offline.init_notebook_mode(connected = True)



def update_plot(Frequency):

    

    '''

    Generate a plot of term frequency for the papers in the dataset

    

    '''

    

    filtered_df = rslt.query('Frequency> ' + str(Frequency))

    data = [go.Bar(x = filtered_df.Keywords,

                   y = filtered_df.Frequency)

                   ]

    layout = go.Layout(

        xaxis = dict(title = 'Keywords'),

        yaxis = dict(title = 'Frequency'),

        title = f'Words with Frequency greater than {Frequency}')

    chart = go.Figure(data = data, layout = layout, )

    py_offline.iplot(chart, filename = 'Keyword Frequency')



widgets.interact_manual(update_plot, Frequency=widgets.IntSlider(min=100, max=1000, step=100, value=200)) #This is purely for publishing purpose to showcase interactivity

widgets.interact(update_plot, Frequency=widgets.IntSlider(min=100, max=1000, step=100, value=200))
# Cleaning up after each section to save space

del language_list; del filtered_language_list

del cleaned_lemma_list

del max_vocab; del num_of_topics

del topic_list; del topic_words_list

del text_information

del dtm

del LDAvis_prepared

del graphing_data_3

del english_df

del count_value_list; del count_topic_list

del final_topic_list; del final_value_list

del list_of_topics



gc.collect()
def creating_search_field(question):

    

    '''

    Take in a question and create a search field (list of search words and synonyms) 

    to identify relevant papers

    

    Input:  question (string) - input question from the search engine query

    

    Output: search field synonyms (list) - list of search words with associated synonyms

    

    '''

    

    # Call the process text() helper function to clean and lemmatize the question string.

    # task_question is a spaCy "Doc" object created from the string lemmatized question text

    task_question = nlp(process_text(question))

    

    search_field = []     # list of search words extracted from the input question

    

    # Examine each token in the question, and if the token is a noun or a verb, add this

    # as a search word to the search_field[] list

    for token in task_question:

        if token.pos_ == 'NOUN' or token.pos == 'VERB':

            search_field.append(token.lemma_)

            

    search_field_synonyms = []     # list of synonyms for the search words

    

    # i is one of the search words in the search_field[] list

    for i in search_field:

        

        # Use the Python nltk wordnet library to generate synonyms for each search word

        syns = wordnet.synsets(i)

        for x in syns:

            search_field_synonyms.append(x.name().split('.')[0])

    

    # Combine the synonyms and the original search word list into a single list      

    search_field_synonyms = list(set(search_field_synonyms + search_field))

    

    # Due to the limited number of papers that have the top keywords of 'coronavirus' and 'covid-19', 

    # we create an artifical list to widen search. This will not be necessary as number of papers increases.

    if 'coronavirus' or 'covid-19' or 'covid19' or 'covid' or 'sars' in search_field_synonyms:

        

        search_field_synonyms.append('cov')

        search_field_synonyms.append('sars-cov-2')

        search_field_synonyms.append('sars-cov')

        search_field_synonyms.append('pdcov')

        search_field_synonyms.append('mers')

        search_field_synonyms.append('sars')

        search_field_synonyms.append('mers-cov')

        search_field_synonyms.append('ncov')

    

    return search_field_synonyms



def search_database(dataset, search_field_synonyms, area_of_search = 'Sub-Topic'):

    

    '''

    Create a DataFrame consisting of only the relevant papers based on a list of search words

    

    Inputs:  dataset (DataFrame) - set of papers to extract the relevant ones based on search words

             search_field_synonyms (list) - list of search words (strings)

             area_of_search (string) - column in dataset to compare against the search words to

                                       identify relevant papers



    Output:  search_df (DataFrame) - set of relevant papers based on the search words

    

    A paper is considered relevant if its associated sub-topic matches one of the

    search words.

    

    '''

    

    search_df = pd.DataFrame()     # set of relevant papers based on the search words

    

    # i is one of the search words

    for i in search_field_synonyms:

        

        # Iterate over the non-null values in the target column of dataset

        for x in filter(None,dataset[area_of_search].unique().tolist()):

            

            if i in x:

                

                # search word matches paper - extract corresponding row of dataset

                subset_df = dataset[dataset[area_of_search] == x]

                

                # add this row to the set of relevant papers

                search_df = pd.concat([search_df, subset_df], axis = 0)

    

    # Create a list of the relevant paper indexes

    index_list = search_df['index'].unique().tolist()

    

    # Reset the index for the set of relevant papers

    search_df.reset_index().drop(['index'], axis = 1, inplace = True)

   

    # Fill in any holes in the "url" column of the set of relevant papers

    search_df['url'].fillna(value = 'None', inplace = True) 

    

    # Create a list of the topic numbers from the topic model for the set of relevant papers

    topic_list = search_df['Topic Number'].unique().tolist()

    

    # If the set of relevant papers is less than 50, add any papers not already in the set

    # with a topic number matching any of the topic numbers of the papers in the relevant set.

    if search_df.shape[0] <= 50:

        

        subset_df = dataset[dataset['Topic Number'].isin(topic_list)]

        subset_df = subset_df[~subset_df['index'].isin(index_list)]

        

        search_df = pd.concat([search_df, subset_df], axis = 0)

    

    gc.collect()

    

    return search_df







def search_engine(dataset, question):

    

    '''

    Compare the input question with the set of relevant papers to find the best matching papers

    

    Inputs:  dataset (DataFrame) - contains only the set of relevant papers corresponding to the

                                   input question

             question (string) - question we want to answer from the set of relevant papers

            

    Outputs: dataset (DataFrame) - modified set of relevant papers that has been scored and sorted

    

    For each paper in the set, we evaluate each sentence of the paper text for similarity to the

    input question using the spaCy Doc object similarity score. We then select the sentence

    with the best score, along with the sentences before and after for context. We add

    the information on the selected sentence and score to the input dataset and sort the papers

    in the set based on the scores.

    

    '''

        

    prior_sentence_list = []        # list of sentences before the selected sentence for each paper

    selected_sentence_list = []     # list of selected sentences for each paper (best similarity score)

    posteriori_sentence_list = []   # list of sentences after the selected sentence for each paper

    

    sentence_score_list = []        # list of best similarity score for each paper

    sentence_position_list = []     # list of selected sentence position for each paper

    

    document_score_list = []        # list of overall scores for each paper

    document_length_list = []       # list of the number of sentences for each paper

    

    # Call the process_text() helper function to clean and lemmatize the question string.

    # task_question is a spaCy "Doc" object created from the string lemmatized question text

    task_question = nlp(process_text(question))

    

    # Reset the index on the input dataset

    dataset = dataset.reset_index().drop(['index'], axis = 1)



    # i is the text for one paper (string)

    # Use the log_progress() helper function to give the user a progress indication on the execution

    for i in log_progress(dataset['Text'], every = 1):

        

        sentence_list = []                # list of sentences (strings) from the current paper text

        sentence_similarity_list = []     # list of similarity scores for each sentence

        position_list = []                # list of positions for each sentence (1,  2, 3, etc.)

        

        # Performing Sentence Boundary Detection

        nlp.max_length = len(i)*1.1

        

        # document is a spaCy "Doc" object from the paper sentence text

        document = nlp(i)

        

        count = 1     # keep track of sentence position (first sentence, second sentence, etc.)

        

        # Count the number of sentences in the paper text

        number_of_sentences_in_document = len(list(document.sents))

        

        # Perform sSemantic Search - for each sentence, get similarity score relative to the input question

        # each_sentence is one sentence in the paper text

        for each_sentence in document.sents:

              

            # Call the process_text() helper function to clean and lemmatize the current sentence.

            # sentence_search is a spaCy "Doc" object created from the string lemmatized sentence text

            # for the current sentence            

            sentence_search = nlp(process_text(str(each_sentence)))

            

            if (sentence_search.vector_norm):

                

                # Get similarity score for each sentence relative to the input question

                similarity_score = task_question.similarity(sentence_search)

                

                # Add the sentence text, similarity score, and position to the appropriate lists

                sentence_list.append(each_sentence)

                sentence_similarity_list.append(similarity_score)

                position_list.append(count)

                

            count += 1     # increment sentence position counter

        

        # Identify the maximum similarity score for all the sentences in the current document

        max_similarity_score = max(sentence_similarity_list)

        

        # Identify the position of the sentence in the current document with the maximum score

        pos_max_similarity_score = sentence_similarity_list.index(max_similarity_score)



        # Get the sentence with the best score and its position in the document

        max_similarity_sentence = sentence_list[pos_max_similarity_score]

        max_position = position_list[pos_max_similarity_score]



        # Compute an average score for the current paper

        overall_score = round(sum(sentence_similarity_list) / len(sentence_similarity_list),4)

        

        # Identify sentences before and after the selected sentence for

        # each paper to provide contextual understanding.

        # Handle various edge cases in addition to the normal case

        if len(sentence_list) == 1:                                         # edge case 1



            start_sentence = ''

            end_sentence = ''



        elif len(sentence_list) == 2 and max_position == position_list[0]:  # edge case 2



            start_sentence = ''

            end_sentence = sentence_list[pos_max_similarity_score + 1]



        elif len(sentence_list) == 2 and max_position == position_list[-1]: # edge case 3



            start_sentence = sentence_list[pos_max_similarity_score - 1]

            end_sentence = ''



        elif len(sentence_list) > 2 and max_position == position_list[0]:   # edge case 4



            start_sentence = ''

            end_sentence = sentence_list[pos_max_similarity_score + 1]



        elif len(sentence_list) > 2 and max_position == position_list[-1]:  # edge case 5



            start_sentence = sentence_list[pos_max_similarity_score - 1]

            end_sentence = ''



        else:                                                               # normal case



            start_sentence = sentence_list[pos_max_similarity_score - 1]

            end_sentence = sentence_list[pos_max_similarity_score + 1]

        

        

        # Add all the information to the appropriate lists defined above

        

        prior_sentence_list.append(start_sentence)

        selected_sentence_list.append(max_similarity_sentence)

        posteriori_sentence_list.append(end_sentence)



        sentence_score_list.append(max_similarity_score)

        sentence_position_list.append(max_position)



        document_score_list.append(overall_score)

        document_length_list.append(number_of_sentences_in_document)

     

    # Add columns to the set of relevant papers based on the search and scoring results

    dataset['Top Sentence'] = selected_sentence_list

    dataset['Sentence Score'] = sentence_score_list

    dataset['Document Score'] = document_score_list

    dataset['Top Sentence Location'] = sentence_position_list

    dataset['Start Sentence'] = prior_sentence_list

    dataset['End Sentence'] = posteriori_sentence_list

    dataset['Document Length'] = document_length_list



    # Sort the set of relevant papers by document score and sentence score values (best scores first)

    dataset = dataset.sort_values(by=['Document Score', 'Sentence Score'], ascending=False).reset_index().drop(['index'], axis = 1)

    

    gc.collect()

    

    return dataset







def generating_best_sentence(dataset):

    

    '''

    From the scored and sorted set of papers relative to the input question, identify the

    paper with the best matching sentence, and extract certain key information from that paper

    

    Inputs:  dataset (DataFrame) - the set of papers that have been scored

                                   and sorted for best matching sentences 

                                   

    Outputs: top_paper_url   - url of the selected paper (string)

             top_paper_title - title of the selected paper (string)

             top_score       - best sentence score of the selected paper (float)

             top_sentence    - best sentence of the selected paper (string)

             top_position    - best sentence position in the selected paper (integer)

             start_sentence  - sentence before the best sentence in the selected paper (string)

             end_sentece     - sentence after the best sentence in the selected paper (string)

    

    '''

    

    # Identify the paper from the set that has the best matching sentence for the input question

    row = dataset[dataset['Sentence Score'] == dataset['Sentence Score'].max()].reset_index()

    

    # Extract the information we want about the best matching paper

    

    top_score = row['Sentence Score'][0]

    top_sentence = row['Top Sentence'][0]

    top_position = row['Top Sentence Location'][0]



    start_sentence = row['Start Sentence'][0]

    end_sentence = row['End Sentence'][0]

    

    top_paper_url = row['url'][0]

    top_paper_title = row['Title'][0]

    

    gc.collect()

    

    return top_paper_url, top_paper_title, top_score, top_sentence, top_position, start_sentence, end_sentence







def generating_summariaries(dataset, top_n = 10):

    

    '''

    Take the top matching papers from the larger scored and sorted set of papers, and

    add a text summary for each paper using the spaCy NLP pipeline

    

    Inputs:  dataset (DataFrame) - the set of papers that have been scored

                                   and sorted for best matching sentences 

             top_n (integer) - number of matching papers to identify from the input set

    

    Output: focus_df(DataFrame) - set of the top top_n matching papers with summaries

                                  for each paper added to the paper data

    '''



    # Take the first top_n papers from the scored and sorted set of papers 

    focus_df = dataset.head(top_n)

    

    # Instantiate a summarizer in the spaCy pipeline

    summarizer = pipeline(task="summarization", model="bart-large-cnn")

    

    summary_list = []     # list of summaries (text strings) for the top top_n papers



    # For each paper in the top top_n set, generate summary text for the paper

    # Use the log_progress() helper function to give the user a progress indication on the execution

    for i in log_progress(focus_df['Text'], every = 1):

        

        summary = summarizer(i)

        summary_list.append(summary[0]['summary_text'])

    

    

    # Add the summary text for each paper in the top top_n set as a new column

    focus_df['Summary'] = summary_list

    

    gc.collect()

    

    return focus_df







def generating_answers(dataset, question):

    

    '''

    Take the top matching papers from the larger scored and sorted set of papers, and

    generate an answer (text statement) to the input question, based on the abstract

    and body texts for the selected papers, using the spaCy NLP pipeline

    

    Inputs:  dataset (DataFrame) - the set of papers that have been scored

                                   and sorted for best matching sentences

             question (string) - input question from the search engine query

                         

    Output:  printout_answer (string) - a text statement that gives the best answer

                                        to the input question that we can extract 

                                        from the set of papers.

    

    '''

    

    # Re-sort the input dataset by the sentence score for each paper

    focus_df = dataset.sort_values(by=['Sentence Score'], ascending=False).head(150)

    

    # Instantiate a question-answering task in the spaCy pipeline

    question_answer = pipeline(task="question-answering")

    

    ret_list = []    # list of abstract text, the top sentence, and the sentences before and after

                     # for the papers in focus_df

    

    for i in list(zip(focus_df['Abstract'].astype(str), focus_df['Start Sentence'].astype(str), focus_df['Top Sentence'].astype(str), focus_df['End Sentence'].astype(str))):

        ret_list.append(' '.join(i))

        

    # Join all the paper abstract texts and selected sentences together in one string to provide

    # as input to the spaCy question-answering task    

    context_generation = ' '.join(ret_list)



    # Get the text answer and score (how good does spaCy think the answer is) 

    # generated for the input question

    generated_answer = question_answer({'question': question, 'context': context_generation})['answer'].capitalize()

    generated_score = question_answer({'question': question, 'context': context_generation})['score']

    

    # Generate the answer text to return

    if float(generated_score) >= 0.5:

        printout_answer = str('I found a good answer to your question - ') + generated_answer

        

    elif float(generated_score) < 0.5 and float(generated_score) >= 0.3:

        printout_answer = str('Please try to refine your question. In the meantime, here you go - ') + generated_answer

    

    else:

        printout_answer = str('I am thoroughly confused. But here is my best answer - ') + generated_answer

    

    gc.collect()

    

    return printout_answer







def in_depth_display(row_dataset):

    

    '''

    Take one paper from the set of top top_results matching papers and use 

    the spaCy NLP library to identify entities present in the paper

    

    Input:  row_dataset (DataFrame) - one paper from the set of top top_results

                                      matching papers

    

    Output:  print to stdout the list of entities identified in the paper

    

    '''

    

    ent_list = []     # list of entities identified in the paper (list of strings)



    # Create a spaCy "Doc" object for the body text of the paper

    nlp.max_length = len(row_dataset['Text'])

    doc = nlp(row_dataset['Text']) 

    #displacy.render(doc, style='ent')

    

    # Extract each entity from the "Doc" object and add it to the paper entity list

    for ent in doc.ents:

        ent_list.append(ent.text)

        

    cleaned_ent_list = [i.lower() for i in ent_list]

    

    gc.collect()

    

    # Sort the list of entities and print it to stdout as a single string

    print(', '.join(sorted(list(set(ent_list)))))



    

    

def generate_report(dataset, search_question = 'What do we know about COVID-19 risk factors?', top_results = 5):

    

    '''

    Generate a report of the search engine results for an input question. This is the main function

    that is called to exercise the search engine.



    Inputs:  dataset (DataFrame) - set of papers to analyze to answer the question

             search_question (string) - question which we want to answer from the dataset

             top_results (integer) - number of relevant papers we want to identify from the dataset



    Outputs:  report (printed to stdout) that includes the best answer to the question and a

              short list of relevant papers

              

    The output report includes the following:

    

       - the input question

       - the best answer we can get for the question (text statement)

       - the number of papers found to be relevant to the question

       - the single best matching paper for the question:

           * paper url

           * paper title

           * top highlights from the paper (best matching sentence in context)

       - top top_results papers that best match the question; for each paper:

           * url

           * overall score

           * title

           * key sentence in context

           * summary

           * list of entities

    

    '''

    

    question = search_question

    nlp = en_core_sci_lg.load()

    

    # Generate the list of search words for the input question (creating_search_field() function above)

    search_list = creating_search_field(question)

    

    # Generate the set of relevant papers based on the search words (search_database() function above)

    search_df = search_database(dataset, search_list)

    

    # Get the scored and sorted set of the relevant papers (search_engine() function above)

    result_df = search_engine(search_df, question)

    

    # Get the key information from the paper with the best matching sentence to the input

    # question (generating_best_sentence() function above)

    top_paper_url, top_paper_title, top_score, top_sentence, top_position, start_sentence, end_sentence = generating_best_sentence(result_df)    

    

    # Select the top top_results matching papers, and add a text summary for each

    # paper to the paper data (generating_summaries() function above)

    output_df = generating_summariaries(result_df, top_n = top_results)

    

    # Generate the best answer (text statement) to the input question

    # (generating_answers() function above)

    printout_answer = generating_answers(result_df, question) 

    

    # Print the output report

    

    print('Question : {q1}'.format(q1 = question))

    print('Answer : {q2}'.format(q2 = printout_answer))

    print()

    print("``````````````````````````````````````````````````````````````````````````````````````````````````")

    print('{a} out of {b} Research Papers are related to question'.format(a = search_df.shape[0], b = dataset.shape[0]))

    print()

    print('Paper URL : {x}'.format(x = top_paper_url))

    print('Title of Paper : {f}'.format(f = top_paper_title))

    print('Top Highlights : [{c}] | Sentence {d} | {e1} \033[1;34;47m {e2} \033[0m {e3}'.format(c = round(top_score,4), d = top_position, e1 = start_sentence, e2 = top_sentence, e3 = end_sentence))

    print()

    print("``````````````````````````````````````````````````````````````````````````````````````````````````")

    print()



    counter = 1

    

    while counter <= top_results:



        print(str(counter) + '.')

        print('Paper URL : {x}'.format(x = output_df['url'][counter - 1]))

        print('Paper Overall Score : {y}'.format(y = output_df['Document Score'][counter - 1]))

        print('Title of Paper : {f}'.format(f = output_df['Title'][counter - 1]))

        print('Key Sentence : [{g}] | Sentence {h1} out of {h2} | {i1} \033[1;34;47m {i2} \033[0m {i3}'.format(g = round(output_df['Sentence Score'][counter - 1],4), h1 = output_df['Top Sentence Location'][counter - 1], h2 = output_df['Document Length'][counter - 1], i1 = output_df['Start Sentence'][counter - 1], i2 = output_df['Top Sentence'][counter - 1], i3 = output_df['End Sentence'][counter - 1]))

        print()

        print('Summary of Paper: {z}'.format(z = output_df['Summary'][counter - 1]))

        print()

        print('List of Entities in Paper: ')

        in_depth_display(output_df.iloc[counter - 1])

        print()

        counter += 1

        

    gc.collect()
test_df = test_df.drop(columns=['Lem Text'])
nlp = en_core_sci_lg.load()   # Scientific NLP Library
def widget_interaction(task_question):

    print('Loading your Search!')

    generate_report(test_df, search_question = task_question, top_results = 5)

    print()

    print('Completed!')

    

#Just to show interactive ability when published on Kaggle <- IGNORE this if you are running on your local systems

widgets.interact_manual(widget_interaction,

                         task_question= ['What is known about coronavirus transmission, incubation, and environmental stability?',

                                        'What is the incubation period for coronavirus in humans?',

                                        'How does the incubation periods for coronavirus varies with age and health?',

                                        'How long does a human remain contagious with coronavirus?',

                                        'How long does a human remain contagious with coronavirus after recovery?',

                                        'What is the prevalence of coronavirus asymptomatic shedding?',

                                        'What is the prevalence of coronavirus asymptomatic shedding in children?',

                                        'What is the prevalence of coronavirus transmission in human?',

                                        'What is the prevalence of coronavirus transmission in children?',

                                        'How does seasonality affect coronavirus transmission?',

                                        'What is the physical science behind coronavirus?',

                                        'What is the charge distribution of coronavirus?',

                                        'What is coronavirus adhesive ability to hydrophilic or phobic surfaces?',

                                        'What is coronavirus environmental survivability?',

                                        'What is coronavirus viral shedding ability?',

                                        'What is the persistence of coronavirus on surfaces of different material?',

                                        'What is the persistence of coronavirus on copper, stainless steel and plastic?',

                                        'What is the persistence and stability of coronavirus on various substrates and sources?',

                                        'What is the persistence and stability of coronavirus in nasal discharge?',

                                        'What is the persistence and stability of coronavirus in sputum?',

                                        'What is the persistence and stability of coronavirus in urine and fecal matter?',

                                        'What is the persistence and stability of coronavirus in blood?',

                                        'What is the history of coronavirus?',

                                        'What is the process of coronavirus from shedding to infecting?',

                                        'What is the process of coronavirus from an infected person?',

                                        'What is the implementation of diagnostics for coronavirus?',

                                        'What products can be used to improve clinical processes for coronavirus?',

                                        'What is the disease models from humans and animal is similar to coronavirus infection, disease and transmission?',

                                        'What tools and studies exist to monitor phenotypic change and potential adaptation of coronavirus?',

                                        'What immune response and immunity are there to coronavirus?',

                                        'What is the effectiveness of movement control strategies to prevent secondary coronavirus transmission in health care and community settings?',

                                        'What is the effectiveness of personal protective equipment against coronavirus to reduce risk of transmission in health care and community settings?',

                                        'What is the role of the environment in the transmission of coronavirus?'])

    

widgets.interact(widget_interaction,

                 task_question= ['What is known about coronavirus transmission, incubation, and environmental stability?',

                                'What is the incubation period for coronavirus in humans?',

                                'How does the incubation periods for coronavirus varies with age and health?',

                                'How long does a human remain contagious with coronavirus?',

                                'How long does a human remain contagious with coronavirus after recovery?',

                                'What is the prevalence of coronavirus asymptomatic shedding?',

                                'What is the prevalence of coronavirus asymptomatic shedding in children?',

                                'What is the prevalence of coronavirus transmission in human?',

                                'What is the prevalence of coronavirus transmission in children?',

                                'How does seasonality affect coronavirus transmission?',

                                'What is the physical science behind coronavirus?',

                                'What is the charge distribution of coronavirus?',

                                'What is coronavirus adhesive ability to hydrophilic or phobic surfaces?',

                                'What is coronavirus environmental survivability?',

                                'What is coronavirus viral shedding ability?',

                                'What is the persistence of coronavirus on surfaces of different material?',

                                'What is the persistence of coronavirus on copper, stainless steel and plastic?',

                                'What is the persistence and stability of coronavirus on various substrates and sources?',

                                'What is the persistence and stability of coronavirus in nasal discharge?',

                                'What is the persistence and stability of coronavirus in sputum?',

                                'What is the persistence and stability of coronavirus in urine and fecal matter?',

                                'What is the persistence and stability of coronavirus in blood?',

                                'What is the history of coronavirus?',

                                'What is the process of coronavirus from shedding to infecting?',

                                'What is the process of coronavirus from an infected person?',

                                'What is the implementation of diagnostics for coronavirus?',

                                'What products can be used to improve clinical processes for coronavirus?',

                                'What is the disease models from humans and animal is similar to coronavirus infection, disease and transmission?',

                                'What tools and studies exist to monitor phenotypic change and potential adaptation of coronavirus?',

                                'What immune response and immunity are there to coronavirus?',

                                'What is the effectiveness of movement control strategies to prevent secondary coronavirus transmission in health care and community settings?',

                                'What is the effectiveness of personal protective equipment against coronavirus to reduce risk of transmission in health care and community settings?',

                                'What is the role of the environment in the transmission of coronavirus?'])