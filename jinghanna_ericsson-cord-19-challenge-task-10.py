!pip install cord-19-tools

!pip install spacy-langdetect

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import cotools as co

import gc



import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from IPython.core.display import display, HTML

from collections import defaultdict

import functools

import spacy

from spacy.matcher import PhraseMatcher

from spacy_langdetect import LanguageDetector

import en_core_sci_lg

import os

import re

import sys

import glob

  

from sklearn import cluster

from sklearn import metrics

from sklearn.manifold import TSNE



#from bert_serving.client import BertClient  # if using bert



from gensim.models import Word2Vec

from gensim.summarization.summarizer import summarize 

from gensim.summarization import keywords 



from tqdm.notebook import tqdm



from nltk.corpus import stopwords

from string import punctuation

from nltk.stem.lancaster import LancasterStemmer

from nltk.tokenize import sent_tokenize,word_tokenize

from nltk.probability import FreqDist

from nltk.cluster import KMeansClusterer

import nltk

from heapq import nlargest



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None
# Use the ipwidgets package to display a progress bar to give the user an indication

# of the progress of code execution for certain portions of the notebook.

# This function is intended to be used in the definition of a for loop to indicate

# how far execution has gotten through the object being iterated over in the for loop.

#

# Inputs: (sequence) - contains the for loop iteration (e.g., list or iterator)

#         (every)    - number of steps to display

#

# Outputs: displays the progress bar in the notebook

#          (yield record) - returns the current iteration object back to the calling for loop

#

def log_progress(sequence, every=None, size=None, name='Items'):

    

    '''

    Tracks the progress of for loop iteration

    

    Inputs: (sequence) - contains the for loop iteration

            (every) - the number of steps to display

    Outputs: (display) - a tracking bar that shows the progress of for loop iteration

    '''

    

    from ipywidgets import IntProgress, HTML, VBox

    from IPython.display import display



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

        

# Use the spaCy library natural language processing capabilities to clean an input text, 

# in string format, for punctuation, stop words, and lemmatization.

#

# Inputs (text) - a string to clean and lemmatize

#

# Outputs - a modified version of the input string that has been cleaned by removing punctuation, 

#           stop words, and pronouns, and has had the remaining words converted into corresponding lemmas

#         

def process_text(text):

    

    '''

    Cleans an input text in string format for punctuation, stopwords and lemmatization

    

    Inputs: (string) - input text

    Outputs: (string) - a cleaned output that removes punctuation, stopwords and converts words to lemma

    '''

    

    # Create a spaCy "Doc" object from the input text string.

    doc = nlp(text.lower())

    result = [] # list that will contain the lemmas for each word in the input string

    

    for token in doc:

        

        if token.text in nlp.Defaults.stop_words: #screen out stop words

            continue

        if token.is_punct:                        #screen out punctuations

            continue

        if token.lemma_ == '-PRON-':              #screen out pronouns

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
# Checks if string input can be interpreted as a date

#    

# Inputs: (string) - string to check whether it is a valid date

# Outputs: (bool) - True if string is a valid date; False otherwise

#

def is_date(string, fuzzy=False):

    

    '''

    Checks if string input can be interpreted as a date

    

    Inputs: (string) - to check if date interpretation is possible

    Outputs: (bool) - True if possible else False

    '''

    from dateutil.parser import parse

    

    try: 

        parse(string, fuzzy=fuzzy)

        return True



    except ValueError:

        return False
#print('Please input in the earliest date to filter the research paper (yyyy-mm-dd)!')

#filter_date = str(input())



# Modify this date per user requirements, or enter a non-valid date string to disable publication date filtering

filter_date = '2019-12-01'



# paper_id_list is a list of the IDs for all papers published after the specified date

# (or all papers if the date filtering is disabled).



if is_date(filter_date) == True:

    paper_id_list = metadata_information[metadata_information['publish_time'] >= filter_date].dropna(subset=['sha'])['sha'].tolist()

    

else:

    paper_id_list = metadata_information['sha'].tolist()
def create_library(list_of_folders, list_of_papers = paper_id_list):

    

    import json

    #Internal Library

    internal_library = []



    for i in log_progress(list_of_folders, every = 1):



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

    Converts JSON files into CSV based on various criteria

    

    Inputs: (list) - List_of_folders ; the names of the sub-directories in the library

            (DataFrame) - metadata ; metadata information provided in the library

            (string) - date ; filter criteria on publishing date information available in metadata

            (list) - list_of_papers ; containing the index information of papers published after date

    Outputs: (DataFrame) - dataframe containing on relevant papers

    '''    

    internal_library = create_library(list_of_folders = selected_folders, list_of_papers = paper_id_list)



    title_list = []          # list of paper titles

    abstract_list = []       # list of paper abstracts

    text_list = []           # list of paper full texts



    # Extracting title, abstract and text information for each paper

    # each_dataset is a list of dictionaries, where each dictionary corresponds to one paper

    for i in list(range(0, len(internal_library))):



        title_list.append(internal_library[i].get('metadata').get('title'))



        try:

            abstract_list.append(co.abstract(internal_library[i]))

        except:

            abstract_list.append('No Abstract')



        text_list.append(co.text(internal_library[i]))



    #Extracting Paper ID Information

    paper_id = [i.get('paper_id') for i in internal_library]   # list of the ID for each paper



    #Extracting the location and country that published the research paper

    primary_location_list = []      # list of the primary locations for the authors of each paper

    primary_country_list = []       # list of the primary countries for the authors of each paper



    # Extracting the primary location, and country for the authors of each paper

    # each_dataset is a list of dictionaries, where each dictionary corresponds to one paper



    # Extract list of metadata dictionaries for each paper

    internal_metadata = [i['metadata'] for i in internal_library]



    # individual_paper_metadata is the 'metadata' dictionary for one paper

    for individual_paper_metadata in internal_metadata:



        # Extract the list of author dictionaries for the current paper

        authors_information = individual_paper_metadata.get('authors')



        if len(authors_information) == 0:

            primary_location_list.append('None')

            primary_country_list.append('None')



        else:

            location = None

            country = None

            i = 1



            # Find the first author of the paper with valid data for "institution",

            # location, and country, extract this information, and add to

            # the respective lists for all the papers

            while location == None and i <= len(authors_information):



                if bool(authors_information[i-1].get('affiliation')) == True:



                    location = authors_information[i-1].get('affiliation').get('location').get('settlement')

                    country = authors_information[i-1].get('affiliation').get('location').get('country')



                i += 1



            primary_location_list.append(location)

            primary_country_list.append(country)

                

    #Loading all the extracted information into a DataFrame for merger

    index_df = pd.DataFrame(paper_id, columns =  ['paper_id'])



    geographical_df = pd.DataFrame(primary_location_list, columns = ['Location'])

    geographical_df['Country'] = primary_country_list



    paper_info_df = pd.DataFrame(title_list, columns = ['Title'])

    paper_info_df['Abstract'] = abstract_list

    paper_info_df['Text'] = text_list

    

    #This dataframe contains all the information extracted from the JSON files and converted into CSV.

    combined_df = pd.concat([index_df, geographical_df, paper_info_df], axis = 1)

    

    #Creating the merger between the metadata (45000+) and the research papers (33000+)

    part_1 = metadata[['sha', 'abstract', 'url', 'publish_time']]



    test_df = combined_df.merge(part_1, left_on = ['paper_id'], right_on = ['sha'], how = 'left')

    test_df.drop(['sha'], axis = 1,inplace = True)

    test_df = test_df[['paper_id', 'url', 'publish_time', 'Location', 'Country', 'Title', 'Abstract', 'abstract', 'Text']]

    

    #In the event where the JSON's abstract is null but there is an abstract in the metadata, it will be used as a substitute.

    test_df['Abstract'] = np.where(test_df['Abstract'] == '', test_df['abstract'], test_df['Abstract'])

    test_df.drop(['abstract'], axis = 1, inplace = True)

    

    gc.collect()

    

    return test_df
# Create a list of the datasets corresponding to each subdirectory over which we can iterate



# Define as a list the names of all the subdirectories in the 'Raw Data'

# directory where the dataset files are stored

selected_folders = ['comm_use_subset', 'noncomm_use_subset', 'custom_license', 'biorxiv_medrxiv']

test_df = data_creation(list_of_folders = selected_folders, metadata = metadata_information)
test_df.columns
test_df.to_csv(output_dir + 'Checkpoint_1.csv', index = False)
#Cleaning up after each section to save space

del paper_id_list

del metadata_information

del selected_folders



import gc

gc.collect()
def cleaning_dataset(dataset, columns_to_clean):

    

    # each_column is on of the defined text section columns from the DataFrame

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
## Cleaning up Dataset in the selected text columns

text_columns = ['Title', 'Abstract', 'Text']

test_df = cleaning_dataset(dataset = test_df, columns_to_clean = text_columns)
test_df['Abstract'].describe(include='all')
test_df.drop_duplicates(['Abstract', 'Text'], inplace = True)
test_df['Text'].describe(include = 'all')
test_df.to_csv(output_dir + 'Checkpoint_2.csv', index = False)
test_df.columns
test_df.dropna(subset = ['Text'], inplace = True)
def dimension_reduction(dataset, n = 3, n_components = 3, use_hashing_vectorizer = False):



    dataset = dataset.reset_index().drop(['index'], axis = 1)

    

    #Extracting Trigrams vectors for all 3885 documents

    if use_hashing_vectorizer == False:

    

        vectorizer=TfidfVectorizer(ngram_range=(n,n))

        vectorized_vectors=vectorizer.fit_transform(dataset['Text'].tolist())

        

    else:

        

        vectorizer=HashingVectorizer(ngram_range=(n,n))

        vectorized_vectors=vectorizer.fit_transform(dataset['Text'].tolist())



    #Dimensionality Reduction

    tsne_reduction = TSNE(n_components = 3, perplexity = 10, learning_rate = 100, random_state = 777)

    tsne_data = tsne_reduction.fit_transform(vectorized_vectors)



    #Converting components of T-SNE into dataframe

    tsne_df = pd.DataFrame(tsne_data, columns = [i for i in range(0, tsne_data.shape[1])])

    gc.collect()

    return tsne_df



def visualizing_dimensions(dataset):



    fig = plt.figure(1, figsize=(7, 5))

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



    ax.scatter(dataset[0], dataset[1], dataset[2], c=dataset[2], cmap='viridis', linewidth=0.5)



    ax.set_xlabel('Component A')

    ax.set_ylabel('Component B')

    ax.set_zlabel('Component C')



    plt.show()

    gc.collect()

    

def outlier_removals(dim_reduced_dataset, dataset, n_components = 3, number_std_dev = 2.5, verbose = 1):

    

    outlier_papers = []

    print('{a} standard deviation is being used to clean the dataset.'.format(a = number_std_dev))

    print()

    for i in range(0, n_components):

        

        upper = dim_reduced_dataset[i].mean() + number_std_dev*dim_reduced_dataset[i].std()

        lower = dim_reduced_dataset[i].mean() - number_std_dev*dim_reduced_dataset[i].std()



        outlier_df = dim_reduced_dataset[(dim_reduced_dataset[i] >= upper) | (dim_reduced_dataset[i] <= lower)]

        outlier_list = outlier_df.reset_index()['index'].tolist()

        

        outlier_papers += outlier_list

        

    outlier_papers = list(set(outlier_papers))

    

    if verbose == 1:

        print('There are {a} outlier papers identified.'.format(a = len(outlier_papers)))

        print()

        

    outlier_papers_df = dataset.iloc[outlier_papers,:]

    

    if verbose == 1:

        print('These are the texts that are determined as abnormal.')

        print()

        for i in outlier_papers_df['Text']:

            print(i)

            print()

    

    #remove outliers

    cleaned_df = dataset.drop(outlier_papers, axis = 0)

    cleaned_df.reset_index().drop(columns = ['index'], axis = 1)

    gc.collect()

    return cleaned_df



def full_cleaning_process(dataset, n = 3, n_components = 3, use_hashing_vectorizer = False, std_dev = 3, verbose = 1):

    

    starting_datashape = dataset.shape[0]

    dim_reduced_dataset = dimension_reduction(dataset, n = n, n_components = n_components, use_hashing_vectorizer = use_hashing_vectorizer)

    print('Before Cleaning Up -')

    visualizing_dimensions(dim_reduced_dataset)

    output_df = outlier_removals(dim_reduced_dataset, dataset, n_components = n_components, number_std_dev = std_dev, verbose = verbose)

    ending_datashape = output_df.shape[0]

    print('{a} rows were dropped in this cleaning process.'.format(a = starting_datashape - ending_datashape))

    print()

    print('After Cleaning Up -')

    visualizing_dimensions(dimension_reduction(output_df, n = 3, n_components = 3, use_hashing_vectorizer = False))

    gc.collect()

    return output_df
test_df = full_cleaning_process(test_df, std_dev = 2.5)
minimum_word_count = 150



test_df = test_df.reset_index().drop(['index'], axis = 1)

test_df['Text Word Count'] = [len(i.split()) for i in test_df['Text']]



dirty_list = []



for index, value in test_df.iterrows():

    

    if (value['Text Word Count'] <= minimum_word_count):

        dirty_list.append(index)

        

weird_papers_df = test_df.iloc[dirty_list,:]



for index, value in weird_papers_df.iterrows():

    print(value['Text Word Count'], value['Text'])

    print()
test_df = test_df.drop(dirty_list, axis = 0)

test_df = test_df.reset_index().drop(['index'], axis = 1)
test_df.to_csv(output_dir + 'Checkpoint_3.csv', index = False)
#Cleaning up after each section to save space

gc.collect()
#Scientific NLP library has been loaded to find articles that may or may not be in english. This acts as a final data

#clean-up, ensuring that the subsequent ML techniques are performed on as clean a dataset as possible.

nlp = en_core_sci_lg.load()

nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
#There exists a possibility that in this library of research papers, there exists non-English papers.



#Since some texts exceed the maximum length and may cause memory allocation, we will cut-off the text at the maximum length

#instead of changing it - to control computational resources



language_list = []

for i in log_progress(test_df['Text'], every = 1):

    

    if len(i) <= 1000000:

    

        doc = nlp(i)

        language_list.append(doc._.language)

        

    else:

        

        cut_off_index = i[:1000000].rfind('.')

        focus_i = i[:cut_off_index + 1]

        

        doc = nlp(focus_i)

        language_list.append(doc._.language)
#Storing information on the language detected of the paper. This score provides an indication

#of how much of the paper is in that particular language detected - helping us deal with papers with a combination of languages.

filtered_language_list = [i['language'].upper() for i in language_list]

test_df['Language'] = filtered_language_list



#Filtering out only research papers in English to perform topic modelling.

english_df = test_df[test_df['Language'] == 'EN']

print('There are {a} research papers in English out of {b} research papers.'.format(a = english_df.shape[0], b = test_df.shape[0]))
# drop off Language column, as all articles are English

english_df.drop(columns='Language', inplace=True)
english_df.shape
english_df.to_csv(output_dir + 'Checkpoint_4.csv', index = False)
#Cleaning up after each section to save space

gc.collect()
# Load libraries 

import os 

import numpy as np 

import pandas as pd 

import glob

import gc



from tqdm.notebook import tqdm



# Load word cloud function

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 



import spacy

from spacy.matcher import PhraseMatcher #import PhraseMatcher class

nlp = spacy.load('en_core_web_lg') # Language class with the English model 'en_core_web_lg' is loaded
def wordcloud_draw(text, color = 'white'):

    """

    Plots wordcloud of string text after removing stopwords

    """

    cleaned_word = " ".join([word for word in text.split()])

    wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color=color,

                      width=1000,

                      height=1000

                     ).generate(cleaned_word)

    plt.figure(1,figsize=(15, 15))

    plt.imshow(wordcloud)

    plt.axis('off')

    display(plt.show())
# Load checkpoint #4



df=pd.read_csv(output_dir + 'Checkpoint_4.csv')

df.shape
# clean up

del test_df

del english_df



gc.collect()
# Set list data directory

lists_data_dir = '/kaggle/input/task10lists/master'

os.chdir(lists_data_dir)

os.getcwd()
# Load list of therapeutics:

df_therapeutics = pd.read_csv('therapeutics.csv')

#df_therapeutics.shape

#df_therapeutics.head()

therapeutics_list = df_therapeutics.iloc[:, 0].tolist()

print(therapeutics_list)
# Load list of vaccines:

df_vaccine = pd.read_csv('vaccines.csv')

#df_vaccine.shape

#df_vaccine.head()

vaccine_list = df_vaccine.iloc[:, 0].tolist()

print(vaccine_list)
# Load list of animals:

df_animals = pd.read_csv('animals.csv')

#df_animals.shape

#df_animals.head()

animals_list = df_animals.iloc[:, 0].tolist()

print(animals_list)
# Load list of covid19:

df_covid19 = pd.read_csv('covid-19.csv')

#df_covid19.shape

#df_covid19.head()

covid19_list = df_covid19.iloc[:, 0].tolist()

print(covid19_list)
# Load list of drugs:

df_drugs = pd.read_csv('drugs.csv')

#df_drugs.shape

#df_drugs.head()

drugs_list = df_drugs.iloc[:, 0].tolist()

print(drugs_list)
# effectivenes

df_effectivenes = pd.read_csv('effectivenes.csv')

#df_effectivenes.shape

#df_effectivenes.head()

effectivenes_list = df_effectivenes.iloc[:, 0].tolist()

print(effectivenes_list)
# symptom

df_symptom = pd.read_csv('symptom.csv')

#df_symptom.shape

#df_symptom.head()

symptom_list = df_symptom.iloc[:, 0].tolist()

print(symptom_list)
# human

df_human = pd.read_csv('human.csv')

#df_human.shape

#df_human.head()

human_list = df_human.iloc[:, 0].tolist()

print(human_list)
# model

df_model = pd.read_csv('model.csv')

#df_model.shape

#df_model.head()

model_list = df_model.iloc[:, 0].tolist()

print(model_list)
# recipient

df_recipient = pd.read_csv('recipient.csv')

#df_recipient.shape

#df_recipient.head()

recipient_list = df_recipient.iloc[:, 0].tolist()

print(recipient_list)
# antiviral 

df_antiviral_agent  = pd.read_csv('antiviral.csv')

#df_antiviral_agent.shape

#df_antiviral_agent.head()

antiviral_agent_list = df_antiviral_agent.iloc[:, 0].tolist()

print(antiviral_agent_list)
# challenge

df_challenge = pd.read_csv('challenge.csv')

#df_challenge.shape

#df_challenge.head()

challenge_list = df_challenge.iloc[:, 0].tolist()

print(challenge_list)
# universal

df_universal = pd.read_csv('universal.csv')

#df_universal.shape

#df_universal.head()

universal_list = df_universal.iloc[:, 0].tolist()

print(universal_list)
# prioritize

df_prioritize = pd.read_csv('prioritize.csv')

#df_prioritize.shape

#df_prioritize.head()

prioritize_list = df_prioritize.iloc[:, 0].tolist()

print(prioritize_list)
# scarce

df_scarce = pd.read_csv('scarce.csv')

#df_scarce.shape

#df_scarce.head()

scarce_list = df_scarce.iloc[:, 0].tolist()

print(scarce_list)
# healthcare

df_healthcare = pd.read_csv('healthcare.csv')

#df_healthcare.shape

#df_healthcare.head()

healthcare_list = df_healthcare.iloc[:, 0].tolist()

print(healthcare_list)
# ppe

df_ppe = pd.read_csv('ppe.csv')

#df_ppe.shape

#df_ppe.head()

ppe_list = df_ppe.iloc[:, 0].tolist()

print(ppe_list)
# risk

df_risk = pd.read_csv('risk.csv')

#df_risk.shape

#df_risk.head()

risk_list = df_risk.iloc[:, 0].tolist()

print(risk_list)
# ADE

df_ADE = pd.read_csv('antibody.csv')

#df_ADE.shape

#df_ADE.head()

ADE_list = df_ADE.iloc[:, 0].tolist()

print(ADE_list)
# Use LOWER case

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
# Load list into NLP 



patterns = [nlp.make_doc(text) for text in therapeutics_list] 

matcher.add("1", None, *patterns)



patterns = [nlp.make_doc(text) for text in vaccine_list] 

matcher.add("2", None, *patterns)



patterns = [nlp.make_doc(text) for text in animals_list] 

matcher.add("3", None, *patterns)



patterns = [nlp.make_doc(text) for text in covid19_list] 

matcher.add("4", None, *patterns)



patterns = [nlp.make_doc(text) for text in drugs_list] 

matcher.add("5", None, *patterns)



patterns = [nlp.make_doc(text) for text in effectivenes_list]

matcher.add('6', None, *patterns)



patterns = [nlp.make_doc(text) for text in symptom_list]

matcher.add('7', None, *patterns)



patterns = [nlp.make_doc(text) for text in human_list]

matcher.add('8', None, *patterns)



patterns = [nlp.make_doc(text) for text in model_list]

matcher.add('9', None, *patterns)



patterns = [nlp.make_doc(text) for text in recipient_list]

matcher.add('10', None, *patterns)



patterns = [nlp.make_doc(text) for text in antiviral_agent_list]

matcher.add('11', None, *patterns)



patterns = [nlp.make_doc(text) for text in challenge_list]

matcher.add('12', None, *patterns)



patterns = [nlp.make_doc(text) for text in universal_list]

matcher.add('13', None, *patterns)



patterns = [nlp.make_doc(text) for text in prioritize_list]

matcher.add('14', None, *patterns)



patterns = [nlp.make_doc(text) for text in scarce_list]

matcher.add('15', None, *patterns)



patterns = [nlp.make_doc(text) for text in healthcare_list]

matcher.add('16', None, *patterns)



patterns = [nlp.make_doc(text) for text in ppe_list]

matcher.add('17', None, *patterns)



patterns = [nlp.make_doc(text) for text in risk_list]

matcher.add('18', None, *patterns)



patterns = [nlp.make_doc(text) for text in ADE_list]

matcher.add('19', None, *patterns)
df.describe()
import gc

gc.collect()
# add column to data to prepare ranking for given paper

df = df.assign(p_1=0,p_2=0,p_3=0,p_4=0,p_5=0,p_6=0,p_7=0,p_8=0,p_9=0,p_10=0,p_11=0,p_12=0,p_13=0,p_14=0,p_15=0,p_16=0,p_17=0,p_18=0,p_19=0)
df.head()
matching_rows = []

matching_paper_id = []



nlp.max_length = 206000000



pbar = tqdm()

pbar.reset(total=len(df)) 



for i, row in df[:].iterrows(): 

    pbar.update()

    if pd.isnull(row["Text"]):

        continue

    doc = nlp(row["Text"])

    matches = matcher(doc)

    if len(matches) > 0:

        matching_rows.append(i)

        matching_paper_id.append(row["paper_id"])

    for match_id, start, end in matches:

        # Get the string representation 

        string_id = nlp.vocab.strings[match_id]  #string_id shows matching location 

        span = doc[start:end]  

        df.loc[i, "p_" + string_id] = 1  
df.describe()
# Prepare ranking: drugs + convid 19 + effectiveness

df = df.assign(rank=df["p_5"] + df["p_4"] + df["p_6"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe(include='all')
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))
#print number of articles with all matching

df[df["rank"] == 3]
##word cloud matching drugs + effectiveness + sympton + covid 19 

text_world_cloud=""

for i, row in df[df["rank"] == 2].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 2

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: vacciness + ADE + covid 19  

df = df.assign(rank=df["p_2"] + df["p_19"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))
#print number of articles with all matching

df[df["rank"] == 3]
##word cloud matching vacciness + receipts + ade   

text_world_cloud=""

for i, row in df[df["rank"] == 3].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 3

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: vacciness + animals + human + model + covid 19 

df = df.assign(rank=df["p_2"] + df["p_3"] + df["p_8"] + df["p_9"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))

print(len(df[df["rank"] == 4]))

print(len(df[df["rank"] == 5]))
#print number of articles with all matching

df[df["rank"] == 5]
##word cloud matching vacciness + animals + human + model  

text_world_cloud=""

for i, row in df[df["rank"] == 4].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 5

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: therapeutics + symptons + antiviral agent + covid 19   

df = df.assign(rank=df["p_1"] + df["p_7"] + df["p_11"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))

print(len(df[df["rank"] == 4]))
#print number of articles with all matching

df[df["rank"] == 4]
##word cloud matching therapeutics + symptons + antiviral agent 

text_world_cloud=""

for i, row in df[df["rank"] == 4].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 4

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: therapeutics + prioritize + covid 19  

df = df.assign(rank=df["p_1"] + df["p_14"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))
#print number of articles with all matching

df[df["rank"] == 3]
##word cloud matching therapeutics + prioritize + scarce 

text_world_cloud=""

for i, row in df[df["rank"] == 3].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 4

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: vaccines + universal + covid 19 

df = df.assign(rank=df["p_13"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))
#print number of articles with all matching

df[df["rank"] == 2]
##word cloud matching vaccines + universal 

text_world_cloud=""

for i, row in df[df["rank"] == 2].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 2

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: animals + model + challenge + Covid 19 

df = df.assign(rank=df["p_3"] + df["p_9"] + df["p_12"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))

print(len(df[df["rank"] == 4]))
#print number of articles with all matching

df[df["rank"] == 4]
##word cloud matching animals + model + challenge  

text_world_cloud=""

for i, row in df[df["rank"] == 4].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 4

wordcloud_draw(text_world_cloud.lower()) 
# Prepare ranking: healthcare + ppe + covid 19 

df = df.assign(rank=df["p_16"] + df["p_17"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))
#print number of articles with all matching

df[df["rank"] == 3]
##word cloud matching healthcare + ppe 

text_world_cloud=""

for i, row in df[df["rank"] == 3].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 3

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: animals + model + challenge+ covid 19

df = df.assign(rank=df["p_3"] + df["p_9"] + df["p_12"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))

print(len(df[df["rank"] == 4]))
#print number of articles with all matching

df[df["rank"] == 4]
##word cloud matching animals + model + challenge 

text_world_cloud=""

for i, row in df[df["rank"] == 4].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 4

wordcloud_draw(text_world_cloud.lower())
# Prepare ranking: therapeutics + animals + model + covid 19

df = df.assign(rank=df["p_1"] + df["p_2"] + df["p_3"] + df["p_4"])
# This result should be added to the narrative, it's going to show the % per list matching

df.describe()
# Show total number of articles with matching list 

print(len(df[df["rank"] == 1]))

print(len(df[df["rank"] == 2]))

print(len(df[df["rank"] == 3]))

print(len(df[df["rank"] == 4]))
#print number of articles with all matching

df[df["rank"] == 4]

##word cloud matching animals + model + challenge 

text_world_cloud=""

for i, row in df[df["rank"] == 4].iterrows(): 

    text_world_cloud = text_world_cloud +" " + str (row["Title"])

#Visualization rank == 4

wordcloud_draw(text_world_cloud.lower())
#Cleaning up after each section to save space

gc.collect()
questions=['Effectiveness of drugs being developed and tried to treat COVID-19 patients.',

          'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.',

          'Exploration of use of best animal models and their predictive value for a human vaccine.',

          'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',

          'Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up.',

          'Efforts targeted at a universal coronavirus vaccine.',

          'Efforts to develop animal models and standardize challenge studies.',

          'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers.',

          'Approaches to evaluate risk for enhanced disease after vaccination.',

          'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models.']
def list_entity(path):  # read the list of entites

    df = pd.read_csv(path)

    lists=[]

    for row in df.iterrows():

        lists.append(row[1].values[0])

    return lists  



# search the dataframe for covid_19 the keywords

df_covid=df[functools.reduce(lambda a, b: a|b, (df['Text'].str.contains(s) for s in list_entity('../covid-19.csv')))]



 

len(df_covid)

main_ent={'main_1':'drugs','main_2':'antibody','main_3':'animals','main_4':'therapeutics'

          ,'main_5':'therapeutics','main_6':'vaccines','main_7':'animals','main_8': 'healthcare','main_9':'risk', 'main_10': 'vaccines'}



def entity_list(task,main_ent):

    path=f"../subtask_{task}"

    labels=defaultdict(list)

    paths_list=defaultdict(list)

    try:

        file_name=[]

        for file in os.listdir(path):

            if file.endswith(".csv"):

                file_name.append(file)

        main_name=f"main_{task}"

        labels['main'] = [os.path.splitext(name)[0] for name in file_name if os.path.splitext(name)[0]==main_ent[main_name]]

        labels['sides'] = [os.path.splitext(name)[0] for name in file_name if os.path.splitext(name)[0]!=main_ent[main_name]]

        for name in file_name:

            paths_list[os.path.splitext(name)[0]] = os.path.join(path,name)

    except (OSError, IOError) as e:

        print("The folder does not exist")

        

    return labels,paths_list

    



def list_entity(path):  # read the list of entites

    df = pd.read_csv(path)

    lists=[]

    for row in df.iterrows():

        lists.append(row[1].values[0])

    return lists  



def add_entities(labels,paths):  # add the list of entites

    for key in labels.keys():

        for val in labels[key]:

            patterns = [nlp(text) for text in list_entity(paths[val])] 

            matcher.add(val, None, *patterns) 



def remove_entities(labels):    # remove the list of entites

    for key in labels.keys():

        for val in labels[key]:

            matcher.remove(val) 

    

def check_existance(par,where_ind,doc):  # check if any entity exist on the par, output: give the dict with key equll to entity and value equll to 1 if it exist 

    dict_list=defaultdict(list)

    st=LancasterStemmer()

    for key in where_ind:

        for val in where_ind[key]:

            stem_par=[st.stem(word) for word in word_tokenize(par)]

            if st.stem(str(doc[val[0]:val[1]])) in stem_par:

                dict_list[key]=1

    return dict_list  



def prefrom_or(dict_list,labels):  

    exist=0

    for val in labels['sides']:

            if dict_list[val]==1:

                exist=1

    return exist  









def print_title_summary(titel_main,all_sent,publish_time,nlp_question):

    

    unique_titles = list(set(titel_main))

    scores=[]

    all_titles=[]

    all_text=[]

    out_put=pd.DataFrame(columns=['title','publish_time','text','scores'])

    

    for title in unique_titles:

        indices = [i for i, x in enumerate(titel_main) if x == title]

        text = []

        time=[]

        if indices: 

            for ind in indices:

                text.append(all_sent[ind])

                combined_text = ' '.join(text)

                time.append(publish_time[ind])

            

            score = nlp_question.similarity(nlp(combined_text))

            out_put=out_put.append({'title':title,'publish_time':time,'text':combined_text,'scores':score}, ignore_index=True)



 

    out_put=out_put.sort_values(by=['scores'],ascending=False)

    #for row in out_put.iterrows():

    #    display(HTML('<b>'+row[1]['title']+'</b> : <i>'+row[1]['text']+'</i>, ')) 

        

    return out_put

    #display(HTML('<b>'+title+'</b> : <i>'+combined_text+'</i>, '))     

            


def sent_vectorizer(sent, model):

    sent_vec =[]

    numw = 0

    for w in sent:

        try:

            if numw == 0:

                sent_vec = model[w]

            else:

                sent_vec = np.add(sent_vec, model[w])

            numw += 1

        except:

            pass

     

    return np.asarray(sent_vec) / numw





def sent2words(all_sent):

    sent_as_words = []

    #if all_sent is list():

    for s in all_sent:

        sent_as_words.append(s.split())

            

    #else:

    #    sent_as_words=all_sent.split()

    

    return sent_as_words





def sent_embedding(solution,all_sent):

    if solution== 'bert':

#            all_sent_list=  sent_tokenize(all_sent)      

        client = BertClient()

        embadded_vec = client.encode(all_sent) 

    else:



        sent_as_words = sent2words(all_sent)    

        model = Word2Vec(sent_as_words, min_count=1)

        embadded_vec=[]

        for sentence in sent_as_words:

            embadded_vec.append(sent_vectorizer(sentence, model))   



    return embadded_vec



def cluster_alg(NUM_CLUSTERS,embadded_vec):

    

    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)

    assigned_clusters = kclusterer.cluster(embadded_vec, assign_clusters=True)

    

    return assigned_clusters



def post_processing_bert(all_sent,ratio):

    NUM_CLUSTERS = 3

    embadded_vec = sent_embedding('bert',all_sent)  # embedding the sent based on solution

    assigned_clusters = cluster_alg(NUM_CLUSTERS,embadded_vec)

    #display(HTML('<b>'+'Highlights'+'</b>'))

    summary=cluster_summry(sent2words(all_sent),NUM_CLUSTERS,assigned_clusters,ratio)

    return summary
#spacy.load('en_core_web_sm')

def cluster_summry(sent_as_words,NUM_CLUSTERS,assigned_clusters,ratio):

    st=LancasterStemmer()



    summary_dataframe=pd.DataFrame(columns=['keyword','summary'])

    summary_par = []

    keys_max=[]

    for c in range(NUM_CLUSTERS):

        sent_in_cluster = []

        for j in range(len(sent_as_words)):

            if (assigned_clusters[j] == c):

          

                sent_in_cluster.append(' '.join(sent_as_words[j]))

        if len(' '.join(sent_in_cluster)) > ratio :    

            summary_par = summarize(' '.join(sent_in_cluster), word_count = ratio)

            

        else: 

            summary_par = ' '.join(sent_in_cluster)





        j=0

        keyword_intia = keywords(' '.join(sent_in_cluster)).split('\n')[0]

        while st.stem(keyword_intia)  in keys_max:

            j+=1 

            keyword_intia=keywords(' '.join(sent_in_cluster)).split('\n')[j]

            

        keys_max.append(st.stem(keyword_intia))

        

        

        

        summary_dataframe=summary_dataframe.append({'keyword':keyword_intia,'summary':summary_par}, ignore_index=True)

    

    return summary_dataframe

                  

import sys



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

    

import random    





def search_task(task, df_covid, main_ent,questions,q):



    all_sent = []

    titles = []

    publish_time=[]

    labels,paths = entity_list(task, main_ent)

    if labels:

        df_reduced=df_covid[functools.reduce(lambda a, b: b|a, (df_covid['Text'].str.contains(s) for s in list_entity(paths[main_ent[f"main_{task}"]])))]

        # to reduce runtime, randomly sample half of the data if rows > 2000

        #if len(df_reduced) > 200:

        #    df_reduced = df_reduced.sample(frac=0.1, replace=True, random_state=1)

        #print(df_reduced.shape)



        pbar = tqdm()

        pbar.reset(total=len(df_reduced)) 

        add_entities(labels,paths)   # add the entity to the exiting model

    

        for row in df_reduced.iterrows():  # go through body of each paper in dataframe

            pbar.update()

            doc = nlp(row[1]['Text'])         

            matches = matcher(doc)    # tag the predefined entities

            rule_id = []

            where_ind = defaultdict(list)

        

            for match_id, start, end in matches:

                rule = nlp.vocab.strings[match_id]

                nlp.max_length = 206000000

                rule_id.append(rule)  # get the unicode ID, i.e. 'COLOR'

                where_ind[rule].append((start,end))

            exist=0    

            for st in labels['sides']:

                if st in rule_id:

                    exist=1

                

            if labels['main'][0] in rule_id and exist:    # check the paper talk about at the first main topic

                for par in doc.sents:

                    dict_list = check_existance(par.text,where_ind,doc)

                

                    if dict_list[labels['main'][0]] == 1: 

                    

                        if prefrom_or(dict_list,labels)==1:      # check if the par has the combination of entities

                            all_sent.append(par.text)  # all senteces

                            titles.append(row[1]['Title'])

                            publish_time.append(row[1]['publish_time'])

                         

                            #display(HTML('<b>'+row[1]['title']+'</b> : <i>'+par.text+'</i>, '))  # print the related part of paper 

                

        #display(HTML('<b>'+questions[task-1]+'</b>' ))                 

        

        if all_sent:

            nlp_question = nlp(questions[task-1])

            score_papers=print_title_summary(titles,all_sent,publish_time,nlp_question)

            #print('csv out',task)

            score_papers.to_csv(output_dir + f"papers_subtask_{task}.csv")

            

            #summary=post_processing_bert(all_sent,100)

            #summary.to_csv(output_dir + f"summary_subtask_{task}.csv")

        else:

            #print('no all sent - task',task)

            score_papers=[]

            #remove_entities(labels)     # remove the existing entities

        

    #q.put((score_papers,task))

    #return(score_papers,task)

     

    res = 'Process worker ' + str(q)

    print("Worker finish job",q)

    q.put(res)

    return res
import multiprocessing as mp

import time



matcher = PhraseMatcher(nlp.vocab)  



def listener(q):

    """listens for messages on the q, writes to file. """

    print("start listener")

    while 1:

        m = q.get()

        print("listener get message: {}".format(m))

        if m == None:

            print("listener get kill message")

            break



def main():

    #must use Manager queue here, or will not work

    manager = mp.Manager()

    q = manager.Queue()    

    pool = mp.Pool(mp.cpu_count()+2)

    #put listener to work first

    watcher = pool.apply_async(listener, (q,))

    

    pbar = tqdm()

    pbar.reset(total=len(range(1,11))) 

    #fire off workers

    jobs = []



    for task in range(1,11):

        print('processing task', task)

        pbar.update()

        job=pool.apply_async(search_task,(task,df_covid, main_ent,questions,q) )

        jobs.append(job)



    # collect results from the workers through the pool result queue

    for job in jobs:

        #print('Get job -',job)

        job.get()

        

    #now we are done, kill the listener

    q.put(None)

    #q.task_done

    pool.close()

    pool.join()

    

if __name__ == "__main__":

   main()
#nlp = spacy.load("en")

#matcher = PhraseMatcher(nlp.vocab)  

#for task in range(1,11):

#    print('processing task', task)

#    search_task(task,df_covid, main_ent,questions)


def print_output(path):

    papers=pd.read_csv(path)

    df=papers.drop_duplicates()   

    df=df.dropna()

    df=df.drop(['Unnamed: 0'], axis=1)





    time=[]

    for i in range(len(df)):

        if df.iloc[i]['publish_time'][1:4]=='nan':

            time.append('nan')

        else:    

            time.append(df.loc[i]['publish_time'][2:12])



    df['publish time']=time

    df=df.drop(['publish_time'], axis=1)

    display(HTML(df.to_html()))
from PIL import Image
Image.open('/kaggle/input/ericsson-task10-img/T10_Task1.png')
if os.path.exists(output_dir + 'papers_subtask_1.csv'):

    print_output(output_dir + "papers_subtask_1.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task2.png')
if os.path.exists(output_dir + 'papers_subtask_2.csv'):

    print_output(output_dir + "papers_subtask_2.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task3.png')
if os.path.exists(output_dir + 'papers_subtask_3.csv'):

    print_output(output_dir + "papers_subtask_3.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task4.png')
if os.path.exists(output_dir + 'papers_subtask_4.csv'):

    print_output(output_dir + "papers_subtask_4.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task5.png')
if os.path.exists(output_dir + 'papers_subtask_5.csv'):

    print_output(output_dir + "papers_subtask_5.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task6.png')
if os.path.exists(output_dir + 'papers_subtask_6.csv'):

    print_output(output_dir + "papers_subtask_6.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task7.png')
if os.path.exists(output_dir + 'papers_subtask_7.csv'):

    print_output(output_dir + "papers_subtask_7.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task8.png')
if os.path.exists(output_dir + 'papers_subtask_8.csv'):

    print_output(output_dir + "papers_subtask_8.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task9.png')
if os.path.exists(output_dir + 'papers_subtask_9.csv'):

    print_output(output_dir + "papers_subtask_9.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 
Image.open('/kaggle/input/ericsson-task10-img/T10_Task10.png')
if os.path.exists(output_dir + 'papers_subtask_10.csv'):

    print_output(output_dir + "papers_subtask_10.csv")

else: 

    display(HTML('<b>'+"No related article is found"+'</b>' )) 