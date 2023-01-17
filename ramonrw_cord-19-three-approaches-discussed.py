!pip install wikipedia

!pip install rake-nltk

!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz

!pip install pytextrank

!pip install nltk

!pip install requests

!pip install spacy

!python -m spacy download en_core_web_lg



#Libraries

import os

import io

import re

import requests

import pandas as pd



#Read all paths, create a list and store it as csv.

paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #if os.path.splitext(filename)[-1] == ".json" and filename.endswith('.xml.json') is not True:

        if filename.endswith(".json") is True and filename.endswith('.xml.json') is not True:

            paths.append(os.path.join(dirname, filename))

        else:

            continue

        

pd.DataFrame(paths).to_csv("/kaggle/working/paths.csv")

path_list = pd.read_csv("/kaggle/working/paths.csv").iloc[:,1].to_list()
def retrieve_listofgenomes(url='https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup.cgi?taxid=10239&cmd=download',

                           viruspattern = 'corona'):

    

    '''

    This code extracts all information from an ncbi genome sample table. Especially virusname, 

    dates of creation & last update and NCBI Accession Code to be referred to in research.

    

    @param url is predefined for ncbi webpage - table with virus genomes. Another link will not work, 

            because the code is highly customized to this specific table.

    @param viruspattern is predefined to search for corona viruses. This can be adjusted to any other keyword.

    @output pandas table with the 11 attributes defined on the ncbi webpage filtered for the given viruspattern.

    '''

    

    import pandas as pd

    import re

    import requests

    import io

    

    s=requests.get(url).content

    column_names = []

    counter = 0

        

    for n, line in enumerate(io.StringIO(s.decode('utf-8'))):

        

        if n == 1:

            for chunknumber, chunk in enumerate(line.split('\t')):

                tmp = re.sub(r'[\")(,;\r\n\[\]]','',chunk)

                column_names.append(tmp)

        

        genome_information = dict.fromkeys(column_names, [])

    

    listofcoronaviruses = pd.DataFrame(genome_information)

    



    for n, line in enumerate(io.StringIO(s.decode('utf-8'))):

                

        if viruspattern in line:

            if len(line.split('\t')) < 11:

                pass

            else:

                for chunknumber, chunk in enumerate(line.split('\t')):

                    if chunknumber == 11:

                        pass

                    else:

                        listofcoronaviruses.loc[counter, list(genome_information.keys())[chunknumber]] = ' '.join(chunk.split())

                        

                counter += 1



    return listofcoronaviruses
listofcoronaviruses = retrieve_listofgenomes()

listofcoronaviruses.to_csv("finallistofcoronaviruses.csv")
# Function 1 - Extract Dict from Path



def preprocess_article(path):

    '''

    @param path from json document in kaggle challenge (as prepared in for statement * path_list = pd.read_csv("/kaggle/working/paths.csv").iloc[:,1].to_list() * )

    @output article_dict is a dictionary with paper_id, title, authors, abstract and textbody in one row to be further processed in pandas

    '''

    

    import json

    

    with open(path) as file:

        article_dict_load = json.load(file)

        

        #Add PaperID and Title to dict

        article_dict = {'paper_id': [article_dict_load['paper_id']],

                        'title': [article_dict_load['metadata']['title']]

                       }

        

        #Add Authors to dict

        authors_list = []

        for i in range(len(article_dict_load['metadata']['authors'])):

            try:

                authors_list.append(article_dict_load['metadata']['authors'][i]['first'][0] + '. ' + article_dict_load['metadata']['authors'][i]['last'])            

            except:

                authors_list.append(article_dict_load['metadata']['authors'][i]['last'])

        article_dict['authors'] = [', '.join(authors_list)]

        

        #Add Abstract to dict

        if len(article_dict_load['abstract']) == 1:

            article_dict['abstract'] = [article_dict_load['abstract'][0]['text'].replace('\"', '\'')]

        else:

            abstract_list = []

            for i in range(len(article_dict_load['abstract'])):

                abstract_list.append(article_dict_load['abstract'][i]['text'].replace('\"', '\''))

            article_dict['abstract'] = [' '.join(abstract_list)]

                    

        #Add textbody to dict

        if len(article_dict_load['body_text']) == 1:

            article_dict['textbody'] = [article_dict_load['body_text'][0]['text'].replace('\"', '\'')]

        else:

            textbody_list = []

            for i in range(len(article_dict_load['body_text'])):

                textbody_list.append(article_dict_load['body_text'][i]['text'].replace('\"', '\''))

            article_dict['textbody'] = [' \n '.join(textbody_list)]

    

    return article_dict

#Function 2 - Evaluate for Accession Number Appearance



def enhance_articledict(article_dict, finallistofcoronaviruses):

    '''

    @param article_dict is python dictionary resulting from a file processed with the function "preprocess_article"

    @param finallistofcoronaviruses is a list of NCBI Accession numbers related to a Coronavirus

    @output the dict is added with additional keys with the respective viruses referred to in the article based on the accession number. The value is 1.

    '''

    

    import re

    

    for corona in finallistofcoronaviruses:

        for chunk in article_dict['abstract'][0].split(' '): #Probably is splitting not necessary for this stage. Maybe potential optimization.

            if re.findall(corona, chunk):

                article_dict[corona] = [1]

                #print(corona)

        

        for chunk in article_dict['textbody'][0].split(' '): #Probably is splitting not necessary for this stage. Maybe potential optimization.

            if re.findall(corona, chunk):

                article_dict[corona] = [1]

                #print(corona)

      

    return article_dict
#Function 3 - Create enhanced Pandas Entry for Path



def retrieve_relevant_paper(path, path_finallistofcoronaviruses = "/kaggle/working/finallistofcoronaviruses.csv"):

    '''

    @param path takes an os-path referring to an json file containing a scientific article.

    @output pandas table to be appended 

    '''

    

    import pandas as pd

    

    finallistofcoronaviruses = pd.read_csv(path_finallistofcoronaviruses).iloc[:,2].to_list()

    

    article_dict = preprocess_article(path)

    

    article_dict_enhanced = enhance_articledict(article_dict, finallistofcoronaviruses)

    

    keys = list(article_dict.keys())

    keys.extend(finallistofcoronaviruses)

    

    df = pd.DataFrame({key: [] for key in keys})

    df = df.append(pd.DataFrame(article_dict_enhanced))

    

    return df
import pandas as pd

import time



path_list = pd.read_csv("/kaggle/working/paths.csv").iloc[:,1].to_list()



start_time = time.time()



for n, path in enumerate(path_list):

    if n == 0:

        df = retrieve_relevant_paper(path, path_finallistofcoronaviruses = "/kaggle/working/finallistofcoronaviruses.csv")

    else:

        df = df.append(retrieve_relevant_paper(path, path_finallistofcoronaviruses = "/kaggle/working/finallistofcoronaviruses.csv"))

        

print("--- %s seconds ---" % (time.time() - start_time))



df.to_csv('finaldataframe.csv')
def textscore_animal(spacy_nlp, inputtext, animal = 'animal', modus = 'default', max_length = 50000):

    '''

    The best way to utilize this module efficiently is to load the spacy model of your choice, preferrable 'en_core_web_lg' before and input it to the function as spacy_nlp. 

    This will decrease a single computation by over 90% as the loading of the model is the bottleneck and should be done outside your iteration.

    

    @param spacy_nlp accepts a string referring to a spacy model or an english spacy model with type() spacy.lang.en.Englisch

    @param inputtext accepts a string in whatever length

    @param animal accepts a string of length 1 word

    @param modus has 3 attributes

        'default' - for maximal value of score

        'list' - list of all word scores

    @param max_length the spacy model has a maximum character length it can sufficiently deal with (this is 100000). We set a max_length to deal with long text input by splitting it into chunks.

    @output animal_score according to modus 

    '''

    

    import spacy

    

    #Test Spacy model and load if necessary

    if isinstance(spacy_nlp, str):

        nlp = spacy.load(spacy_nlp)

    elif type(spacy_nlp) == spacy.lang.en.English:

        nlp = spacy_nlp

    else:

        print("Variable spacy_nlp has the wrong format. It is neither a string or a spacy.lang.en.English model.")

    

    #Initialize Scores List

    scores = []

     

    #Create Scores comparing vocab lists.

    if isinstance(inputtext, str) and isinstance(animal, str) and len(animal.split()) == 1:

        animal_vocab = nlp.vocab[animal]

        if len(inputtext) > max_length:

            for split in inputtext.split('.'):

                for token in nlp(split):

                    if nlp.vocab[token.text].vector[0] != 0.0 and (token.tag_ == "NN" or token.tag_ == "NNP"):

                        scores.append(nlp.vocab[token.text].similarity(animal_vocab))

        else:

            for token in nlp(inputtext):

                if nlp.vocab[token.text].vector[0] != 0.0 and (token.tag_ == "NN" or token.tag_ == "NNP"):

                    scores.append(nlp.vocab[token.text].similarity(animal_vocab))

    else:

        print("Variable inputtext or animal has the wrong format. It is not a string.")

    

    #Create output according to modus.

    if scores == []:

        print('No Scores...!')

        animal_score = 0

    elif modus == 'default':

        animal_score = max(scores)

    elif modus == 'list':

        animal_score = scores

    #'average' could also be a viable modus with giving the average of the highest 10 values. Variations of it are also possible

    else:

        print('Variable modus has the wrong format. It only accepts a string with default or list.')

        

    return animal_score
import pandas as pd



coronaviruses = pd.read_csv("finallistofcoronaviruses.csv")

coronaviruses = pd.DataFrame(coronaviruses.loc[coronaviruses['Host'].str.contains('human, vertebrates')]['Accession'].reset_index(drop = True))

coronaviruses['scores'] = None

document_df = pd.read_csv('finaldataframe.csv')

scores = []

for i, corona in enumerate(coronaviruses['Accession']):

    coronaviruses.loc[i, 'scores'] = sum(document_df[corona] == 1)



covid_candidate = coronaviruses.loc[coronaviruses['scores'].values.argmax(), 'Accession']

print(covid_candidate) #Most mentioned



coronaviruses = pd.read_csv("finallistofcoronaviruses.csv")

covid_candidate_no2 = coronaviruses.loc[pd.to_datetime(coronaviruses['Date updated']).values.argmax(),'Accession']

print(covid_candidate_no2) #Most recent updated



column_names1 = ['title', 'textbody', 'abstract', covid_candidate]

document_df1 = document_df[column_names1]

document_df1 = document_df1[document_df1[covid_candidate] == 1].reset_index(drop=True)



column_names2 = ['title', 'textbody', 'abstract', covid_candidate_no2]

document_df2 = document_df[column_names2]

document_df2 = document_df2[document_df2[covid_candidate_no2] == 1].reset_index(drop=True)
from IPython.display import display

import spacy



livestock = ['cattle', 'sheep', 'pig']

spacy_nlp = spacy.load('en_core_web_lg')



for animal in livestock:

    document_df1[animal] = None

    for n, document in enumerate(document_df1['textbody']):

        document_df1.loc[n, animal] = textscore_animal(spacy_nlp, document, animal = animal, modus = 'default')



for animal in livestock:

    document_df2[animal] = None

    for n, document in enumerate(document_df1['textbody']):

        document_df2.loc[n, animal] = textscore_animal(spacy_nlp, document, animal = animal, modus = 'default')



display(document_df1)

display(document_df2)
coronaviruses = pd.read_csv("finallistofcoronaviruses.csv")

coronaviruses1 = coronaviruses.loc[coronaviruses['Accession'] == covid_candidate].loc[:, 'Genome'].reset_index(drop=True)

coronaviruses2 = coronaviruses.loc[coronaviruses['Accession'] == covid_candidate_no2].loc[:, 'Genome'].reset_index(drop=True)

livestock = ['cattle', 'sheep', 'pig']

for stock in livestock:

    print('''

    {}\n

    A Preliminary Analysis on {} as a representant of Livestock and Coronaviruses yielded the following result:

    The most recent updated virus is the \"{}\", we identified {} references in the scientific literature.

    On a threshold of 66% of similarity to {}, we counted {} articles mentioning {}.

    This is in about {:2f}% of all articles referencing this coronavirus.\n

    The most refferenced virus is the \"{}\", we identified {} references in the scientific literature.

    On a threshold of 66% of similarity to {}, we counted {} articles mentioning {}.

    This is in about {:2f}% of all articles referencing this coronavirus.\n\n

    '''.format(stock, 

                stock, 

                coronaviruses2[0], len(document_df2[stock]),

                stock, sum(document_df2[stock] > 0.66), stock,

                sum(document_df2[stock] > 0.66) / len(document_df2[stock]),

                coronaviruses1[0], len(document_df1[stock]),

                stock, sum(document_df1[stock] > 0.66), stock,

                sum(document_df1[stock] > 0.66) / len(document_df1[stock])

              ))
# Add all import and preprocessor definitions



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from collections import Counter

from nltk.stem import WordNetLemmatizer # used for preprocessing



#!pip install num2words

#from num2words import num2words



import nltk

import os

import string

import numpy as np

import copy

import pandas as pd

import pickle

import re

import math

import time

import datetime



from csv import writer

import json

from collections import OrderedDict



from multiprocessing import Process, Value, Lock, Manager, Pool

from joblib import Parallel, delayed, parallel_backend

from math import modf



from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_new



import warnings

warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings
print("Initializing definitions", end="", flush=True)

def convert_lower_case(data):

    return np.char.lower(data)





# remove urls, handles, and the hashtag from hashtags (taken from https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression)

def remove_urls(text):

    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

    return new_text

# make all text lowercase

def text_lowercase(text):

    return text.lower()

# remove numbers

def remove_numbers(text):

    result = re.sub(r'\d+', '', text)

    return result

# remove punctuation

def remove_punctuation(text):

    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)

# tokenize

def tokenize(text):

    text = word_tokenize(text)

    return text

# remove stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):

    text = [i for i in text if not i in stop_words]

    return text

# lemmatize

lemmatizer = WordNetLemmatizer()

def lemmatize(text):

    text = [lemmatizer.lemmatize(token) for token in text]

    return text



#instantiate stemmer

stemmer = PorterStemmer()

def stemming(text):

    text = [stemmer.stem(token) for token in text]

    return text



def preprocess(text):

    text = text_lowercase(text)

    text = remove_urls(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = tokenize(text)

    text = remove_stopwords(text)

    text = stemming(text)

    text = remove_stopwords(text)

    #text = lemmatize(text)

    return text





nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('punkt')



print("...[DONE]")
print("Reading all file names", end="", flush=True)

alpha = 0.3



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



N = 0

files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.lower().endswith((".json")):

            files.append(os.path.join(dirname, filename))

            N = N + 1

#for x in folder:

#    print(x)

# Any results you write to the current directory are saved as output.



print("...[DONE]")



print("Number of files to calculate: ", end="", flush=True)

print(N)
print("extra definitions", end="", flush=True)

def extract_values(obj, key):

    """Pull all values of specified key from nested JSON."""

    arr = []



    def extract(obj, arr, key):

        """Recursively search for values of key in JSON tree."""

        if isinstance(obj, dict):

            for k, v in obj.items():

                if isinstance(v, (dict, list)):

                    extract(v, arr, key)

                elif k == key:

                    arr.append(v)

        elif isinstance(obj, list):

            for item in obj:

                extract(item, arr, key)

        return arr



    results = extract(obj, arr, key)

    return results



##DEFINES

def print_doc(id):

    print(dataset[id])

    file = open(dataset[id][0], 'r', encoding='cp1250')

    text = file.read().strip()

    file.close()

    print(text)



def doc_freq(word):

    c = 0

    try:

        c = DF[word]

    except:

        pass

    return c



def matching_score(k, tokens, tf_idf):

    query_weights = {}



    for key in tf_idf:

        

        if key[1] in tokens:

            try:

                query_weights[key[0]] += tf_idf[key]

            except:

                query_weights[key[0]] = tf_idf[key]

    

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    

    l = []

    

    for i in query_weights[:k]:

        l.append(i[1])

    

    if not l:

        l.append(0)

        

    return l



def cosine_sim(a, b):

    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cos_sim

    

def gen_vector(tokens,total_vocab):



    Q = np.zeros((len(total_vocab)))

    

    counter = Counter(tokens)

    words_count = len(tokens)



    query_weights = {}

    

    for token in np.unique(tokens):

        

        tf = counter[token]/words_count

        df = doc_freq(token)

        idf = math.log((N+1)/(df+1))



        try:

            ind = total_vocab.index(token)

            Q[ind] = tf*idf

        except:

            pass

    return Q



def cosine_similarity(k, tokens,total_vocab, D, dataset):

    d_cosines = []

    

    query_vector = gen_vector(tokens,total_vocab)

    

    for d in D:

        score = cosine_sim(query_vector, d)

        d_cosines.append(cosine_sim(query_vector, d))

        

    return d_cosines



def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):

    """

    Call in a loop to create terminal progress bar

    @params:

        iteration   - Required  : current iteration (Int)

        total       - Required  : total iterations (Int)

        prefix      - Optional  : prefix string (Str)

        suffix      - Optional  : suffix string (Str)

        decimals    - Optional  : positive number of decimals in percent complete (Int)

        length      - Optional  : character length of bar (Int)

        fill        - Optional  : bar fill character (Str)

    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))

    filledLength = int(length * iteration // total)

    bar = fill * filledLength + '-' * (length - filledLength)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

    # Print New Line on Complete

    if iteration == total:

        print()



# function to add to JSON 

def write_json(data, filename='data.json'): 

    with open(filename,'w') as f: 

        json.dump(data, f, indent=4) 

#END OF NEW DEFINES



def write_csv(list_of_elem, filename='data.csv'):

    # Open file in append mode

    with open(filename, 'a+', newline='') as write_obj:

        # Create a writer object from csv module

        csv_writer = writer(write_obj)

        # Add contents of list as last row in the csv file

        csv_writer.writerow(list_of_elem)





print("...[DONE]")
print("Testwords initializing", end="", flush=True)



# Corona

lst_corona = ["Corona", "corona", "corona virus", "coronavirus", "corona viruses", "coronaviruses", "Coronaviridae", "coronaviridae", "COVID-19", "Covid-19", "covid-19", "COVID", "COV", "SARS"]



# Main Task

lst_genetics = ["genetics"]

lst_origin = ["origin", "member", "family"]

lst_evolution = ["evolution", "development", "develops", "developed"]

lst_task = [lst_genetics, lst_origin, lst_evolution]



# Sub task 1 - Real-time tracking ...

lst_subtask_1_genome = ["Genome", "genome"]

lst_subtask_1_dissemination = ["dissemination", "Dissemination", "propagation", "Propagation", "spread", "Spread", "spreading", "Spreading"]

lst_subtask_1_treatment = ["treatment", "Treatment", "diagnostic", "Diagnostic", "diagnostics", "Diagnostics", "therapeutics", "Therapeutics"]

lst_subtask_1_variation = ["Difference" , "in contrast", "variation", "deviation", "shows mutations", "enrichment", "similarities"]

lst_subtask_1_reference = ["Accession number", "reference", "sample", "identification of"]

lst_subtask_1_known = ["Known", "already published", "already reported"]

lst_subtask_1 = [lst_subtask_1_genome, lst_subtask_1_dissemination, lst_subtask_1_treatment, lst_subtask_1_variation, lst_subtask_1_reference, lst_subtask_1_known, lst_corona]



# Sub task 2 - Access to geographic ...

lst_subtask_2_1 = []

lst_subtask_2_2 = []

lst_subtask_2_3 = []

lst_subtask_2_4 = []

lst_subtask_2_5 = []

lst_subtask_2 = [lst_subtask_2_1, lst_subtask_2_2, lst_subtask_2_3, lst_subtask_2_4, lst_subtask_2_5]

# Sub task 3 - Evidence that livestock ...



# Sub sub task 3-1

lst_subtask_3_1_livestock = ["farm" , "wildlife", "wild animal", "undomesticated", "livestock"]

lst_subtask_3_1_area = ["Southeast-Asia"]

lst_subtask_3_1_control = ["surveil", "control", "screen", "check", "monitor", "examine"]

lst_subtask_3_1 = [lst_subtask_3_1_livestock, lst_subtask_3_1_area, lst_subtask_3_1_control, lst_corona]



# Sub sub task 3-2

lst_subtask_3_2_livestock = ["farm" , "wildlife", "wild animal", "undomesticated", "livestock"]

lst_subtask_3_2_area = ["Southeast-Asia"]

lst_subtask_3_2_control = ["surveil", "control", "screen", "check", "monitor", "examine"]

lst_subtask_3_2 = [lst_subtask_3_2_livestock, lst_subtask_3_2_area, lst_subtask_3_2_control, lst_corona]



# Sub sub task 3-3

lst_subtask_3_3_1_host = ["host" , "organism", "human"]

lst_subtask_3_3_2_infection = ["infection", "disease", "respiratory syndrom"]

lst_subtask_3_3_3_lab = ["experimental", "laboratory", "under conditions"]

lst_subtask_3 = [lst_subtask_3_3_1_host, lst_subtask_3_3_2_infection, lst_subtask_3_3_3_lab, lst_corona]



# Sub task 4

lst_subtask_4_host = ["animal", "host", "hosts", "Host", "Hosts", "human", "Human", "Humans", "humans", "CoV-Host", "organism"]

lst_subtask_4_transmission = ["pathogen", "spill-over", "intraspecies", "interaction", "host-shift", "spread", "evolution", "transmission", "infection"]

lst_subtask_4_evidence = ["evidence", "proof", "association", "connection", "associated"]

lst_subtask_4 = [lst_subtask_4_host, lst_subtask_4_transmission, lst_subtask_4_evidence, lst_corona]



# Sub task 5

lst_subtask_5_1 = []

lst_subtask_5_2 = []

lst_subtask_5_3 = []

lst_subtask_5_4 = []

lst_subtask_5 = []



# Sub task 6

lst_subtask_6_1 = []

lst_subtask_6_2 = []

lst_subtask_6_3 = []

lst_subtask_6_4 = []

lst_subtask_6_5 = []

lst_subtask_6 = []



testwords = []

testwords.append(["lst_corona",preprocess(" ".join(lst_corona))])

testwords.append(["lst_genetics",preprocess(" ".join(lst_genetics))])

testwords.append(["lst_origin",preprocess(" ".join(lst_origin))])

testwords.append(["lst_evolution",preprocess(" ".join(lst_evolution))])

testwords.append(["lst_subtask_1_genome",preprocess(" ".join(lst_subtask_1_genome))])

testwords.append(["lst_subtask_1_dissemination",preprocess(" ".join(lst_subtask_1_dissemination))])

testwords.append(["lst_subtask_1_treatment",preprocess(" ".join(lst_subtask_1_treatment))])

testwords.append(["lst_subtask_1_variation",preprocess(" ".join(lst_subtask_1_variation))])

testwords.append(["lst_subtask_1_reference",preprocess(" ".join(lst_subtask_1_reference))])

testwords.append(["lst_subtask_1_known",preprocess(" ".join(lst_subtask_1_known))])

testwords.append(["lst_subtask_2_1",preprocess(" ".join(lst_subtask_2_1))])

testwords.append(["lst_subtask_2_2",preprocess(" ".join(lst_subtask_2_2))])

testwords.append(["lst_subtask_2_3",preprocess(" ".join(lst_subtask_2_3))])

testwords.append(["lst_subtask_2_4",preprocess(" ".join(lst_subtask_2_4))])

testwords.append(["lst_subtask_2_5",preprocess(" ".join(lst_subtask_2_5))])

testwords.append(["lst_subtask_3_1_livestock",preprocess(" ".join(lst_subtask_3_1_livestock))])

testwords.append(["lst_subtask_3_1_area",preprocess(" ".join(lst_subtask_3_1_area))])

testwords.append(["lst_subtask_3_1_control",preprocess(" ".join(lst_subtask_3_1_control))])

testwords.append(["lst_subtask_3_2_livestock",preprocess(" ".join(lst_subtask_3_2_livestock))])

testwords.append(["lst_subtask_3_2_area",preprocess(" ".join(lst_subtask_3_2_area))])

testwords.append(["lst_subtask_3_2_control",preprocess(" ".join(lst_subtask_3_2_control))])

testwords.append(["lst_subtask_3_3_1_host",preprocess(" ".join(lst_subtask_3_3_1_host))])

testwords.append(["lst_subtask_3_3_2_infection",preprocess(" ".join(lst_subtask_3_3_2_infection))])

testwords.append(["lst_subtask_3_3_3_lab",preprocess(" ".join(lst_subtask_3_3_3_lab))])

testwords.append(["lst_subtask_4_host",preprocess(" ".join(lst_subtask_4_host))])

testwords.append(["lst_subtask_4_transmission",preprocess(" ".join(lst_subtask_4_transmission))])

testwords.append(["lst_subtask_4_evidence",preprocess(" ".join(lst_subtask_4_evidence))])

testwords.append(["lst_subtask_5_1",preprocess(" ".join(lst_subtask_5_1))])

testwords.append(["lst_subtask_5_2",preprocess(" ".join(lst_subtask_5_2))])

testwords.append(["lst_subtask_5_3",preprocess(" ".join(lst_subtask_5_3))])

testwords.append(["lst_subtask_5_4",preprocess(" ".join(lst_subtask_5_4))])

testwords.append(["lst_subtask_6_1",preprocess(" ".join(lst_subtask_6_1))])

testwords.append(["lst_subtask_6_2",preprocess(" ".join(lst_subtask_6_2))])

testwords.append(["lst_subtask_6_3",preprocess(" ".join(lst_subtask_6_3))])

testwords.append(["lst_subtask_6_4",preprocess(" ".join(lst_subtask_6_4))])

testwords.append(["lst_subtask_6_5",preprocess(" ".join(lst_subtask_6_5))])



testwords_string = []

for i in range(len(testwords)):

    try:

        testwords_string.append([testwords[i][0]," ".join(testwords[i][1])])

    except:

        testwords_string.append([testwords[i][0]," "])

        

print("...[DONE]")

## END OF TESTWORDS
print("Cleaning and initializing output directory", end="", flush=True)



method = "CSV"



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        if filename.lower().endswith((".json")):

            os.remove(filename)

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        if filename.lower().endswith((".csv")):

            os.remove(filename)

            

if method == "JSON":          

    for x in range(len(testwords)):

        json_name = testwords[x][0]+".json"

        with open(json_name, 'w+') as json_file: 

            data = []

            data.append(["",""])

        write_json(data,json_name)

    for x in range(len(testwords)):

        json_name = testwords[x][0]+"_cosine.json"

        with open(json_name, 'w+') as json_file: 

            data = []

            data.append(["",""])

        write_json(data,json_name)

elif method == "CSV":

    for x in range(len(testwords)):

        csv_name = testwords[x][0]+".csv"

        data = ["File Title","Score"]

        write_csv(data,csv_name)

    for x in range(len(testwords)):

        csv_name = testwords[x][0]+"_cosine.csv"

        data = ["File Title","Score"]

        write_csv(data,csv_name)

    

print("...[DONE]")
class CountProgress(object):

    def __init__(self, manager, initval=0, time = 0):

        self.val = manager.Value('i', initval)

        self.lock = manager.Lock()



    def increment(self):

        with self.lock:

            self.val.value += 1



    def value(self):

        with self.lock:

            return self.val.value



class CountTime(object):

    def __init__(self, manager, initval=0):

        self.val = manager.Value('d', initval)

        self.lock = manager.Lock()

        

    def totaltime(self,t):

        with self.lock:

            self.val.value += t

            

    def value(self):

        with self.lock:

            return self.val.value
DEBUG = -1



def process_file(file,countprogress,countime,paralleljobs):

    if DEBUG == 1:

        start_time = time.time()

    elif DEBUG == 2:

        start_time = time.time()

    with open(file) as data_file:

        file_start_time = time.time()

        dataset = []

        data = json.load(data_file, object_pairs_hook=OrderedDict)

        dataset.append([file,extract_values(data, 'text')])

        if DEBUG == 1:

            print("--- %s seconds to open file---" % (time.time() - start_time))

            

        processed_text = []

        processed_text_string = []



        try:

            text = " ".join(dataset[0][1])

            processed_text.append([dataset[0][0],preprocess(text)])

        except:

            processed_text.append([dataset[0][0],[" "]])

        

        if DEBUG == 1:

            print("--- %s seconds to process---" % (time.time() - start_time))

            

        try:

            processed_text_string.append(" ".join(processed_text[0][1]))

        except:

            processed_text_string.append(" ")

        #end



        if DEBUG == 1:

            print("--- %s seconds to process text string---" % (time.time() - start_time))

        DF = []



        i=0

        DF.append({})

        tokens = processed_text[i][1]

        for w in tokens:

            try:

                DF[i][w].add(i)

            except:

                DF[i][w] = {i}



        for w in DF[i]:

            DF[i][w] = len(DF[i][w])



        ##END OF DF

        if DEBUG == 1:

            print("--- %s seconds to END OF DF---" % (time.time() - start_time))

        total_vocab_size = []

        total_vocab = []



        total_vocab_size.append(len(DF[i]))

        #print(total_vocab_size[i])

        total_vocab.append([x for x in DF[i]])



        ##END OF VOCAB SIZE

        if DEBUG == 1:

            print("--- %s seconds to END OF VOCAB---" % (time.time() - start_time))

        doc = 0



        tf_idf = []



        tf_idf.append({})

        tokens = processed_text[i][1]



        counter = Counter(tokens)

        words_count = len(tokens)



        for token in np.unique(tokens):



            tf = counter[token]/words_count

            df = doc_freq(token)

            idf = np.log((N+1)/(df+1))



            tf_idf[i][doc, token] = tf*idf



        doc += 1



        ## END OF TF

        if DEBUG == 1:

            print("--- %s seconds to END OF TF/IDF---" % (time.time() - start_time))

        for j in tf_idf[i]:

            tf_idf[i][j] *= alpha



        ## END OF MERGE

        if DEBUG == 1:

            print("--- %s seconds to END OF MERGE---" % (time.time() - start_time))

        D = []

        #     for i in out:

        #         print(i, dataset[i][0])

        D.append(np.zeros((1, total_vocab_size[i])))

        for j in tf_idf[i]:

            try:

                ind = total_vocab[i].index(j[1])

                D[i][j[0]][ind] = tf_idf[i][j]

            except:

                pass

        if DEBUG == 1:

            print("--- %s seconds to END OF D---" % (time.time() - start_time))

            

        for x in range(len(testwords)):

            results = []

            score = matching_score(1, testwords[x][1], tf_idf[i])

            results.append([dataset[i][0],score[0]])

            if method == "CSV":

                csv_name = testwords[x][0]+".csv"

                data = [dataset[i][0],score[0]]

                write_csv(data,csv_name)

            elif method == "JSON":  

                json_name = testwords[x][0]+".json"

                with open(json_name) as json_file: 

                    data = json.load(json_file) 

                    temp = data[0] 

                    temp.append(results) 

                write_json(data,json_name)

            elif method == "screen":

                data = [dataset[i][0],score[0]]

                print(data)

            elif method == "none":

                data = [dataset[i][0],score[0]]

                

            if DEBUG == 1:

                print("--- %s seconds to END OF TF/IDF SCORING---" % (time.time() - start_time))



            Q = []

            Q.append(cosine_similarity(1, testwords[x][1],total_vocab[i], D[i], dataset[i]))



            if DEBUG == 1:

                print("--- %s seconds to END OF Q---" % (time.time() - start_time))

                

            results = []

            score = Q[i][i]

            if math.isnan(score):

                score = 0

            results.append([dataset[i][0],score])

            if method == "CSV":

                csv_name = testwords[x][0]+"_cosine.csv"

                data = [dataset[i][0],score]

                write_csv(data,csv_name)

            elif method == "JSON":

                with open(testwords[x][0]+"_cosine.json", 'w') as json_file:

                    data = json.load(json_file) 

                    temp = data[0] 

                    temp.append(results) 

                    json.dump(results, json_file)

            elif method == "screen":

                data = [dataset[i][0],score]

                print(data)

            elif method == "none":

                data = [dataset[i][0],score]



            if DEBUG == 1:

                print("--- %s seconds to END OF COSINE SIMMILARITY---" % (time.time() - start_time))

        if DEBUG == 1:

            print("--- %s seconds to END OF FILE---" % (time.time() - start_time))

        elif DEBUG == 2:

            print("--- %s seconds to END OF FILE---" % (time.time() - start_time))

            

            

    countprogress.increment()

    elapsed_time = time.time() - file_start_time

    countime.totaltime(elapsed_time)

    

    totaltime = countime.value()

    cprogress = countprogress.value()

    remaining = ((totaltime/cprogress) * (N - cprogress))

    

    if DEBUG == 0:

        totaltime = totaltime/paralleljobs

        remaining = remaining/paralleljobs

    

    totaltime = str(datetime.timedelta(seconds=totaltime))

    remaining = str(datetime.timedelta(seconds=remaining))

    

    suf = 'Complete ['+str(cprogress)+'/'+str(N)+'] File processed in: ' + str(round(elapsed_time,3)) 

    suf += 's ETTF -> ' + remaining + ' elapsed -> ' + totaltime

    printProgressBar(cprogress, N, prefix = 'Progress:', suffix = suf, length = 50)
print("Starting main loop...")

manager = Manager()

manager2 = Manager()



result = manager.dict()

result2 = manager.dict()



countprogress = CountProgress(manager, 0)

countime = CountTime(manager2, 0)



paralleljobs = 4



if DEBUG == 1:

    for file in files[:1]:

        process_file(file,countprogress,countime,paralleljobs)

elif DEBUG == 0:

    Parallel(n_jobs=paralleljobs, prefer="threads")(delayed(process_file)(file,countprogress,countime,paralleljobs) for file in files)

else:

    for file in files:

        process_file(file,countprogress,countime,paralleljobs)

        

        

        ## END OF COSINE SIMILARITY

print()

print("...[DONE]")
def get_number_of_elements_nested_list(list_of_keyword_lists):

    counter = 0

    for lst in list_of_keyword_lists:

        counter += len(lst)

    return counter



def evaluate_text_via_list_of_list_of_keywords(text, list_of_keyword_lists):

    num_keyword_lists = len(list_of_keyword_lists)

    len_text = len(text)

    #num_total_keyword_hits = 0

    arr_keyword_list_hits = np.zeros(num_keyword_lists)

    for num_word in range(len_text):

        for num_keyword_list in range(num_keyword_lists):

            if text[num_word] in list_of_keyword_lists[num_keyword_list]:

                #num_total_keyword_hits += 1

                arr_keyword_list_hits[num_keyword_list] += 1

    return arr_keyword_list_hits



def evaluate_journal_by_keywords(df_paper, list_of_keyword_lists):

    """

    df_paper                 pd.DataFrame with Covid 19 papers

    list_of_keyword_lists    Nested list. Contains multiple lists with keywords. Each list is a subgroup/clustering of keywords.

    """

    num_entries = df_paper.shape[0]

    dct_abstracts = {}

    num_keywords = get_number_of_elements_nested_list(list_of_keyword_lists)

    # For every document

    for num_paper in range(num_entries):

        # Continue with paper if abstract is not empty

        if type(df_paper.iloc[num_paper,8]) != float:

            

            # Get abstract

            txt_abstract = df_paper.iloc[num_paper,8].split()

            idx = num_paper

            

            # Evaluate abstract

            arr_keyword_list_hits = evaluate_text_via_list_of_list_of_keywords(txt_abstract, list_of_keyword_lists)



            # Scoring abstract

            int_keyword_frequency = sum(arr_keyword_list_hits)

            num_keyword_sources = len(arr_keyword_list_hits[arr_keyword_list_hits > 0])

            num_keyword_lists = len(list_of_keyword_lists)

            numerator = num_keyword_sources + int_keyword_frequency

            denominator = num_keyword_lists + num_keywords



            abstract_score = numerator/denominator



            dct_abstracts.update({num_paper: {"Abstract Score": abstract_score}})



    return dct_abstracts
data_dir = "../input/CORD-19-research-challenge/"

data_file = "metadata.csv"

data = pd.read_csv(data_dir+data_file)
# Corona

lst_corona = ["Corona", "corona", "corona virus", "coronavirus", "corona viruses", "coronaviruses", "Coronaviridae", "coronaviridae", "COVID-19", "Covid-19", "covid-19", "COVID", "COV", "SARS"]



# Main Task

lst_genetics = ["genetics"]

lst_origin = ["origin", "member", "family"]

lst_evolution = ["evolution", "development", "develops", "developed"]

lst_task = [lst_genetics, lst_origin, lst_evolution]



# Sub task 1 - Real-time tracking ...

lst_subtask_1_genome = ["Genome", "genome"]

lst_subtask_1_dissemination = ["dissemination", "Dissemination", "propagation", "Propagation", "spread", "Spread", "spreading", "Spreading"]

lst_subtask_1_treatment = ["treatment", "Treatment", "diagnostic", "Diagnostic", "diagnostics", "Diagnostics", "therapeutics", "Therapeutics"]

lst_subtask_1_variation = ["Difference" , "in contrast", "variation", "deviation", "shows mutations", "enrichment", "similarities"]

lst_subtask_1_reference = ["Accession number", "reference", "sample", "identification of"]

lst_subtask_1_known = ["Known", "already published", "already reported"]

lst_subtask_1 = [lst_subtask_1_genome, lst_subtask_1_dissemination, lst_subtask_1_treatment, lst_subtask_1_variation, lst_subtask_1_reference, lst_subtask_1_known, lst_corona]



# Sub task 2 - Access to geographic ...

lst_subtask_2_ = []

lst_subtask_2_ = []

lst_subtask_2_ = []

lst_subtask_2_ = []

lst_subtask_2_ = []

lst_subtask_2 = []



# Sub task 3 - Evidence that livestock ...



# Sub sub task 3-1

#lst_subtask_3_1_Test = []

#lst_subtask_3_1 = []



# Sub sub task 3-2

lst_subtask_3_2_livestock = ["farm" , "wildlife", "wild animal", "undomesticated", "livestock"]

lst_subtask_3_2_area = ["Southeast-Asia"]

lst_subtask_3_2_control = ["surveil", "control", "screen", "check", "monitor", "examine"]

lst_subtask_3_2 = [lst_subtask_3_2_livestock, lst_subtask_3_2_area, lst_subtask_3_2_control, lst_corona]



# Sub sub task 3-3

lst_subtask_3_1_host = ["host" , "organism", "human"]

lst_subtask_3_2_infection = ["infection", "disease", "respiratory syndrom"]

lst_subtask_3_3_lab = ["experimental", "laboratory", "under conditions"]

lst_subtask_3 = [lst_subtask_3_1_host, lst_subtask_3_2_infection, lst_subtask_3_3_lab, lst_corona]



# Sub task 4

lst_subtask_4_host = ["animal", "host", "hosts", "Host", "Hosts", "human", "Human", "Humans", "humans", "CoV-Host", "organism"]

lst_subtask_4_transmission = ["pathogen", "spill-over", "intraspecies", "interaction", "host-shift", "spread", "evolution", "transmission", "infection"]

lst_subtask_4_evidence = ["evidence", "proof", "association", "connection", "associated"]

lst_subtask_4 = [lst_subtask_4_host, lst_subtask_4_transmission, lst_subtask_4_evidence, lst_corona]



# Sub task 5

lst_subtask_5_ = []

lst_subtask_5_ = []

lst_subtask_5_ = []

lst_subtask_5_ = []

lst_subtask_5 = []



# Sub task 6

lst_subtask_6_ = []

lst_subtask_6_ = []

lst_subtask_6_ = []

lst_subtask_6_ = []

lst_subtask_6_ = []

lst_subtask_6 = []
dct_res = evaluate_journal_by_keywords(data, lst_subtask_1)
lst_ranked = sorted(dct_res, key = lambda x: (dct_res[x]["Abstract Score"]), reverse = True)

lst_ranked[:3]
data.iloc[23643, 8]