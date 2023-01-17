dirs = {

    'biorxiv_dir': '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/',

    'comm': '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/',

    'noncomm': '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/',

    'pmc': '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/',

}
import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import json

import os

import glob

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm



pd.set_option('display.max_columns', 50)

pd.options.display.max_colwidth = 100
def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)



def load_files(source, dirname):

    filenames = os.listdir(dirname)

    raw_files = []

    print("Loading {} articles retrieved from {}:".format(len(filenames), source))

    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(dirs=dirs):

    cleaned_files = []



    for d in dirs:

        all_files = load_files(d, dirs[d])

        print('Processing files for DF')

        for file in tqdm(all_files):

            features = [

                file['paper_id'],

                d,

                file['metadata']['title'],

                format_authors(file['metadata']['authors']),

                format_authors(file['metadata']['authors'], 

                               with_affiliation=True),

                format_body(file['abstract']),

                format_body(file['body_text']),

                format_bib(file['bib_entries']),

                file['metadata']['authors'],

                file['bib_entries']

            ]



            cleaned_files.append(features)



    col_names = ['paper_id', 'source', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
clean_df = generate_clean_df()
clean_df.head()
clean_df.to_csv('clean_df.csv',index=False)

del clean_df



import gc

gc.collect()



clean_df = pd.read_csv('/kaggle/working/clean_df.csv')
pd.options.display.max_colwidth = 50

keywords = ["ethics", "ethical", "social media", "anxiety", "psychological"]

keyword_df = clean_df[clean_df.abstract.str.contains('|'.join(keywords), case=False).any(level=0)] # isolate the studies with abstracts that contain the listed keywords



keyword_df.head()
print("Number of studies with abstracts containing social-science-related keywords:", len(keyword_df))
pd.options.display.max_colwidth = 10000

keyword_df.iloc[[2],5]
keyword_df.to_csv('covid19_socialscienceresearch.csv',index=False)
word_dict = {}



for k in keywords:

    count = len(keyword_df[keyword_df.abstract.str.contains(k, case=False)])

    word_dict[k] = count



word_dict
keys = word_dict.keys()

values = word_dict.values()



plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(15, 10))

plt.bar(keys, values)

plt.title("Social Science Keywords Contained in COVID-19 Research Articles", pad=20)

plt.ylabel("Count", labelpad=20)

plt.xlabel("Keyword", labelpad=20)

plt.show()
pd.options.display.max_colwidth = 50

del keyword_df

gc.collect()
vacc_keywords = ["antibody-dependent enhancement", "antibody dependent enhancement", "naproxen", "clarithromycin", "minocycline"]

vacc_df = clean_df[clean_df.abstract.str.contains('|'.join(vacc_keywords), case=False).any(level=0)] # isolate the studies with abstracts that contain the listed keywords



print("Number of studies with abstracts containing vaccine-related keywords:", len(vacc_df))
vacc_df.head()
vacc_df.to_csv('covid19_vaccine_research.csv',index=False)
word_dict = {}



for k in vacc_keywords:

    count = len(vacc_df[vacc_df.abstract.str.contains(k, case=False)])

    word_dict[k] = count



keys = word_dict.keys()

values = word_dict.values()



plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(15, 10))

plt.bar(keys, values)

plt.title("Vaccine-Related Keywords Contained in COVID-19 Research Articles", pad=20)

plt.ylabel("Count", labelpad=20)

plt.xlabel("Keyword", labelpad=20)

plt.xticks(rotation=20)

plt.show()
from wordcloud import WordCloud

from sacremoses import MosesDetokenizer

from nltk.corpus import stopwords

import nltk

from nltk.tokenize import word_tokenize



text = ' '.join(vacc_df['text'].tolist())

text = text.lower()



stoplist = stopwords.words('english')

new_stopwords = """patient patients result may use case method conclusion study research abstract methods et al model one outbreak information data"""



stoplist += new_stopwords.split()



clean_text = [word for word in text.split() if word not in stoplist]



detokenizer = MosesDetokenizer()

clean_text = detokenizer.detokenize(clean_text, return_str=True)



# Create the wordcloud object

wordcloud = WordCloud(width=480, height=480, margin=0).generate(clean_text)

 

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()
del vacc_df

gc.collect()



evo_keywords = ["fruit bat", "livestock", "farmers", "origins"]

evo_df = clean_df[clean_df.abstract.str.contains('|'.join(evo_keywords),case=False).any(level=0)] # isolate the studies with abstracts that contain the listed keywords



pd.options.display.max_colwidth = 140

evo_df.head()
word_dict = {}



for k in evo_keywords:

    count = len(evo_df[evo_df.abstract.str.contains(k, case=False)])

    word_dict[k] = count



keys = word_dict.keys()

values = word_dict.values()



plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(15, 10))

plt.bar(keys, values)

plt.title("Evolution/Origin Keywords Contained in COVID-19 Research Articles", pad=20)

plt.ylabel("Count", labelpad=20)

plt.xlabel("Keyword", labelpad=20)

plt.show()
pd.options.display.max_colwidth = 10000

evo_df.to_csv('covid19_evolution.csv',index=False)

evo_df.iloc[[11],5]
from wordcloud import WordCloud

from sacremoses import MosesDetokenizer

from nltk.corpus import stopwords

import nltk

from nltk.tokenize import word_tokenize



text = ' '.join(evo_df['text'].tolist())

text = text.lower()



stoplist = stopwords.words('english')

new_stopwords = """patient patients result may use case method conclusion study research abstract methods et al model one outbreak information data"""



stoplist += new_stopwords.split()



clean_text = [word for word in text.split() if word not in stoplist]



detokenizer = MosesDetokenizer()

clean_text = detokenizer.detokenize(clean_text, return_str=True)



# Create the wordcloud object

wordcloud = WordCloud(width=480, height=480, margin=0).generate(clean_text)

 

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()