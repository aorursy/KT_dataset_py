import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm
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
def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

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



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []



for filename in filenames:

    filename = biorxiv_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
file = all_files[0]

print("Dictionary keys:", file.keys())
pprint(file['abstract'])
print("body_text type:", type(file['body_text']))

print("body_text length:", len(file['body_text']))

print("body_text keys:", file['body_text'][0].keys())
print("body_text content:")

pprint(file['body_text'][:2], depth=3)
texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}

for section, text in texts:

    texts_di[section] += text



pprint(list(texts_di.keys()))
body = ""



for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"



print(body[:3000])

print(format_body(file['body_text'])[:3000])
print(all_files[0]['metadata'].keys())
print(all_files[0]['metadata']['title'])
authors = all_files[0]['metadata']['authors']

pprint(authors[:3])
for author in authors:

    print("Name:", format_name(author))

    print("Affiliation:", format_affiliation(author['affiliation']))

    print()

pprint(all_files[4]['metadata'], depth=4)
authors = all_files[4]['metadata']['authors']

print("Formatting without affiliation:")

print(format_authors(authors, with_affiliation=False))

print("\nFormatting with affiliation:")

print(format_authors(authors, with_affiliation=True))
bibs = list(file['bib_entries'].values())

pprint(bibs[:2], depth=4)
format_authors(bibs[1]['authors'], with_affiliation=False)
bib_formatted = format_bib(bibs[:5])

print(bib_formatted)

cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

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
col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



clean_df = pd.DataFrame(cleaned_files, columns=col_names)

clean_df.head()

clean_df.to_csv('biorxviv.csv')
clean_df.head()
text1="Range of incubation periods for the disease in humans and how long individuals are contagious, even after recovery"

text2="Prevalence of asymptomatic shedding and transmission"

text3="Seasonality of transmission"

text4="Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)."

text5="Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)."

text6="Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)"

text7="Natural history of the virus and shedding of it from an infected person"

text8="Implementation of diagnostics and products to improve clinical processes"

text9="Disease models, including animal models for infection, disease and transmission"

text10="Tools and studies to monitor phenotypic change and potential adaptation of the virus"

text11="Immune response and immunity"

text12="Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings"

text13="Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings"

text14="Role of the environment in transmission"

corp=[text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13,text14]

no_docs=len(corp)
#Grab all the text.

text=clean_df['text']

#Create a list of text for each entry.

text_list = [title for title in text]

#Collapse the texts into one large text for processing.

big_text_string = ' '.join(text_list)



from nltk.tokenize import word_tokenize



#Tokenize the string into words.

tokens = word_tokenize(big_text_string)



#Remove non alphabetic tokens such as punctuation

words = [word.lower() for word in tokens if word.isalpha()]



from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



words = [word for word in words if not word in stop_words]



import gensim

# Load word2vec model (trained on an enormous Google corpus)

model = gensim.models.KeyedVectors.load_word2vec_format("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", binary = True) 

# Check dimension of word vectors

model.vector_size
# Filter the list of vectors to include only those that Word2Vec has a vector for

vector_list = [model[word] for word in words if word in model.vocab]



# Create a list of the words corresponding to these vectors

words_filtered = [word for word in words if word in model.vocab]



# Zip the words together with their vector representations

word_vec_zip = zip(words_filtered, vector_list)



# Cast to a dict so we can turn it into a DataFrame

word_vec_dict = dict(word_vec_zip)

df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
df.head()
from sklearn.manifold import TSNE



# Initialize t-SNE

tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)



# Use only 400 rows to shorten processing time

tsne_df = tsne.fit_transform(df[:400])
import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

# Initialize figure

fig, ax = plt.subplots(figsize = (11.7, 8.27))

sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 0.5)



# Import adjustText, initialize list of texts

from adjustText import adjust_text

texts = []

words_to_plot = list(np.arange(0, 400, 10))



# Append words to list

for word in words_to_plot:

    texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize = 14))

    

# Plot text using adjust_text (because overlapping text is hard to read)

adjust_text(texts, force_points = 0.4, force_text = 0.4, 

            expand_points = (2,1), expand_text = (1,2),

            arrowprops = dict(arrowstyle = "-", color = 'black', lw = 0.5))



plt.show()

def document_vector(word2vec_model, doc):

    # remove out-of-vocabulary words

    doc = [word for word in doc if word in model.vocab]

    return np.mean(model[doc], axis=0)



# Our earlier preprocessing was done when we were dealing only with word vectors

# Here, we need each document to remain a document 

def preprocess(text):

    text = text.lower()

    doc = word_tokenize(text)

    doc = [word for word in doc if word not in stop_words]

    doc = [word for word in doc if word.isalpha()] 

    return doc



# Function that will help us drop documents that have no word vectors in word2vec

def has_vector_representation(word2vec_model, doc):

    """check if at least one word of the document is in the

    word2vec dictionary"""

    return not all(word not in word2vec_model.vocab for word in doc)



# Filter out documents

def filter_docs(corpus, texts, condition_on_doc):

    """

    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.

    """

    number_of_docs = len(corpus)



    if texts is not None:

        texts = [text for (text, doc) in zip(texts, corpus)

                 if condition_on_doc(doc)]



    corpus = [doc for doc in corpus if condition_on_doc(doc)]



    print("{} docs removed".format(number_of_docs - len(corpus)))



    return (corpus, texts)
# Preprocess the corpus

corpus = [preprocess(title) for title in text_list]



# Remove docs that don't include any words in W2V's vocab

corpus, text_list = filter_docs(corpus, text_list, lambda doc: has_vector_representation(model, doc))



# Filter out any empty docs

corpus, text_list = filter_docs(corpus, text_list, lambda doc: (len(doc) != 0))

x = []

for doc in corpus: # append the vector for each document

    x.append(document_vector(model, doc))

    

X = np.array(x) # list to array
# Initialize t-SNE

tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)



# Again use only 400 rows to shorten processing time

tsne_df = tsne.fit_transform(X[:400])

fig, ax = plt.subplots(figsize = (14, 10))

sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 0.5)



from adjustText import adjust_text

texts = []

titles_to_plot = list(np.arange(0, 400, 40)) # plots every 40th title in first 400 titles



# Append words to list

for title in titles_to_plot:

    texts.append(plt.text(tsne_df[title, 0], tsne_df[title, 1], titles_list[title], fontsize = 14))

    

# Plot text using adjust_text

adjust_text(texts, force_points = 0.4, force_text = 0.4, 

            expand_points = (2,1), expand_text = (1,2),

            arrowprops = dict(arrowstyle = "-", color = 'black', lw = 0.5))



plt.show()

commuse_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'

filenames = os.listdir(commuse_dir)

print("Number of articles retrieved from Commercial Use:", len(filenames))
all_files = []



for filename in filenames:

    filename = commuse_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
file = all_files[0]

print("Dictionary keys:", file.keys())
pprint(file['abstract'])
print("body_text type:", type(file['body_text']))

print("body_text length:", len(file['body_text']))

print("body_text keys:", file['body_text'][0].keys())
print("body_text content:")

pprint(file['body_text'][:2], depth=3)
texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}

for section, text in texts:

    texts_di[section] += text



pprint(list(texts_di.keys()))
body = ""



for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"



print(body[:3000])
print(format_body(file['body_text'])[:3000])
print(all_files[0]['metadata'].keys())
print(all_files[0]['metadata']['title'])
authors = all_files[0]['metadata']['authors']

pprint(authors[:3])
for author in authors:

    print("Name:", format_name(author))

    print("Affiliation:", format_affiliation(author['affiliation']))

    print()
pprint(all_files[4]['metadata'], depth=4)
authors = all_files[4]['metadata']['authors']

print("Formatting without affiliation:")

print(format_authors(authors, with_affiliation=False))

print("\nFormatting with affiliation:")

print(format_authors(authors, with_affiliation=True))
bibs = list(file['bib_entries'].values())

pprint(bibs[:2], depth=4)
format_authors(bibs[1]['authors'], with_affiliation=False)
bib_formatted = format_bib(bibs[:5])

print(bib_formatted)
cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

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
col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



commuse_df = pd.DataFrame(cleaned_files, columns=col_names)

commuse_df.head()
customlicense_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/'

filenames = os.listdir(customlicense_dir)

print("Number of articles retrieved from Commercial Use:", len(filenames))
all_files = []



for filename in filenames:

    filename = customlicense_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
file = all_files[0]

print("Dictionary keys:", file.keys())
pprint(file['abstract'])
print("body_text type:", type(file['body_text']))

print("body_text length:", len(file['body_text']))

print("body_text keys:", file['body_text'][0].keys())
print("body_text content:")

pprint(file['body_text'][:2], depth=3)
texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}

for section, text in texts:

    texts_di[section] += text



pprint(list(texts_di.keys()))
body = ""



for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"



print(body[:3000])
print(format_body(file['body_text'])[:3000])
print(all_files[0]['metadata'].keys())
print(all_files[0]['metadata']['title'])
authors = all_files[0]['metadata']['authors']

pprint(authors[:3])
for author in authors:

    print("Name:", format_name(author))

    print("Affiliation:", format_affiliation(author['affiliation']))

    print()
pprint(all_files[4]['metadata'], depth=4)
authors = all_files[4]['metadata']['authors']

print("Formatting without affiliation:")

print(format_authors(authors, with_affiliation=False))

print("\nFormatting with affiliation:")

print(format_authors(authors, with_affiliation=True))
bibs = list(file['bib_entries'].values())

pprint(bibs[:2], depth=4)
format_authors(bibs[1]['authors'], with_affiliation=False)
bib_formatted = format_bib(bibs[:5])

print(bib_formatted)
cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

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
col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



custom_df = pd.DataFrame(cleaned_files, columns=col_names)

custom_df.head()
noncommuse_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'

filenames = os.listdir(noncommuse_dir)

print("Number of articles retrieved from Commercial Use:", len(filenames))
all_files = []



for filename in filenames:

    filename = noncommuse_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
file = all_files[0]

print("Dictionary keys:", file.keys())
pprint(file['abstract'])
print("body_text type:", type(file['body_text']))

print("body_text length:", len(file['body_text']))

print("body_text keys:", file['body_text'][0].keys())
print("body_text content:")

pprint(file['body_text'][:2], depth=3)
texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}

for section, text in texts:

    texts_di[section] += text



pprint(list(texts_di.keys()))
body = ""



for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"



print(body[:3000])
print(format_body(file['body_text'])[:3000])
print(all_files[0]['metadata'].keys())
print(all_files[0]['metadata']['title'])
authors = all_files[0]['metadata']['authors']

pprint(authors[:3])
for author in authors:

    print("Name:", format_name(author))

    print("Affiliation:", format_affiliation(author['affiliation']))

    print()
pprint(all_files[4]['metadata'], depth=4)
authors = all_files[4]['metadata']['authors']

print("Formatting without affiliation:")

print(format_authors(authors, with_affiliation=False))

print("\nFormatting with affiliation:")

print(format_authors(authors, with_affiliation=True))
bibs = list(file['bib_entries'].values())

pprint(bibs[:2], depth=4)
format_authors(bibs[1]['authors'], with_affiliation=False)
bib_formatted = format_bib(bibs[:5])

print(bib_formatted)
cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

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
col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



noncom_df = pd.DataFrame(cleaned_files, columns=col_names)

noncom_df.head()