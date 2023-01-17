import os

import json

from pprint import pprint

from copy import deepcopy



import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



! pip install glove_python

from glove import Corpus, Glove

import re

from IPython.display import HTML



from sklearn.decomposition import PCA



! pip install country_converter --upgrade

! pip install pycountry

import country_converter as coco

import pycountry
# helper functions to clean data and parse into tabular format



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

#             format_authors(file['metadata']['authors']),

#             format_authors(file['metadata']['authors'], 

#                            with_affiliation=True),

#             format_body(file['abstract']),

            format_body(file['body_text'])

#             format_bib(file['bib_entries']),

#             file['metadata']['authors'],

#             file['bib_entries']

        ]



        cleaned_files.append(features)



#     col_names = ['paper_id', 'title', 'authors',

#                  'affiliations', 'abstract', 'text', 

#                  'bibliography','raw_authors','raw_bibliography']

    col_names = ['paper_id', 'title', 'text']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    



    return clean_df
# load files from each folder

# biorxiv/medrxiv

#brx_files = load_files('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/')



# custom license

#pdf_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'

#pdf_files = load_files(pdf_dir)



# common use subset

#comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

#comm_files = load_files(comm_dir)



# noncommon use subset

#noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

#noncomm_files = load_files(noncomm_dir)
# read files in each folder into a dataframe and combined into one single csv called complete_df.csv and save to the output folder

# biorxiv/medrxiv

#complete_df = generate_clean_df(brx_files)



# custom license

#tmp_df = generate_clean_df(pdf_files)

#complete_df = pd.concat([complete_df, tmp_df])



# common use subset

#tmp_df = generate_clean_df(comm_files)

#complete_df = pd.concat([complete_df, tmp_df])



# noncommon use subset

#tmp_df = generate_clean_df(noncomm_files)

#complete_df = pd.concat([complete_df, tmp_df])



# save to disk

#complete_df.reset_index(inplace= True ,drop = True)

#complete_df.to_csv('complete_df.csv')
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
source_count = metadata.groupby('source_x')['cord_uid'].nunique().sort_values(ascending = False)

# Plot a bar graph:

plt.bar(

    source_count.index,

    source_count.values,

    align="center",

    color="orange"

)



plt.title("Number of papers by database source")

plt.xticks(source_count.index)

plt.show()
topjournal = metadata.groupby('journal')['cord_uid'].nunique().sort_values(ascending = False).reset_index()

print('total number of papers:', np.sum(topjournal['cord_uid']))

topjournal.rename(columns={'cord_uid':'number of papers'}, inplace=True)

HTML(topjournal[:10].to_html(index=False)) # show top 10 journals
def clean_title(row):

    if row is None:

        row = ""

    row = str(row).lower()   

    row = re.sub('[^A-Za-z0-9]+', ' ', row)  

    word_list = row.split() 

    wnl = WordNetLemmatizer()

    stop_list = stopwords.words('english')

    word_list = [wnl.lemmatize(word) for word in word_list if word not in stop_list]  

    row = " ".join(word_list) 

    return row



metadata_title_df = pd.DataFrame({'Title':list(metadata['title'])})

metadata_title_df['Title'] = metadata_title_df['Title'].apply(clean_title)

# understand the most frequent words

metadata_words = " ".join(word for word in metadata_title_df.Title)

freq_words_metadata = pd.DataFrame(pd.value_counts(metadata_words.split(" ")).sort_values(ascending=False).reset_index())

freq_words_metadata.columns = ['words', 'freq']

freq_words_metadata = freq_words_metadata.reset_index(drop = True)
# more_stopwords = (list(freq_words_metadata['words'][1:30]))

# wnl = WordNetLemmatizer()

# metadata_words_clean = " ".join([wnl.lemmatize(word) for word in metadata_words.split(" ") if word not in more_stopwords])

wordcloud1 = WordCloud(max_font_size=30, max_words=200, background_color="white").generate(metadata_words)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud1, interpolation="bilinear")

plt.axis("off")

plt.show()
def word_tokenizer(sentence):

    return word_tokenize(sentence) 



def clean_words(tokenized_sentences):

    stop_words = set(stopwords.words('english'))

    # added a lemmatizer

    wnl = WordNetLemmatizer()

    return [wnl.lemmatize(t) for t in tokenized_sentences if not t in stop_words] 



def remove_unchars(doc):

    doc = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', doc, flags=re.MULTILINE)

    doc = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', doc)

    doc = re.sub(r'\b[0-9]+\b\s*', '', doc)

    return doc



def preprocessing_text(text):

    text = text.lower()

    text = remove_unchars(text)

    words = word_tokenizer(text)

    cleaned_words = clean_words(words)

    return cleaned_words



def create_corpus(df):

    corpus=[]

    stop_words = stopwords.words('english')

    wnl = WordNetLemmatizer()

    for abstract in tqdm(df['abstract']):

        words=[wnl.lemmatize(word.lower()) for word in word_tokenize(abstract) if((word not in stop_words))]

        corpus.append(words)

    return corpus
metadata['abstract'] = metadata['abstract'].astype(str)

metadata['abstract']= metadata['abstract'].apply(lambda x : remove_unchars(x))

corpus_abstract = create_corpus(metadata)
#training the corpus to generate the co occurence matrix which is used in GloVe

corpus = Corpus ()

corpus.fit(corpus_abstract, window=10)



#creating a Glove object which will use the matrix created in the above lines to create embeddings

#We can set the learning rate as it uses Gradient Descent and number of components



glove = Glove(no_components=40, learning_rate=0.01)

 

glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=False)

glove.add_dictionary(corpus.dictionary) 

glove.save('glove_40d.model')
# helper function for checking similar word results

def get_similar_words(word_list, model, n_words=20, show_html=True):

    """

    return top n words from a word list, output in table format

    """

    new_df = pd.DataFrame()

    words = []

    for word in word_list:

        sim_result = model.most_similar(word, n_words+1)

        df = pd.DataFrame(sim_result, columns=['similar to \'' + word + '\'', 'score'])

        new_df = pd.concat([new_df, df], axis=1)

        words = words + df.iloc[:, 0].tolist()

        

    if show_html:

        display(HTML(new_df.to_html(escape=False,index=False)))    

        

    return(new_df, words)
# load saved model and dataset (optional)

#glove = Glove.load('glove_40d.model')

#complete_df = pd.read_csv('complete_df.csv')
_,_= get_similar_words(['detection'], glove)
# text search function

def text_search(word_list, data, column, limit=10):

    """

    count how many words from a word list appear in a given body of text

    return indices of counts in descending order

    """

    text_list = [str(text).lower() for text in data.loc[:, column].tolist()]

    

    counts = np.zeros(len(text_list))

    for idx, text in enumerate(text_list):

        tc = [1 if word in text else 0 for word in word_list]

        counts[idx] = np.sum(tc)

    

    idx = np.argsort(-counts)[:limit]

    

    return((idx, counts[idx]))





def parse_result(metadata, idx, topic_str):

    """

    format search output result in the notebook to match the suggested submission format

    formatting code inspired by https://www.kaggle.com/mlconsult/summary-page-covid-19-risk-factors

    """

    df_table = pd.DataFrame(columns = ["pub_date", "authors", "title"])

    meta_sub = metadata.iloc[idx, :]

    for index, row in meta_sub.iterrows():

        authors=str(row["authors"]).split(", ")

        link=row['doi']

        title=row["title"]

        linka='https://doi.org/'+str(link)

        linkb=title

        final_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,linkb)

        to_append = [row['publish_time'],authors[0]+' et al.',final_link]

        df_length = len(df_table)

        df_table.loc[df_length] = to_append

    

    filename=topic_str+'.csv'

    df_table.to_csv(filename,index = False)

    df_table=HTML(df_table.to_html(escape=False,index=False))

    display(df_table)

    

    return(meta_sub['sha'])



def subtopic_to_df(metadata, idx, topic_str):

    """

    save the artile title, publish date, id and cordid into a df

    """

    df_table = pd.DataFrame(columns = ["title", "pub_date", "authors", "sha"])

    meta_sub = metadata.iloc[idx, :]

    for index, row in meta_sub.iterrows():

        title=row["title"]

        publish = row['publish_time']

        authors=str(row["authors"]).split(", ")

        sha = row['sha']

        to_append = [title, publish,authors[0]+' et al.',sha]

        df_length = len(df_table)

        df_table.loc[df_length] = to_append

    

    return(df_table)

manual_words = ['detection', 'asymptomatic', 'antibody']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx1,_ = text_search(w, metadata, 'abstract', limit=20)



ids1 = parse_result(metadata, idx1, 'topic1')
manual_words = ['surveillance', 'platform']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx2,_ = text_search(w, metadata, 'abstract', limit=20)



ids2 = parse_result(metadata, idx2, 'topic2')
manual_words = ['local', 'support']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx3,_ = text_search(w, metadata, 'abstract', limit=20)



ids3 = parse_result(metadata, idx3, 'topic3')
manual_words = ['best', 'practice', 'communication', 'public']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx4,_ = text_search(w, metadata, 'abstract', limit=20)



ids4 = parse_result(metadata, idx4, 'topic4')
manual_words = ['pointofcare', 'rapid', 'tradeoff']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx5,_ = text_search(w, metadata, 'abstract', limit=20)



ids5 = parse_result(metadata, idx5, 'topic5')
manual_words = ['pcr', 'adhoc', 'intervention']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx6,_ = text_search(w, metadata, 'abstract', limit=20)



ids6 = parse_result(metadata, idx6, 'topic6')
manual_words = ['issue', 'migrate', 'instrument']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx7,_ = text_search(w, metadata, 'abstract', limit=20)



ids7 = parse_result(metadata, idx7, 'topic7')
manual_words = ['evolution', 'mutation']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx8,_ = text_search(w, metadata, 'abstract', limit=20)



ids8 = parse_result(metadata, idx8, 'topic8')
manual_words = ['viralload']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx9,_ = text_search(w, metadata, 'abstract', limit=20)



ids9 = parse_result(metadata, idx9, 'topic9')
manual_words = ['cytokine']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx10,_ = text_search(w, metadata, 'abstract', limit=20)



ids10 = parse_result(metadata, idx10, 'topic10')
manual_words = ['protocol', 'screening']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx11,_ = text_search(w, metadata, 'abstract', limit=20)



ids11 = parse_result(metadata, idx11, 'topic11')
manual_words = ['supply', 'swab', 'reagent']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx12,_ = text_search(w, metadata, 'abstract', limit=20)



ids12 = parse_result(metadata, idx12, 'topic12')
manual_words = ['technology', 'roadmap']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx13,_ = text_search(w, metadata, 'abstract', limit=20)



ids13 = parse_result(metadata, idx13, 'topic13')
manual_words = ['coalition', 'preparedness']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx14,_ = text_search(w, metadata, 'abstract', limit=20)



ids14 = parse_result(metadata, idx14, 'topic14')
manual_words = ['crispr']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx15,_ = text_search(w, metadata, 'abstract', limit=20)



ids15 = parse_result(metadata, idx15, 'topic15')
manual_words = ['genomic', 'scale']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx16,_ = text_search(w, metadata, 'abstract', limit=20)



ids16 = parse_result(metadata, idx16, 'topic16')
manual_words = ['sequencing', 'bioinformatics']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx17,_ = text_search(w, metadata, 'abstract', limit=20)



ids17 = parse_result(metadata, idx17, 'topic17')
manual_words = ['technology', 'unknown', 'naturally']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx18,_ = text_search(w, metadata, 'abstract', limit=20)



ids18 = parse_result(metadata, idx18, 'topic18')
manual_words = ['spillover', 'pathogen']

print('Initial search words: ' + ', '.join(manual_words))

_, w = get_similar_words(manual_words, glove, show_html=False)

print('Similar keywords: ', w)

idx19,_ = text_search(w, metadata, 'abstract', limit=20)



ids19 = parse_result(metadata, idx19, 'topic19')
# combine all the articles of these 19 topics into a single dataframe

all_idx = [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10, idx11, idx12, idx13, idx14, idx15, idx16, idx17, idx18, idx19]

df_all = {}

for topic_id in range(1,20):

    idx = all_idx[topic_id-1]

    df_name = 'df_' + str(topic_id)

    df_data = subtopic_to_df(metadata, idx, topic_id)

    df_data['number of times appeared in topic search'] = 'topic' + str(topic_id)

    df_all[df_name] = df_data

selected_articles = pd.DataFrame(columns=["title", "pub_date", "authors", "sha", "number of times appeared in topic search"])

for df in df_all:

    selected_articles = selected_articles.append(df_all[df])

selected_articles = selected_articles.reset_index(drop = True)

#selected_articles.to_csv("selected articles.csv",index = False) # "optional"
overlap = pd.DataFrame(selected_articles.groupby(['title', "authors", "pub_date"])['number of times appeared in topic search'].count()).sort_values(by = ['number of times appeared in topic search'], ascending=False)

overlap = overlap.reset_index()

overlap.head(10)
selected_articles['pub_date'] = pd.to_datetime(selected_articles['pub_date'], format = "%Y-%m-%d")

selected_articles['pub_year'] = selected_articles['pub_date'].dt.year

publish_by_year = pd.DataFrame(selected_articles.groupby(['pub_year'])['sha'].nunique()).reset_index()



plt.figure(figsize=(15,8))

plt.plot(publish_by_year['pub_year'], publish_by_year['sha'])  



plt.xlabel("Publish Year", fontsize=12)

plt.ylabel("Number of Articles", fontsize=12)

plt.title("Number of selected articles by publish year", fontsize=15)