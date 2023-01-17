# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option("display.max_colwidth", 200)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

import re



import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

from gensim.models import LsiModel

from gensim.models.coherencemodel import CoherenceModel



from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer 

import nltk



from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV



import spacy



import pyLDAvis

import pyLDAvis.gensim

import pyLDAvis.sklearn



from pprint import pprint

from sklearn.decomposition import NMF
import time

start_time = time.time()
nltk.download('stopwords')

nltk.download('wordnet')
train_1 = pd.read_csv('/kaggle/input/npr-data/npr.csv')

#train = train_1[:5000]

train = train_1

train.shape
print(train.columns)

train.head()
# Convert to list

data = train['Article'].values.tolist()



data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]



data = [re.sub('\s+', ' ', sent) for sent in data]



data = [re.sub("\'", "", sent) for sent in data]



pprint(data[:1])
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



data_words = list(sent_to_words(data))



print(data_words[:1])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))

    return texts_out



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)

# Run in terminal: python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only Noun, Adj, Verb, Adverb

data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(data_lemmatized[:2])
print(len(data_words))

print(len(data_lemmatized))
n_features = 5000

n_components = 12

n_top_words = 10



# ignore terms that have a document frequency strictly higher than 95%, 

# ignore terms that have a document frequency strictly lower than 2

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,

                                   max_features=n_features,

                                   stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(data_lemmatized)



# alpha=0 means no regularization, l1_ratio=.5, the penalty is a combination of L1 and L2

nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

nmf_output = nmf.fit_transform(tfidf)
topicnames = ["Topic" + str(i) for i in range(nmf.n_components)]
docnames = ["Doc" + str(i) for i in range(len(data))]
df_document_topic = pd.DataFrame(np.round(nmf_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")

df_topic_distribution.columns = ['Topic Num', 'Num Documents']

df_topic_distribution
# Topic-Keyword Matrix

df_topic_keywords = pd.DataFrame(nmf.components_)



# Assign Column and Index

df_topic_keywords.columns = tfidf_vectorizer.get_feature_names()

df_topic_keywords.index = topicnames
# View

df_topic_keywords.head()
# Show top n keywords for each topic

def show_topics(vectorizer=tfidf_vectorizer, lda_model=nmf, n_words=10):

    keywords = np.array(vectorizer.get_feature_names())

    topic_keywords = []

    for topic_weights in lda_model.components_:

        top_keyword_locs = (-topic_weights).argsort()[:n_words]

        topic_keywords.append(keywords.take(top_keyword_locs))

    return topic_keywords



#topic_keywords = show_topics(vectorizer=tfidf_vectorizer, lda_model=nmf, n_words=10)        

topic_keywords = show_topics()        



# Topic - Keywords Dataframe

df_topic_keywords = pd.DataFrame(topic_keywords)

df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]

df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

df_topic_keywords
type(df_topic_keywords)
elapsed_time = time.time() - start_time

time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = df_topic_keywords

#df = df_topic_distribution



# create a link to download the dataframe

create_download_link(df)
data[13]
df_document_topic.loc['Doc13']
import time

start_time = time.time()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option("display.max_colwidth", 200)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

import re



import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

from gensim.models import LsiModel

from gensim.models.coherencemodel import CoherenceModel



from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer 

import nltk



import spacy



import pyLDAvis

import pyLDAvis.gensim



from pprint import pprint
nltk.download('stopwords')

nltk.download('wordnet')
documents = train['Article'].values.tolist()

print(type(documents))

print(type(documents[0]))

print(len(documents))
documents[1]
def preprocess_data1(doc_set):

    """

    Input  : docuemnt list

    Purpose: preprocess text (tokenize, removing stopwords, and stemming)

    Output : preprocessed text

    """

    # initialize regex tokenizer

    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list

    en_stop = set(stopwords.words('english'))

    #en_stop = set(stopwords)

    # Create p_stemmer of class PorterStemmer

    lemmatizer = WordNetLemmatizer()

    # list for tokenized documents in loop

    texts = []

    # loop through document list

    for i in doc_set:

        # clean and tokenize document string

        raw = i.lower()

        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens

        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens

        stemmed_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]

        # add tokens to list

        texts.append(stemmed_tokens)

    return texts
clean_text=preprocess_data1(documents)
print(type(clean_text))

print(type(clean_text[0]))

print(len(clean_text))

clean_text[1]
def prepare_corpus(doc_clean):

    """

    Input  : clean document

    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix

    Output : term dictionary and Document Term Matrix

    """

    

    global doc_term_matrix

    

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # generate LDA model

    return dictionary,doc_term_matrix
def create_gensim_lda_model(doc_clean,number_of_topics,words):

    

    global lsamodel

    """

    Input  : clean document, number of topics and number of words associated with each topic

    Purpose: create LSA model using gensim

    Output : return LSA model

    """

    dictionary,doc_term_matrix=prepare_corpus(doc_clean)

    # generate LSA model

    #ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model

    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model

    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))

    return lsamodel
number_of_topics=12

words=10

model=create_gensim_lda_model(clean_text,number_of_topics,words)
pprint(lsamodel.print_topics())
optimal_model = lsamodel

model_topics = optimal_model.show_topics(formatted=False)

pprint(optimal_model.print_topics(num_words=10))
def format_topics_sentences(ldamodel=lsamodel, corpus=doc_term_matrix, texts=documents):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)





df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=doc_term_matrix, texts=documents)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']



# Show

df_dominant_topic.head(10)

df_dominant_topic.shape
df_dominant_topic[13:14]
# Group top 5 sentences under each topic

sent_topics_sorteddf_mallet = pd.DataFrame()



sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')



for i, grp in sent_topics_outdf_grpd:

    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 

                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 

                                            axis=0)



# Reset Index    

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)



# Format

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]



# Show

sent_topics_sorteddf_mallet.head()

sent_topics_sorteddf_mallet.shape
elapsed_time = time.time() - start_time

time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
type(model_topics)
model_topics
df = pd.DataFrame(df_dominant_topic['Keywords'].unique())
df
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = df

#df = df_topic_distribution



# create a link to download the dataframe

create_download_link(df)
elapsed_time = time.time() - start_time

time.strftime("%H:%M:%S", time.gmtime(elapsed_time))