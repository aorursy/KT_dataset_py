# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# There are too many paths and printing them takes

# up too much space so I don't do this normally

if 1==1: 

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
# import commands to

# convert text to a matrix of

# tokens

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# import sklearn LDA function

from sklearn.decomposition import LatentDirichletAllocation



# import scispacy, a repo of commands

# to deal with scientific documents

import scispacy



# import spaCy, a repo of commands

# to deal with natural language processing 

# (NLP)

import spacy



# Of the spaCy library, 

# the en_core_sci_sm library contains

# a full spaCy pipeline for biomedical data 

import en_core_sci_sm



# import command to measure

# the Jensen-Shannon distance (metric)

from scipy.spatial.distance import jensenshannon



# import joblib, a repo of commands

# to run python functions as pipeline

# jobs

import joblib



# import pyLDAvis, a repo of commands

# to create interactic topic model visualizations

import pyLDAvis

import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")

clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")



all_data = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True)



all_data.head()
print("Number of Rows in Table: %i" % len(all_data))

print("Number of Titles: %i " % all_data['title'].count())

print("Number of Abstracts: %i " % all_data['abstract'].count())

print("Number of Texts: %i " % all_data['text'].count())
# replace empty text with empty strings

all_text = all_data.text.str.replace('\n\n', '')
all_text[58][:500]
# initalize the spaCy pipeline

# for biomedical data

nlp = en_core_sci_sm.load()
# ignore some words that are irrelevant for the content

stop_words = text.ENGLISH_STOP_WORDS.union({'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', '10'})



tf_vectorizer = CountVectorizer(strip_accents = 'unicode',

                                stop_words = stop_words,

                                lowercase = True

                               )



tf = tf_vectorizer.fit_transform(all_text)



tf.shape
lda_csv_file="../input/exploring-corona-competition/lda_10topics.csv"

ntopics=10

if os.path.exists(lda_csv_file):

    lda_tf = joblib.load(lda_csv_file)

else:

    lda_tf = LatentDirichletAllocation(n_components=ntopics,

                                       random_state=0)

    lda_tf.fit(tf)

joblib.dump(lda_tf, 'lda.csv')
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()
tfidf_feature_names = tf_vectorizer.get_feature_names()

print_top_words(lda_tf, tfidf_feature_names, 20)
viz = pyLDAvis.sklearn.prepare(lda_tf, tf, tf_vectorizer)
pyLDAvis.display(viz)
pyLDAvis.save_html(viz, 'lda.html')
topic_dist = pd.DataFrame(lda_tf.transform(tf))



topic_dist.head()

def get_k_nearest_docs(doc_dist, k=5, use_jensenshannon=True):

    '''

    doc_dist: topic distribution (sums to 1) of one article

    

    Returns the index of the k nearest articles (as by Jensen–Shannon divergence/ Euclidean distance in topic space). 

    '''

    

    if use_jensenshannon:

            distances = topic_dist.apply(lambda x: jensenshannon(x, doc_dist), axis=1)

    else:

        diff_df = topic_dist.sub(doc_dist)

        distances = np.sqrt(np.square(diff_df).sum(axis=1)) # euclidean distance (faster)

        

    return distances[distances != 0].nsmallest(n=k).index
def recommendation(paper_id, k=5):

    '''

    Returns the title of the k papers that are closest (topic-wise) to the paper given by paper_id.

    '''

    

    print(all_data.title[all_data.paper_id == paper_id].values[0])

    print('\nRELATED DOCUMENTS: \n')

    recommended = get_k_nearest_docs(topic_dist[all_data.paper_id == paper_id].iloc[0], k)

    for i in recommended:

        print('- ', all_data.title[i] )
recommendation('11f96d05db6854ef95312aa3a4736724ce1f02d6', k=5)
recommendation('1d2947d5c0addcbbcfa71d87ea074a8b6ff2a973', k=5)
def relevant_articles(tasks, k=3):

    tasks = [tasks] if type(tasks) is str else tasks 

    

    tasks_tf = tf_vectorizer.transform(tasks)

    tasks_topic_dist = pd.DataFrame(lda_tf.transform(tasks_tf))



    for index, bullet in enumerate(tasks):

        print('\n=============================================')

        print('\nBULLET: ' + bullet + '\n')



        print('\nRELATED DOCUMENTS: \n')

        recommended = get_k_nearest_docs(tasks_topic_dist.iloc[index], k)

        for i in recommended:

            print('- ', all_data.title[i] )
npi_task_raw=["Guidance on ways to scale up non-pharmaceutical interventions in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.", \

"Rapid design and execution of experiments to examine and compare non-pharmaceutical interventions currently being implemented. Department of Homeland Security Centers for Excellence could potentially be leveraged to conduct these experiments.", \

"Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.", \

"Methods to control the spread in communities, barriers to compliance and how these vary among different populations..", \

"Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.", \

"Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with non-pharmaceutical interventions.", \

"Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).", \

"Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."]
relevant_articles(npi_task_raw, 5)