import pandas as pd

from tqdm import tqdm

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

import numpy as np

import unidecode

import string



# Libraries for text preprocessing

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

nltk.download('wordnet') 

from nltk.stem.wordnet import WordNetLemmatizer



#Word cloud

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer

import re



#Barplot of most freq Bi-grams

import seaborn as sns



from sklearn.feature_extraction.text import TfidfTransformer



from scipy.sparse import coo_matrix



#removing html tags and text

from lxml import html



# getting ngrams

import nltk

from nltk import word_tokenize

from nltk.util import ngrams

from collections import Counter

import collections



#nlp

from textblob import TextBlob



#nlp-spacy

import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()



import glob

import json

import tqdm



import csv

import requests 

from bs4 import BeautifulSoup

from bs4.element import Comment

import urllib.request



import difflib

from fuzzywuzzy import fuzz 

from fuzzywuzzy import process 



#import libraries specific to below code



import re, nltk, spacy, gensim





# Sklearn

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV

from pprint import pprint



# Plotting tools

import pyLDAvis

import pyLDAvis.sklearn

import matplotlib.pyplot as plt

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

meta_df.head()
len(meta_df)
# subset the meta_df 



meta_df = meta_df.head(5000)

meta_df
meta_df.abstract
meta_df.info()
meta_df = meta_df[meta_df['abstract'].notna()]

meta_df = meta_df.reset_index()

meta_df
print (meta_df.shape)
meta_df['abstract'].describe(include='all')
meta_df.drop_duplicates(['abstract'], inplace=True)

meta_df['abstract'].describe(include='all')
stop_words = set(stopwords.words('english'))





# pos_words and extend words are some common words to be removed from abstract



pos_words = ['highest','among','either','seven','six','plus','strongest','worst','greatest','every','better','per','across','throughout','except','fewer','trillion','fewest','latest','least','manifest','unlike','eight','since','toward','largest','despite','via','finest','besides','easiest','must','million','oldest','behind','outside','smaller','nest','longest','whatever','stronger','worse','two','another','billion','best','near','nine','around','nearest','wechat','lowest','smallest','along','higher','three','older','greater','neither','inside','newest','lower','may','although','though','earlier','upon','five','ca','larger','us','whether','beyond','onto','might','one','out','unless','four','whose','can','fastest','without','ecobooth','broadest','easier','within','like', 'could','biggest','bigger','would','thereby','yet','timely','thus','also','avoid','know','usually','time','year','go','welcome','even','date']

extend_words =['used', 'following', 'go', 'instead', 'fundamentally', 'first', 'second', 'alone', 'everything', 'end', 'also', 'year', 'made', 'many', 'towards', 'truly', 'last', 'often', 'called', 'new', 'date', 'fully', 'thus', 'new', 'include', 'http', 'www','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure','rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI','-PRON-']



pos_words.extend(extend_words)

pos_words

stop_words = stop_words.union(pos_words)



def text_preprocess(text):

    lemma = nltk.wordnet.WordNetLemmatizer()

    

    #Convert to lower

    text = text.lower()

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)



    # remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    

    #Remove accented characters

    text = unidecode.unidecode(text)

    

    #Remove punctuation

    table = str.maketrans('', '', string.punctuation)

    text = [w.translate(table) for w in text.split()]

    

    lemmatized = []

    #Lemmatize non-stop words and save

    other_words = ['virus','study','viral','human','infection'] # common words to remove specific to these articles

    for word in text:

        if word not in stop_words:

            x = lemma.lemmatize(word)

            if x not in other_words:

                lemmatized.append(x)

   

    result = " ".join(lemmatized)

    return result
print(meta_df.abstract[0])



print (meta_df.processed_abstract[0])
meta_df['processed_abstract'] = meta_df['abstract'].apply(text_preprocess)
# Converting processed_abstract into list 



docs_raw = meta_df["processed_abstract"].tolist() 


#Convert to document-term matrix



tf_vectorizer = CountVectorizer(strip_accents = 'unicode',

                                stop_words = 'english',

                                lowercase = True,

                                token_pattern = r'\b[a-zA-Z]{3,}\b',

                                max_df = 0.5, 

                                min_df = 10)

dtm_tf = tf_vectorizer.fit_transform(docs_raw)

print(dtm_tf.shape)
print (tf_vectorizer.get_params())



#(38666, 14273)



#(38666, 80928)
#tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())

tfidf_vectorizer = TfidfVectorizer(use_idf=True)

dtm_tfidf = tfidf_vectorizer.fit_transform(docs_raw)

print(dtm_tfidf.shape)
# get the first vector out (for the first document)

first_vector_tfvectorizer=dtm_tf[1]

 

# place tf-idf values in a pandas data frame

test1 = pd.DataFrame(first_vector_tfvectorizer.T.todense(), index=tf_vectorizer.get_feature_names(), columns=["bow"])

test1.sort_values(by=["bow"],ascending=False)

# get the first vector out (for the first document)

first_vector_tfidfvectorizer=dtm_tfidf[1]

 

# place tf-idf values in a pandas data frame

test1 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

test1.sort_values(by=["tfidf"],ascending=False)



# for TF DTM

lda_tf = LatentDirichletAllocation(n_components=10, random_state=0)

lda_tf.fit(dtm_tf)

# for TFIDF DTM

lda_tfidf = LatentDirichletAllocation(n_components=10, random_state=0)

lda_tfidf.fit(dtm_tfidf)
# Define Search Param

search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

# Init the Model

lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0)

# Init Grid Search Class

model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search

model.fit(dtm_tf)

GridSearchCV(cv=None, error_score='raise',

       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,

             evaluate_every=-1, learning_decay=0.7, learning_method=None,

             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,

             mean_change_tol=0.001, n_components=10, n_jobs=1,

             #n_components=None,

             perp_tol=0.1, random_state=None,

             topic_word_prior=None, total_samples=1000000.0, verbose=0),

       #fit_params=None,

        iid=True, n_jobs=1,

       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},

       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',

       scoring=None, verbose=0)
# Best Model

best_lda_model = model.best_estimator_

# Model Parameters

print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score

print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity

print("Model Perplexity: ", best_lda_model.perplexity(dtm_tf))
# Create Document — Topic Matrix

lda_output = best_lda_model.transform(dtm_tf)

# column names

topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names

docnames = [i for i in meta_df.cord_uid]

# Make the pandas dataframe

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic["dominant_topic"] = dominant_topic

# Styling

def color_green(val):

 color = 'green' if val > .1 else 'black'

 return 'color: {col}'.format(col=color)

def make_bold(val):

 weight = 700 if val > .1 else 400

 return 'font-weight: {weight}'.format(weight=weight)

# Apply Style

df_document_topics = df_document_topic.head(43).style.applymap(color_green).applymap(make_bold)

#df_document_topics = df_document_topic.style.applymap(color_green).applymap(make_bold)

df_document_topics
# Topic-Keyword Matrix

df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index

df_topic_keywords.columns = tf_vectorizer.get_feature_names()

df_topic_keywords.index = topicnames

# View

df_topic_keywords



# Get the top 15 keywords for each topic



# Show top n keywords for each topic

def show_topics(vectorizer=tf_vectorizer, lda_model=lda_tf, n_words=20):

    keywords = np.array(vectorizer.get_feature_names())

    topic_keywords = []

    for topic_weights in lda_model.components_:

        top_keyword_locs = (-topic_weights).argsort()[:n_words]

        topic_keywords.append(keywords.take(top_keyword_locs))

    return topic_keywords

topic_keywords = show_topics(vectorizer=tf_vectorizer, lda_model=best_lda_model, n_words=15)

# Topic - Keywords Dataframe

df_topic_keywords = pd.DataFrame(topic_keywords)

df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]

df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

df_topic_keywords 
Topics = ['Antibody-dna gene analysis-pig/bat','PEDV/Coronovirus acute Respiration-diarrhea','Immunity-inflammation Induced in Body/gene','High Risk grouping people-older age','HIV-HCV virus cellular membrane','Novel Virus diseases/infectious from Animals','Clinical treatment/dignosis for pneumonia-hadv-respirational failure','Influenza-sars-corona  outbreak/pandemic from china','Data model approach - disease analysis/pattern ','Infectious diseases outbreak globally']

df_topic_keywords["Topics"]=Topics

df_topic_keywords
def apply_predict_topic(text):

    text = text

    infer_topic, topic, prob_scores = predict_topic(text = text)

    return infer_topic
# Define function to predict topic for a given text document.

nlp = spacy.load('en', disable=['parser', 'ner'])

def predict_topic(text, nlp=nlp):

# Step 1: Clean with simple_preprocess

    text_2 = text_preprocess(text)

# Step 3: Vectorize transform

    text_3 = [text_2]

    text_4 = tf_vectorizer.transform(text_3)

# Step 4: LDA Transform

    topic_probability_scores = best_lda_model.transform(text_4)

    #topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 0:14].values.tolist()

    topic = np.argmax(topic_probability_scores, axis=1)

    

# Step 5: Infer Topic

    infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]

    

    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]

    return infer_topic, topic, topic_probability_scores

# Predict the topic



tasks = ["What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control? Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery. Prevalence of asymptomatic shedding and transmission (e.g., particularly children). Seasonality of transmission. Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding). Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood). Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic). Natural history of the virus and shedding of it from an infected person Implementation of diagnostics and products to improve clinical processes Disease models, including animal models for infection, disease and transmission Tools and studies to monitor phenotypic change and potential adaptation of the virus Immune response and immunity Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings Role of the environment in transmission",

        "What do we know about COVID-19 risk factors? What have we learned from epidemiological studies? Data on potential risks factors.Smoking, pre-existing pulmonary disease.Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities.Neonates and pregnant women.Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors.Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups.Susceptibility of populations .Public health mitigation measures that could be effective for control",

        "What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface? Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.Experimental infections to test host range for this pathogen.Animal host(s) and any evidence of continued spill-over to humans.Socioeconomic and behavioral risk factors for this spill-over.Sustainable risk reduction strategies",

        "What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics? Effectiveness of drugs being developed and tried to treat COVID-19 patients.Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.Exploration of use of best animal models and their predictive value for a human vaccine.Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.Efforts targeted at a universal coronavirus vaccine.Efforts to develop animal models and standardize challenge studies.Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers.Approaches to evaluate risk for enhanced disease after vaccination.Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]",

        "What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus? Resources to support skilled nursing facilities and long term care facilities.Mobilization of surge medical staff to address shortages in overwhelmed communities.Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies.Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patientsOutcomes data for COVID-19 after mechanical ventilation adjusted for age.Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.Guidance on the simple things people can do at home to take care of sick people and manage disease.Oral medications that might potentially work.Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials.Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials.Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)",

        "What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions? Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.Methods to control the spread in communities, barriers to compliance and how these vary among different populations..Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay.",

        "What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)? How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.Policies and protocols for screening and testing.Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.Technology roadmap for diagnostics.Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.Coupling genomics and diagnostic testing on a large scale.Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.",

        "Research Question At the time of writing, COVID-19 has spread to at least 114 countries. With viral flu, there are often geographic variations in how the disease will spread and if there are different variations of the virus in different areas. We’d like to explore what the literature and data say about this through this Task.Are there geographic variations in the rate of COVID-19 spread? Are there geographic variations in the mortality rate of COVID-19? Is there any evidence to suggest geographic based virus mutations?",

         "What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response? Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019.Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight.Efforts to support sustained education, access, and capacity building in the area of ethics.Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures).Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.",

         "What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity? Methods for coordinating data-gathering with standardized nomenclature.Sharing response information among planners, providers, and others.Understanding and mitigating barriers to information-sharing.How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).Integration of federal/state/local public health surveillance systems.Value of investments in baseline public health response infrastructure preparednessModes of communicating with target high-risk populations (elderly, health care workers).Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).Communication that indicates potential risk of disease to all population groups.Misunderstanding around containment and mitigation.Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.Measures to reach marginalized and disadvantaged populations.Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"]

       

topics = []

dominant_topic = []

prob_scores =[]

for i in range(len(tasks)):

    dominant, topic, prob_score = predict_topic(text = tasks[i])

    topic = topic[0]

    topics.append(topic)

    dominant_topic.append(dominant)

    prob_scores.append(prob_score)

    

    print (topic)

    print(dominant)

    print (prob_score)





zippedList =  zip(tasks,topics,dominant_topic,prob_scores)

li_result = list(zippedList)



tasks_df = pd.DataFrame(li_result, columns = ['tasks' , 'topic_number', 'dominant_topic','prob_scores'])

tasks_df

meta_df['dominant_topic'] = None

meta_df["dominant_topic"]= meta_df['processed_abstract'].apply(apply_predict_topic)

meta_df.head()

meta_df.to_csv('meta_df_output_topics.csv')

tasks_df.to_csv('tasks_associated_topics.csv')
meta_df.head(16)


tasks_df['related_articles_cord_uid'] = None

for i in range(len(tasks_df)):

    uid = []

    for j in range(len(meta_df)):

        if (tasks_df.dominant_topic[i] == meta_df.dominant_topic[j]):

            uid.append(meta_df.cord_uid[j])

    tasks_df.related_articles_cord_uid[i] =   uid

    

tasks_df
tasks_df.to_csv('submission.csv')
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")

df_topic_distribution.columns = ['Topic Num', 'Num Documents']

df_topic_distribution
pyLDAvis.enable_notebook()

panel = pyLDAvis.sklearn.prepare(best_lda_model, dtm_tf, tf_vectorizer, mds='tsne')

panel
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=7, angle=.99, init='pca')



document_topic_matrix = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)



# 13-D -> 2-D

tsne_lda = tsne_model.fit_transform(document_topic_matrix) # doc_topic is document-topic matrix from LDA or GuidedLDA

tsne_lda
# getting high prob value for each row



maxValuesObj = document_topic_matrix.max(axis=1)

 

print('Maximum value in each row : ')

print(maxValuesObj)





# Joining max value into doc term matrix



document_topic_matrix['dominant_pbb_value'] = maxValuesObj



# getting Dominant topic



dominant_topic = np.argmax(document_topic_matrix.values, axis=1)

document_topic_matrix["dominant_topic"] = dominant_topic



#Joining tsne_lda into existing dataframe



document_topic_matrix['x_tsne'] = tsne_lda[:,0]

document_topic_matrix['y_tsne'] = tsne_lda[:,1]

document_topic_matrix
document_topic_matrix.to_csv('dtm_tf_prob_topics.csv')
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", 1)



# plot

sns.scatterplot(tsne_lda[:,0], tsne_lda[:,1], palette=palette)



plt.title("t-SNE Covid-19 Articles")

plt.savefig("t-sne_covid19.png")

plt.show()
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(meta_df["dominant_topic"])))



# plot

sns.scatterplot(tsne_lda[:,0], tsne_lda[:,1],hue=meta_df["dominant_topic"], legend='full', palette=palette) #

plt.title("t-SNE Covid-19 Articles - Clustered")

plt.savefig("t-sne_covid19_label.png")

plt.show()
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column

from bokeh.models import RadioButtonGroup

from bokeh.models import TextInput

from bokeh.layouts import gridplot

from bokeh.models import Div

from bokeh.models import Paragraph

from bokeh.layouts import column, widgetbox



output_notebook()

y_labels = dominant_topic



# data sources

source = ColumnDataSource(data=dict(

    x= tsne_lda[:,0],  

    y= tsne_lda[:,1],

   # x_backup = tsne_lda[:,0],

   # y_backup = tsne_lda[:,1],

    desc= y_labels, 

    titles= meta_df['title'],

    authors = meta_df['authors'],

    journal = meta_df['journal'],

    abstract = meta_df['abstract'],

    topic = meta_df["dominant_topic"],

    labels = [x for x in meta_df.dominant_topic]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

    ("Topic", "@topic")

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=1000, plot_height=1000, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, LDA output", 

           toolbar_location="right")





# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')



show(p)
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=3, verbose=1, random_state=7, angle=.99, init='pca')



document_topic_matrix = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)



# 13-D -> 2-D

tsne_lda = tsne_model.fit_transform(document_topic_matrix) # doc_topic is document-topic matrix from LDA or GuidedLDA



# getting high prob value for each row



maxValuesObj = document_topic_matrix.max(axis=1)

 

print('Maximum value in each row : ')

print(maxValuesObj)





# Joining max value into doc term matrix





document_topic_matrix['dominant_pbb_value'] = maxValuesObj

document_topic_matrix



# getting Dominant topic



dominant_topic = np.argmax(document_topic_matrix.values, axis=1)

document_topic_matrix["dominant_topic"] = dominant_topic



#Joining tsne_lda into existing dataframe



document_topic_matrix['x_tsne'] = tsne_lda[:,0]

document_topic_matrix['y_tsne'] = tsne_lda[:,1]

document_topic_matrix['z_tsne'] = tsne_lda[:,2]





%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=tsne_lda[:,0],

    ys=tsne_lda[:,1],

    zs=tsne_lda[:,2],

    c=dominant_topic, 

    cmap='tab10'

)

ax.set_xlabel('tsne-one')

ax.set_ylabel('tsne-two')

ax.set_zlabel('tsne-three')

plt.title("tSNE Covid-19 Articles (3D) - LDA")

plt.savefig("tSNE_Covid19_label_3d.png")

plt.show()


