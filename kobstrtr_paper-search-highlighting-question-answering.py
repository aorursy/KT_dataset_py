from IPython.utils import io

!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import numpy as np 

import pandas as pd



from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation



import scispacy

import spacy

import en_core_sci_lg

import tensorflow as tf

import torch

from transformers import *



from scipy.spatial.distance import jensenshannon



import joblib



from IPython.display import HTML, display



from ipywidgets import interact, Layout, HBox, VBox, Box

import ipywidgets as widgets

from IPython.display import clear_output



from tqdm import tqdm

from os.path import isfile



import seaborn as sb

import matplotlib.pyplot as plt

plt.style.use("dark_background")
df = pd.read_csv('../input/cord-19-create-dataframe/cord19_df.csv')
all_texts = df.body_text
filepath = '../input/topic-modeling-finding-related-articles/'
nlp = en_core_sci_lg.load(disable=["tagger", "parser", "ner"])

nlp.max_length = 2000000
def spacy_tokenizer(sentence):

    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]
# Load vectorizer

vectorizer = joblib.load(filepath + 'vectorizer.csv')

data_vectorized = joblib.load(filepath + 'data_vectorized.csv')
# Load LDA Model

lda = joblib.load(filepath + 'lda.csv') 
# Load previously computed topic distribution

doc_topic_dist = pd.read_csv(filepath + 'doc_topic_dist.csv')  
is_covid19_article = df.body_text.str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')
def get_k_nearest_docs(doc_dist, k=5, lower=1950, upper=2020, only_covid19=False, get_dist=False):

    '''

    doc_dist: topic distribution (sums to 1) of one article

    

    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 

    '''

    

    relevant_time = df.publish_year.between(lower, upper)

    

    if only_covid19:

        temp = doc_topic_dist[relevant_time & is_covid19_article]

        

    else:

        temp = doc_topic_dist[relevant_time]

         

    distances = temp.apply(lambda x: jensenshannon(x, doc_dist), axis=1)

    k_nearest = distances[distances != 0].nsmallest(n=k).index

    

    if get_dist:

        k_distances = distances[distances != 0].nsmallest(n=k)

        return k_nearest, k_distances

    else:

        return k_nearest
task1 = ["Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",

"Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",

"Seasonality of transmission.",

"Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",

"Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",

"Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",

"Natural history of the virus and shedding of it from an infected person",

"Implementation of diagnostics and products to improve clinical processes",

"Disease models, including animal models for infection, disease and transmission",

"Tools and studies to monitor phenotypic change and potential adaptation of the virus",

"Immune response and immunity",

"Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

 "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

"Role of the environment in transmission"]



task2 = ['Data on potential risks factors',

'Smoking, pre-existing pulmonary disease',

'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

'Neonates and pregnant women',

'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors', 

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Susceptibility of populations',

'Public health mitigation measures that could be effective for control']

def scoresLDA(paragraphs, query):

    query_vectorized = vectorizer.transform([query])

    query_topic_dist = lda.transform(query_vectorized)[0]



    paragraphs_vectorized = vectorizer.transform(paragraphs)

    paragraphs_topic_dist = lda.transform(paragraphs_vectorized)

    

    dists = [jensenshannon(paragraph_topic_dist, query_topic_dist) for paragraph_topic_dist in paragraphs_topic_dist]

    min_dist, max_dist = min(dists), max(dists)

    

    return [((dist-min_dist) / (max_dist - min_dist))**8 for dist in dists]
def printMatch(query, score_fn):

    query_vectorized = vectorizer.transform([query])

    query_topic_dist = lda.transform(query_vectorized)[0]



    recommended = get_k_nearest_docs(query_topic_dist, 1, 0, 20000, True)

    article = all_texts[recommended[0]]

    recommended = df.iloc[recommended[0]]



    paragraphs = article.split("\n")



    html = '<b>Query:</b><br />'

    html += query

    html += '<br /><br />'



    html += '<b>Best match:</b><br />'

    html += '<a href="' + recommended['url'] + '" target="_blank">'+ recommended['title'] + '</a>'

    html += '<br/><br/>'



    html += '<b>Article (important content is highlighted in red):</b>'

    

    scores = score_fn(paragraphs, query)

    for paragraph, score in zip(paragraphs, scores):

        color = 'rgb({},0,{})'.format(int(255 * score), int(255 * (1-score)))

        html += '<p style="color:'+color+';">'+paragraph+'</p>'



    display(HTML(html))
query = task1[0]

printMatch(query, scoresLDA)
query = task2[1]

printMatch(query, scoresLDA)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
# BERT is not erudite (Ilsebill is split into subword tokens → BERT didn't read Günther Grass)

tokenizer.tokenize("Ilsebill salzte nach.")
def encode_task(pair):

    assert len(pair) == 2

    # automatically takes care of adding [CLS] at the start and [SEP] in between and at the end

    # and of building the attention mask (highlighting which tokens should be masked out during the self attention)

    # and of building the token_type_id mask (highlighting which tokens belong to the question and to the paragraph)

    encoding = tokenizer.encode_plus(pair[0], pair[1], add_special_tokens=True)

    to_tensor = lambda x: tf.constant(x)[None, :]

    return {

        "inputs": to_tensor(encoding['input_ids']),

        # useless for this exercise, but BERT still wants them. Normally tells BERT which tokens to mask out during the attention

        "attention_mask": to_tensor(encoding['attention_mask']),

        # tells BERT which tokens belong to the first and which to the second sequence

        "token_type_ids": to_tensor(encoding['token_type_ids'])

    }



# 101 is the start ([CLS]) token, 102 is the separator

# the token mask is 0 for the first sequence and 1 for the second

encode_task(["this is one sentence", "This is another sentence"])
def bert_sim_score(encoding):

    output = model(**encoding)

    # the model outputs the logits signifying the probability of the two tasks being sequiturs

    return tf.math.softmax(output[0]).numpy()[0,0]



# some simple tests

lettuce = "Lettuce (Lactuca sativa) is an annual plant of the daisy family, Asteraceae. It is most often grown as a leaf vegetable, but sometimes for its stem and seeds. Lettuce is most often used for salads, although it is also seen in other kinds of food, such as soups, sandwiches and wraps; it can also be grilled."

more_lettuce= "One variety, the woju (t:萵苣/s:莴苣), or asparagus lettuce (Celtuce), is grown for its stems, which are eaten either raw or cooked. In addition to its main use as a leafy green, it has also gathered religious and medicinal significance over centuries of human consumption. Europe and North America originally dominated the market for lettuce, but by the late 20th century the consumption of lettuce had spread throughout the world. World production of lettuce and chicory for calendar year 2017 was 27 million tonnes, 56% of which came from China."

lasso = "In statistics and machine learning, lasso (least absolute shrinkage and selection operator; also Lasso or LASSO) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces"



print('lettuce and lettuce, should be true:', bert_sim_score(encode_task([lettuce, more_lettuce])))

print('lettuce and LASSO, doesnt mix well:', bert_sim_score(encode_task([lettuce, lasso])))
def scoresBERT(paragraphs, query):

    score = lambda q, p: bert_sim_score(encode_task([q, p]))

    return [score(query, para) for para in paragraphs]
query = task1[0]

printMatch(query, scoresBERT)
query = task2[1]

printMatch(query, scoresBERT)
# here we switch to pytorch since pretrained SciBERT is not available in TensorFlow

# this model was also finetuned on the SQuAD dataset

# as SQuAD V2 was used (which introduced unanswerable questions) the model should be able to not give back and answer if there is none

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/scibert_scivocab_uncased_squad_v2")

model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/scibert_scivocab_uncased_squad_v2")
def encode_qa(question, text):

    encoding = tokenizer.encode_plus(question, text_pair = text, max_length=512)

    # let BERT know when the second sequence starts by building the token_type embedding

    return encoding



# 101 is the start ([CLS]) token, 102 is the separator

# the token mask is 0 for the first sequence and 1 for the second

encode_qa("Why did the chicken cross the road", "To get to the other side")
def BERT_qa_answer(encoding):

    input_ids = encoding['input_ids']

    to_tensor = lambda x: torch.tensor([x])

    start_scores, end_scores = model(to_tensor(input_ids), 

                                     token_type_ids=to_tensor(encoding['token_type_ids']),

                                     attention_mask=to_tensor(encoding['attention_mask']))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1])

    # get back the words from subword tokens by merging them

    return answer.replace(' ##', '')



question = "What increased the odds of in-hospital death?"

text = "191 patients (135 from Jinyintan Hospital and 56 from Wuhan Pulmonary Hospital) were included in this study, of whom 137 were discharged and 54 died in hospital. 91 (48%) patients had a comorbidity, with hypertension being the most common (58 [30%] patients), followed by diabetes (36 [19%] patients) and coronary heart disease (15 [8%] patients). Multivariable regression showed increasing odds of in-hospital death associated with older age (odds ratio 1·10, 95% CI 1·03–1·17, per year increase; p=0·0043), higher Sequential Organ Failure Assessment (SOFA) score (5·65, 2·61–12·23; p<0·0001), and d-dimer greater than 1 μg/mL (18·42, 2·64–128·55; p=0·0033) on admission. Median duration of viral shedding was 20·0 days (IQR 17·0–24·0) in survivors, but SARS-CoV-2 was detectable until death in non-survivors. The longest observed duration of viral shedding in survivors was 37 days."

print(question, 'answer:', BERT_qa_answer(encode_qa(question, text)))

question = "Why did the chicken cross the road?"

text = '"Why did the chicken cross the road?" is a common riddle joke, with the answer being "To get to the other side". It is an example of anti-humor, in that the curious setup of the joke leads the listener to expect a traditional punchline, but they are instead given a simple statement of fact. "Why did the chicken cross the road?" has become iconic as an exemplary generic joke to which most people know the answer, and has been repeated and changed numerous times over the course of history.'

# BERT was trained to put both start and end on [CLS] if it didn't find an answer

# Regular BERT might have been able to answer this question since we took it from Wikipedia

# But since SciBERT was trained on semanticscholar, it might not have the appropriate knowledge

print(question, 'answer:', BERT_qa_answer(encode_qa(question, text)))
def printAnswers(question):

    query_vectorized = vectorizer.transform([question])

    query_topic_dist = lda.transform(query_vectorized)[0]



    recommended = get_k_nearest_docs(query_topic_dist, 5, 0, 20000, True)

    articles = [all_texts[i] for i in recommended]

    recommendeds = [df.iloc[i] for i in recommended]

    

    

    html = '<b>Question:</b><br />'

    html += question

    html += '<br /><br />'

    

    for article, recommended in zip(articles, recommendeds):

        answered = False

        paragraphs = article.split("\n")



        answers = [

            BERT_qa_answer(encode_qa(question, paragraph)) for paragraph in paragraphs

        ]

        answers = [answer for answer in answers 

                   if answer and "[CLS]" not in answer and "[SEP]" not in answer]

        if answers:

            html += '<b>Matched document:</b><br />'

            html += '<a href="' + recommended['url'] + '" target="_blank">'+ recommended['title'] + '</a>'

            html += '<br/><br/>'



            html += '<b>Extracted answers:</b><br\>'

            for answer in answers:

                html += '<div>' + answer + '</div>'

            html += '<br/><br/>'

    display(HTML(html))
printAnswers("Which risk factors exist?")
printAnswers("What is the average incubation period of the disease?")
printAnswers("What increased the odds of in-hospital death?")