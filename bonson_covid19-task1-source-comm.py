# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



#Data cleaning

import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from string import punctuation

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



#tokenization

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize



#cosine similarity

from sklearn.metrics.pairwise import cosine_similarity



#Get nearest match word in a list for a given word

import difflib



#tfidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



import csv



#convert string to list

import ast
my_stopwords = stopwords.words('english')



def cleanData(text, custom_replace = False,lower_case = False,remove_punc = False,remove_stops = False, custom_stop=False,stemming = False, lemmatization = False,remove_numeric=False):



    custom_stopwords = []

    custom_replace = ["Abstract","CC-BY-NC-ND 4.0","CC-BY-ND 4.0","International license is made available under a","author/funder, who has granted medRxiv a license to display the preprint in perpetuity","is the (which was not peer-reviewed)","The copyright holder for this preprint"]

    

    txt = str(text)

    

    # Remove urls and emails

    txt = re.sub(r'\b[https|http]+:[\S]+', ' ', txt, flags=re.MULTILINE)

    

    

    #convert to lower case

    if lower_case:

        txt = txt.lower()

        

    #rremove custom replace

    if custom_replace:

        for cr in custom_replace:

             txt = txt.replace(cr.lower(),"")

    

    # Remove punctuation from text

    if remove_punc:

        txt = ''.join([c for c in text if c not in punctuation])



    if stemming:

        st = PorterStemmer()

        txt = " ".join([st.stem(w) for w in txt.split()])

    

    if lemmatization:

        wordnet_lemmatizer = WordNetLemmatizer()

        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

    if remove_stops:

        txt = " ".join([w for w in txt.split() if w not in my_stopwords])

    if custom_stop:

        txt = " ".join([w for w in txt.split() if w not in custom_stopwords])

    if remove_numeric:

        txt = ''.join([i for i in txt if not i.isdigit()])



    return txt
csv_data_comm=pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")
#for text

preprocessed_text_comm = csv_data_comm['text'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



#for abstract

preprocessed_abs_comm = csv_data_comm['abstract'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))
related_articles_index_file = pd.read_csv("/kaggle/input/article-ref-comm-task1/article_indexes_comm_task1.csv")
import pickle

with open('/kaggle/input/vectorizer-comm-task1/article_comm_vectorizer_task1.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-comm-task1/abstract_comm_vectorizer_task1.pickle', 'rb') as tf_abs:

    tfidf_vect_abs = pickle.load(tf_abs)

    

vect_scores_text = tfidf_vect_text.fit_transform(preprocessed_text_comm)

vect_scores_abs = tfidf_vect_abs.fit_transform(preprocessed_abs_comm)
def retriev_sentences(ques_v,list_text,num_of_sentences,tfidf_abs):

    answer_sentences = []

    for f_article in list_text:

        org_art_cleaned = cleanData(f_article,custom_replace = True,lower_case = True)

        org_art_sent = sent_tokenize(org_art_cleaned)

        article_sentences_processed = [cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True) for x in org_art_sent]

        

        #sentence to vectors

        sent_vector_score = []

        for sent_idx in range(0,len(org_art_sent)):

            #print(sent_idx)

            sent_vector= tfidf_abs.transform([article_sentences_processed[sent_idx]])

            sent_tmp_score = cosine_similarity(ques_v.toarray(),sent_vector.toarray())

            sent_vector_score.append(sent_tmp_score[0][0])     

        high_scored_sentence = sorted(range(len(sent_vector_score)), key=lambda i: sent_vector_score[i])[-num_of_sentences:]



        final_sentences = "".join([org_art_sent[sent] for sent in high_scored_sentence])

        

        answer_sentences.append(final_sentences)

        

        #print(sent_vector_score)

        #print(high_scored_sentence)

    return answer_sentences
def save_related_articles_extract_abstract(sub_cat_no):

    '''read '''

    art_idx = ast.literal_eval(related_articles_index_file.iloc[sub_cat_no,1])

    final_articles = [csv_data_comm.iloc[idx,5] for idx in art_idx]

    final_abstracts = [csv_data_comm.iloc[idx,4] for idx in art_idx]

    

    question = level1_sub_questions[sub_cat_no]

    question_processed = [cleanData(x,custom_replace = True,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True) for x in [question]]

    #print("question_processed: ",question_processed)

    q_vector = tfidf_vect_abs.transform(question_processed)

    sent_from_abstract = ["abstract not present" if pd.isnull(abst) else "".join(s for s in retriev_sentences(q_vector,[abst],1,tfidf_vect_abs)) for abst in final_abstracts]

    #print("sent_from_abstract: ",sent_from_abstract)

    

    article_title= [csv_data_comm.iloc[idx,1] for idx in art_idx]

    article_number = list(range(1,len(art_idx)+1))

    df = pd.DataFrame(list(zip(article_number, article_title,sent_from_abstract)), 

               columns =['Article Number', 'Article Title','Abstract'])

    #print(art_idx)

    return df
level1_questions = ["1. What is known about transmission, incubation, and environmental stability?",

                   "2. What do we know about COVID-19 risk factors?",

                   "3. What do we know about virus genetics, origin, and evolution?",

                   "4. What do we know about vaccines and therapeutics?",

                   "5. What do we know about non-pharmaceutical interventions?",

                   "6. What has been published about medical care?",

                   "7. Sample task with sample submission",

                   "8. What do we know about diagnostics and surveillance?",

                   "9. What has been published about information sharing and inter-sectoral collaboration?",

                   "10. What has been published about ethical and social science considerations?"]
level1_sub_questions = ["1. Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",

                   "2. Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",

                   "3. seasonality of transmission.",

                   "4. physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",

                   "5. persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",

                   "6. persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",

                   "7. Natural history of the virus and shedding of it from an infected person",

                   "8. implementation of diagnostics and products to improve clinical processes",

                   "9. Disease models, including animal models for infection, disease and transmission",

                   "10. Tools and studies to monitor phenotypic change and potential adaptation of the virus",

                   "11. Immune response and immunity",

                    "12. Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

                    "13. Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

                    "14. Role of the environment in transmission"]
sub_categories_keywords = [["incubation periods", "health status", "contagious"],

                   ["Asymptomatic shedding", "transmission", "children"],

                   ["seasonality", "transmission"],

                   ["physical science", "charge distribution", "adhesion to hydrophilic", "phobic surfaces", "environmental survival","decontamination efforts","viral shedding"],

                   ["persistence and stability", "multitude of substrates","nasal discharge","sputum", "urine", "fecal matter", "blood"],

                   ["persistence of virus", "copper", "stainless steel", "plastic"],

                   ["history of virus", "shedding"],

                   ["diagnostic and products","clinical processes"],

                   ["disease models", "animal models","disease and transmission"],

                   ["tools and studies","phenotypic change", "adaptation of the virus"],

                   ["immune response" , "immunity"], 

                   ["movement control strategies", "prevent secondary transmission"],

                   [ "personal protective equipment", "reduce risk of transmission" , "community settings"],

                   ["environment in transmission"]]
def main(sub_topic):

    pd.options.display.max_colwidth = 200

    

    selected_sub_topic = level1_sub_questions[int(sub_topic)-1]

    print("selected sub topic: ",selected_sub_topic)

    related_articles_df = save_related_articles_extract_abstract(int(sub_topic)-1)

    print("Related articles extracted")

    print(" ")

    related_articles_df = related_articles_df.style.set_properties(**{'text-align': 'left'})

    related_articles_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    print("----Below are the related articles extracted and corresponding abstracts----")

    display(related_articles_df)
#"1. Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",

main("1")
#"2. Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",

main("2")
#"3. seasonality of transmission.",

main("3")
#"4. physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",

main("4")
# "5. persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",

main("5")
#"6. persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",

main("6")
#"7. Natural history of the virus and shedding of it from an infected person",

main("7")
#"8. implementation of diagnostics and products to improve clinical processes",

main("8")
#"9. Disease models, including animal models for infection, disease and transmission",

main("9")

#"10. Tools and studies to monitor phenotypic change and potential adaptation of the virus",

main("10")
#"11. Immune response and immunity",

main("11")
#"12. Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

main("12")
#"13. Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

main("13")
#"14. Role of the environment in transmission"

main("14")