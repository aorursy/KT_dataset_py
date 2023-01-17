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
#Read and other operations

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



import pickle
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
csv_data_bio=pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

csv_data_comm=pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

csv_data_noncomm=pd.read_csv("/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")
related_articles_index_file_bio = pd.read_csv("/kaggle/input/article-ref-bio-task5/article_indexes_bio_task5.csv",sep=";")

related_articles_index_file_comm = pd.read_csv("/kaggle/input/article-ref-comm-task5/article_indexes_comm_task5.csv")

related_articles_index_file_noncomm = pd.read_csv("/kaggle/input/article-ref-noncomm-task5/article_indexes_noncomm_task5.csv")
###bio

#for text

#preprocessed_text_bio = csv_data_bio['text'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



#for abstract

preprocessed_abs_bio = csv_data_bio['abstract'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



#for text

#preprocessed_text_comm = csv_data_comm['text'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



###comm

#for abstract

preprocessed_abs_comm = csv_data_comm['abstract'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



###noncomm

#for text

#preprocessed_text_noncomm = csv_data_noncomm['text'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))



#for abstract

preprocessed_abs_noncomm = csv_data_noncomm['abstract'].map(lambda x: cleanData(x,custom_replace = False,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True))
#bio

with open('/kaggle/input/vectorizer-bio-task5/article_bio_vectorizer_task5.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-bio-task5/abstract_bio_vectorizer_task5.pickle', 'rb') as tf_abs:

    tfidf_vect_abs_bio = pickle.load(tf_abs)

    

#vect_scores_text_bio = tfidf_vect_text.fit_transform(preprocessed_text_bio)

vect_scores_abs_bio = tfidf_vect_abs_bio.fit_transform(preprocessed_abs_bio)



#comm

with open('/kaggle/input/vectorizer-comm-task5/article_comm_vectorizer_task5.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-comm-task5/abstract_comm_vectorizer_task5.pickle', 'rb') as tf_abs:

    tfidf_vect_abs_comm = pickle.load(tf_abs)

    

#vect_scores_text_comm = tfidf_vect_text.fit_transform(preprocessed_text_comm)

vect_scores_abs_comm = tfidf_vect_abs_comm.fit_transform(preprocessed_abs_comm)



#noncomm

with open('/kaggle/input/vectorizer-noncomm-task5/article_noncomm_vectorizer_task5.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-noncomm-task5/abstract_noncomm_vectorizer_task5.pickle', 'rb') as tf_abs:

    tfidf_vect_abs_noncomm = pickle.load(tf_abs)

    

#vect_scores_text_noncomm = tfidf_vect_text.fit_transform(preprocessed_text_noncomm)

vect_scores_abs_noncomm = tfidf_vect_abs_noncomm.fit_transform(preprocessed_abs_noncomm)

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
def save_related_articles_extract_abstract(related_articles_file,sub_cat_no,inp_csv_data,tfidf_scores):

    '''read '''

    art_idx = ast.literal_eval(related_articles_file.iloc[sub_cat_no,1])

    #final_articles = [csv_data_bio.iloc[idx,5] for idx in art_idx]

    final_abstracts = [inp_csv_data.iloc[idx,4] for idx in art_idx]

    

    question = level1_sub_questions[sub_cat_no]

    question_processed = [cleanData(x,custom_replace = True,lower_case = True,remove_punc = False,remove_stops=True,custom_stop=False,stemming=False, lemmatization = True,remove_numeric=True) for x in [question]]

    #print("question_processed: ",question_processed)

    q_vector = tfidf_scores.transform(question_processed)

    sent_from_abstract = ["abstract not present" if pd.isnull(abst) else "".join(s for s in retriev_sentences(q_vector,[abst],1,tfidf_scores)) for abst in final_abstracts]

    #print("sent_from_abstract: ",sent_from_abstract)

    

    article_title= [inp_csv_data.iloc[idx,1] for idx in art_idx]

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
level1_sub_questions = ["1. Resources to support skilled nursing facilities and long term care facilities.",

                   "2. Mobilization of surge medical staff to address shortages in overwhelmed communities",

                   "3. Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies",

                   "4. Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",

                   "5. Outcomes data for COVID-19 after mechanical ventilation adjusted for age.",

                   "6. Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.",

                   "7. Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.",

                   "8. Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.",

                   "9. Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.",

                   "10. Guidance on the simple things people can do at home to take care of sick people and manage disease.",

                   "11. Oral medications that might potentially work.",

                   "12. Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.",

                   "13. Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.",

                   "14. Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",

                   "15. Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials",

                   "16. Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"]
sub_categories_keywords = [["support nursing facilities"],

                   ["mobilization of surge medical", "shortages in overwhelmed communities"],

                   ["mortality data","acute respiratory distress syndrome","ards","viral etiologies"],

                   ["extracorporeal membrane oxygenation", "ecmo"],

                   ["mechanical ventilation"],

                   ["extrapulmonary manifestations","cardiomyopathy","cardiac arrest"],

                   ["regulatory standard", "eua", "clia", "care to crisis standard" ],

                   ["elastomeric respirator", "n95 masks"],

                   ["telemedicine practice", "barriers" , "faciitators"],

                   ["home care"],

                   ["oral medication"],

                   ["use of ai"],

                   ["best practice", "critical challenge", "innovative solution", "workforce protection", "workforce allocation", "community-based support resource", "payment", "supply chain management"],

                   ["history of disease", "public health intervention", "prevention control", "clinical trials"],

                   ["core clinical outcome"],

                   ["adjunctive", "supportive intervention", "steroid", "high flow oxygen"]]
def main(sub_topic):

    pd.options.display.max_colwidth = 200

    

    selected_sub_topic = level1_sub_questions[int(sub_topic)-1]

    print("selected sub topic: ",selected_sub_topic)

    related_articles_df_bio = save_related_articles_extract_abstract(related_articles_index_file_bio,int(sub_topic)-1,csv_data_bio,tfidf_vect_abs_bio)

    related_articles_df_comm = save_related_articles_extract_abstract(related_articles_index_file_comm,int(sub_topic)-1,csv_data_comm,tfidf_vect_abs_comm)

    related_articles_df_noncomm = save_related_articles_extract_abstract(related_articles_index_file_noncomm,int(sub_topic)-1,csv_data_noncomm,tfidf_vect_abs_noncomm)

    print("Related articles extracted")

    #################

    print(" ")

    related_articles_df_bio = related_articles_df_bio.style.set_properties(**{'text-align': 'left'})

    related_articles_df_bio.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    print("----Below are the related articles extracted and corresponding abstracts from 'bio' data source----")

    display(related_articles_df_bio)

    #################

    print(" ")

    related_articles_df_comm = related_articles_df_comm.style.set_properties(**{'text-align': 'left'})

    related_articles_df_comm.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    print("----Below are the related articles extracted and corresponding abstracts from 'comm' data source----")

    display(related_articles_df_comm)

    #################

    print(" ")

    related_articles_df_noncomm = related_articles_df_noncomm.style.set_properties(**{'text-align': 'left'})

    related_articles_df_noncomm.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    print("----Below are the related articles extracted and corresponding abstracts from 'noncomm' data source----")

    display(related_articles_df_noncomm)
#"1. Resources to support skilled nursing facilities and long term care facilities.",

main("1")         
# "2. Mobilization of surge medical staff to address shortages in overwhelmed communities",

main("2")       
#"3. Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies",

main("3") 
#"4. Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",

main("4")
#"5. Outcomes data for COVID-19 after mechanical ventilation adjusted for age.",

main("5")
#"6. Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.",

main("6")
#"7. Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.",

main("7")
#"8. Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.",

main("8")    
#"9. Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.",

main("9")
#"10. Guidance on the simple things people can do at home to take care of sick people and manage disease.",

main("10")
#"11. Oral medications that might potentially work.",

main("11")
#"12. Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.",

main("12")
#"13. Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.",

main("13")                
#"14. Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",

main("14")             
#"15. Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials",

main("15")
#"16. Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"

main("16")