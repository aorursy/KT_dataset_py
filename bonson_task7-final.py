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
related_articles_index_file_bio = pd.read_csv("/kaggle/input/article-ref-bio-task7/article_indexes_bio_task7.csv",sep=";")

related_articles_index_file_comm = pd.read_csv("/kaggle/input/article-ref-comm-task7/article_indexes_comm_task7.csv")

related_articles_index_file_noncomm = pd.read_csv("/kaggle/input/article-ref-noncomm-task7/article_indexes_noncomm_task7.csv")
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

with open('/kaggle/input/vectorizer-bio-task7/article_bio_vectorizer_task7.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-bio-task7/abstract_bio_vectorizer_task7.pickle', 'rb') as tf_abs:

    tfidf_vect_abs_bio = pickle.load(tf_abs)

    

#vect_scores_text_bio = tfidf_vect_text.fit_transform(preprocessed_text_bio)

vect_scores_abs_bio = tfidf_vect_abs_bio.fit_transform(preprocessed_abs_bio)



#comm

with open('/kaggle/input/vectorizer-comm-task7/article_comm_vectorizer_task7.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-comm-task7/abstract_comm_vectorizer_task7.pickle', 'rb') as tf_abs:

    tfidf_vect_abs_comm = pickle.load(tf_abs)

    

#vect_scores_text_comm = tfidf_vect_text.fit_transform(preprocessed_text_comm)

vect_scores_abs_comm = tfidf_vect_abs_comm.fit_transform(preprocessed_abs_comm)



#noncomm

with open('/kaggle/input/vectorizer-noncomm-task7/article_noncomm_vectorizer_task7.pickle', 'rb') as tf_art:

    tfidf_vect_text = pickle.load(tf_art)

    

with open('/kaggle/input/vectorizer-noncomm-task7/abstract_noncomm_vectorizer_task7.pickle', 'rb') as tf_abs:

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
level1_sub_questions = ["1. How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).",

                   "2. Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.",

                   "3. Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.",

                   "4. National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).",

                   "5. Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.",

                   "6. Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).",

                   "7. Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.",

                   "8. Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.",

                   "9. Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.",

                   "10. Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.",

                    "11. Policies and protocols for screening and testing.",

                    "12. Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.",

                    "13. Technology roadmap for diagnostics.",

                    "14. Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.",

                    "15. New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.",

                    "16. Coupling genomics and diagnostic testing on a large scale.",

                    "17. Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.",

                    "18. Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.",

                    "19. One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."]
sub_categories_keywords = [["recommendations on mitigation", "Denominators for testing", "sampling method", "asymptomatic disease", "serosurveys", "early detection"],

                   ["diagnostic platform", "surveillance platform"],

                   ["local expertise" , "operational issue", "legal issue", "ethical issue", "operational issue"],

                   ["national guidance", "best practice","universities and private laboratories"],

                   ["point-of-care test", "influenza test", "bed-side tests"],

                   ["targeted surveillance experiment", "local intervention"],

                   [ "assay development issue", "private sector"],

                   ["evolution of the virus", "genetic drift" , "mutation","surveillance scheme"],

                    ["latency issue", "pathogen", "biological sampling","environmental sampling"],

                    ["host response marker", "cytokines", "predict severe disease progression", "therapeutic intervention"],

                    ["protocols for screening and testing"],

                    ["mass testing", "swabs", "reagents"],

                    ["technology roadmap"],

                    ["future coalition", "accelerator models"],

                    ["new platform", "technology", "future disease"],

                    ["genomics and diagnostic"],

                    ["rapid sequencing","bioinformatics"],

                    [ "analytics", "naturally-occurring pathogen"],

                    ["surveillance of humans", "evolutionary hosts", "transmission hosts", "demographic risk", "occupational risk"]]
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
#"1. How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).",

main("1")  
#"2. Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.",

main("2")
#"3. Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.",

main("3")
#"4. National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).",

main("4")
#"5. Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.",

main("5")
#"6. Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).",

main("6")
#"7. Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.",

main("7")
# "8. Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.",

main("8")
#"9. Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.",

main("9")
#"10. Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.",

main("10")
#"11. Policies and protocols for screening and testing.",

main("11")
#"12. Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.",

main("12")
#"13. Technology roadmap for diagnostics.",

main("13")
#"14. Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.",

main("14")
#"15. New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.",

main("15")
#"16. Coupling genomics and diagnostic testing on a large scale.",

main("16")
#"17. Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.",

main("17")
#"18. Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.",

main("18")
#"19. One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."

main("19")