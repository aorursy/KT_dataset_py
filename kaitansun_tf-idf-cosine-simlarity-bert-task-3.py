# Import Packages

import os

import re

import nltk

import json

import torch

import nltk.corpus  

import pandas as pd

import numpy as np

from copy import deepcopy

from nltk.stem import PorterStemmer

from datetime import datetime

from datetime import timedelta

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

!pip install transformers

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer

from fuzzywuzzy import fuzz 

from tqdm import tqdm



nltk.download('punkt')

nltk.download('stopwords')



# Settings

pd.set_option('display.max_colwidth', None)
# Text Preprocessing `clean_sent()`----------------------------------------------------------------------

porter_stemmer = PorterStemmer()

def clean_sent(sentence):

    """

    Clean the sentence

    :param sentence: text to to be cleaned

    :return: text that has been cleaned

    """

    #nltk.FreqDist(words).most_common(10)

    stopwords = set(nltk.corpus.stopwords.words('english'))

    words = sentence.split()

    # Lowercase all words (default_stopwords are lowercase too)

    words = [word.lower() for word in words]

    #words = sentence

    words = [word for word in words if len(word) > 1]

    # Remove numbers

    words = [word for word in words if not word.isnumeric()]

    # Remove punctuation

    words = [word for word in words if word.isalpha()]

    # Remove stopwords

    words = [word for word in words if word not in stopwords]

    # Porter

    words = [porter_stemmer.stem(word) for word in words]

    #fdist = nltk.FreqDist(words_lc)   

    return " ".join(words)





## Data Load----------------------------------------------------------------------







def col_fill(col_questions,col_names,col_for_excerpt,may_not_have_num,index):

    """

    Get answers for multiple columns and append them back to target table (target_table_dic[index])

    

    :param col_questions: a list of string - questions, each corresponds to a specific column (excluding excerpt)

    :param col_names: a list of string - columns names in target table to which col_questions correspond

    :param col_for_excerpt: string - the most important question in col_questions, on which excerpt based

    :param may_not_have_num: a list of string - column names in col_names, to indicate columns that may not have digits (columns that not put in this list must have digits in its answer)

    :param index: question number in this task



    :return: target_table_dic[index] with content filled in designated columns 

    """

    col_questions_cleaned = [clean_sent(ques) for ques in col_questions]



    # go through papers

    for i in tqdm(range(relevant_paper_dic[index].shape[0])):



        #get sentences of each paper, preprocess it

        paper_sent=re.split(' \.|\.(?=[A-Z])|\. (?=[A-Z])|\n', relevant_paper_dic[index].text[i])

        

#         paper_sent=relevant_paper_dic[index].text[i].split(". ")

        cleaned_paper_sent = [clean_sent(t) for t in paper_sent]



        for num,q in enumerate(col_questions_cleaned):

            # define questions

            col_name=col_names[num]

            full_question= col_questions[num]



            # extract top 3 sentences/paper for a specific questions, join them together as a string

            lis=[]

            for sent in cleaned_paper_sent:

                lis.append(fuzz.ratio(q,sent) )

            top_3_idx = [item[0] for item in sorted(enumerate(lis), key=lambda x: x[1],reverse=True)[0:3]]

            #max_idx=max(enumerate(lis), key=lambda x: x[1])[0]

            string ='; \n'.join([paper_sent[idx] for idx in top_3_idx])



            # Get answers, delete those without a number 

            answer=answer_question(full_question,string)        

            if (not hasNumbers(answer)) and (col_name not in may_not_have_num):

                answer="" 

            if ("[CLS]" in answer) or ("[SEP]" in answer):

                answer = ""

            target_table_dic[index].loc[i,col_name]=answer



            #  Exerpt answer extract

            if col_name == col_for_excerpt:

                excerpt_ans=string

                for idx in top_3_idx:

                    if (answer in paper_sent[idx]) and (answer!=""):

                        excerpt_ans=paper_sent[idx]

                target_table_dic[index].loc[i,"Excerpt"]=excerpt_ans

    return target_table_dic[index]





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



def clean_pdf_files(file_list, keyword_list):

    nth_paper=0

    cleaned_files=[]

    for file in file_list:

        with open(file) as f:

            file=json.load(f)

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

        if(nth_paper%1000)==0:

            print(nth_paper)

        nth_paper=nth_paper+1



        has_keyword = False

        for keyword in keyword_list:

            if keyword in features[5]:

                has_keyword = True

                break

        if has_keyword == True:

            cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text',

                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    return clean_df







# BERT----------------------------------------------------------------------

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, answer_text):

    '''

    Takes a `question` string and an `answer_text` string (which contains the

    answer), and identifies the words within the `answer_text` that are the

    answer. Prints them out.

    '''

    # ======== Tokenize ========

    # Apply the tokenizer to the input text, treating them as a text-pair.

    input_ids = tokenizer.encode(question, answer_text,max_length=500

                                )



    # Report how long the input sequence is.

    #print('Query has {:,} tokens.\n'.format(len(input_ids)))



    # ======== Set Segment IDs ========

    # Search the input_ids for the first instance of the `[SEP]` token.

    sep_index = input_ids.index(tokenizer.sep_token_id)



    # The number of segment A tokens includes the [SEP] token istelf.

    num_seg_a = sep_index + 1



    # The remainder are segment B.

    num_seg_b = len(input_ids) - num_seg_a



    # Construct the list of 0s and 1s.

    segment_ids = [0]*num_seg_a + [1]*num_seg_b



    # There should be a segment_id for every input token.

    assert len(segment_ids) == len(input_ids)



    # ======== Evaluate ========

    # Run our example question through the model.

    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.

                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text



    # ======== Reconstruct Answer ========

    # Find the tokens with the highest `start` and `end` scores.

    answer_start = torch.argmax(start_scores)

    answer_end = torch.argmax(end_scores)

    

    

    # Get the string versions of the input tokens.

    tokens = tokenizer.convert_ids_to_tokens(input_ids)



    # Start with the first token.

    answer = tokens[answer_start]



    # Select the remaining answer tokens and join them with whitespace.

    for i in range(answer_start + 1, answer_end + 1):

        

        # If it's a subword token, then recombine it with the previous token.

        if tokens[i][0:2] == '##':

            answer += tokens[i][2:]

        

        # Otherwise, add a space then the token.

        else:

            answer += ' ' + tokens[i]

            

    s_scores = start_scores.detach().numpy().flatten()

    e_scores = end_scores.detach().numpy().flatten()



    return answer





# Similarity ----------------------------------------------------------------------

def calc_simlarity_score(question_list, text_list,threshold=None, top=None):

    if (threshold==None)  and  (top==None):

        raise ValueError("Parameter `threshold` and `top` cannot both be None")

    dic = {}

    tfidf = TfidfVectorizer()

    corpus_tfidf_matrix = tfidf.fit_transform(text_list)

    ques_tfidf_matrix = tfidf.transform(question_list)

    sim_matrix = cosine_similarity(corpus_tfidf_matrix, ques_tfidf_matrix)

    for ques_idx in range(sim_matrix.shape[1]):

        dic[ques_idx] = []

        if threshold != None:

            if (threshold>1) or (threshold <0):

                raise ValueError("Please enter a value from 0 to 1 for parameter `threshold`")

            for paper_idx in range(sim_matrix.shape[0]):

                score = sim_matrix[paper_idx, ques_idx]

                if score >= threshold:

                    dic[ques_idx].append((paper_idx, score))

            dic[ques_idx]=sorted(dic[ques_idx], key=lambda i: i[1], reverse=True)

        elif top != None:

            top_paper_idx_list = sorted(range(len(sim_matrix[:, ques_idx])), key=lambda i: sim_matrix[:,0][i], reverse=True)[:top]

            dic[ques_idx] = [(top_idx, sim_matrix[top_idx, ques_idx]) for top_idx in top_paper_idx_list]

    return dic, sim_matrix



# Retrieve relevant paper----------------------------------------------------------------------

def retrieve_paper(df, dic):

    df_dic={}

    for ques_idx in dic:

        new_df = df.iloc[[item[0] for item in dic[ques_idx]], :]

        new_df['score'] = [item[1] for item in dic[ques_idx]]

        new_df['question'] = questions[ques_idx]

        df_dic[ques_idx]=new_df.copy()

    return df_dic



# Determine if a string has a value----------------------------------------------------------------------

def hasNumbers(inputString):

     return any(char.isdigit() for char in inputString)

# Set parameters

path = '/kaggle/input/CORD-19-research-challenge/document_parses/pdf_json'

keyword_list = ['novel coronavirus', 'novel-coronavirus', 'coronavirus-2019', 

                'sars-cov-2', 'sarscov2', 'covid-19', 'covid19',

                '2019ncov', '2019-ncov', 'wuhan']



# Get list of file paths

file_list = [os.path.join(r, file)  for r, _, f in os.walk(path)  for file in f]



# Clean （This takes ~15 min）

clean_pdf_df = clean_pdf_files(file_list, keyword_list)
# Append additional info from metadata to main df

metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

clean_pdf_df = clean_pdf_df.merge(metadata[['sha', 'title', 'authors', 'abstract', 'doi', 'publish_time', 'journal']], 

                                  how ='left', left_on='paper_id', right_on='sha')



# Clean columns

clean_pdf_df['title_x'] = clean_pdf_df['title_x'].fillna(clean_pdf_df['title_y'])

clean_pdf_df['authors_x'] = clean_pdf_df['authors_x'].fillna(clean_pdf_df['authors_y'])

clean_pdf_df['abstract_x'] = clean_pdf_df['abstract_x'].fillna(clean_pdf_df['abstract_y'])

clean_pdf_df = clean_pdf_df.drop(['sha', 'title_y', 'authors_y', 'abstract_y'], axis=1)

clean_pdf_df = clean_pdf_df.rename(columns={'title_x': 'title', 'authors_x': 'authors', 'abstract_x': 'abstract'})
clean_pdf_df['text_cleaned'] = clean_pdf_df.apply(lambda row: clean_sent(row['text']), axis=1)
#clean_pdf_df.to_pickle(("./clean_pdf_df.pkl"))

#clean_pdf_df=pd.read_pickle("/kaggle/input/clean-pdf-df/clean_pdf_df.pkl")
# set text

text_cleaned = clean_pdf_df['text_cleaned']



from pathlib import Path



# set questions

path = '/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/3_patient_descriptions/'

file_list = sorted(list(Path(path).glob('*.csv')))

questions = [file.name.split(".csv")[0].strip('_') for file in file_list]

questions_cleaned = [clean_sent(ques) for ques in questions]

for i,q in enumerate(questions):

    print("Question" ,i+1,":",q)
#file_list = [os.path.join(r, file)  for r, _, f in os.walk(path)  for file in f]

table_cols_dic={}

table_dic={}

target_table_dic={}

for i,file in enumerate(file_list):

    df=pd.read_csv(file)

    cols=list(df.columns)

    table_cols_dic[i]=cols[1:]

    table_dic[i]=df

    target_table_dic[i]=pd.DataFrame(columns=cols)

[print(table_cols_dic[key]) for key in table_cols_dic.keys()]
# Select relevant paper to 

dic, sim_matrix = calc_simlarity_score(questions_cleaned, text_cleaned, threshold=0.15)

relevant_paper_dic = retrieve_paper(clean_pdf_df, dic)
for key in target_table_dic.keys():

    target_table_dic[key][['Date', 'Study', 'Journal']]=relevant_paper_dic[key][['publish_time', 'title', 'journal']]

    target_table_dic[key]['Study Link'] = "https://doi.org/" + relevant_paper_dic[key]['doi']

    relevant_paper_dic[key]=relevant_paper_dic[key].reset_index(drop=True)

    target_table_dic[key]=target_table_dic[key].reset_index(drop=True)

    target_table_dic[key]['Added on'] = "10-Jun-2020"
# question index

index=0

print(questions[index])

common_col_set = set(['Date', 'Study', 'Study Link', 'Journal', 'Added On','Added on'])

set(table_cols_dic[index]).difference(common_col_set)
col_questions=["What proportion or percentage can the virus be transmitted asymptomatically or during the incubation period?",

               #"How many patients or samples were used in the experiment or study for asymptomatical transmission?",

               "What is the number of patients, cases, or samples?",

               #"What is the range, mean, median, or IQR of the age of patients or samples?",

               "How old are the patients or what is the age of the patients?",

               "What sample type obtained such as Anal, Blood, Broncho-alveolar lavage, Conjunctival, Fecal, GI tract, Lower respiratory tract, Nasal, Nasopharyngeal, Oropharyngeal, Pharyngeal, Rectal, Respiratory, Sputum, Throat, Urine, or not found ?",

               "What is the study type or article research type like Systematic review, meta-analysis, Prospective observational study, Retrospective observational study, Observational study, Cross-sectional study, Case series, Expert review, Editorial, Simulation, or not found?"]



col_names=['Asymptomatic Transmission','Sample Size',"Age","Sample Obtained","Study Type"]

col_for_excerpt='Asymptomatic Transmission'

may_not_have_num=['Sample Obtained',"Study Type"]

target_table_dic[index]=col_fill(col_questions,col_names,col_for_excerpt,may_not_have_num,index)

# Clean "Study Type","Sample Obtained"

cols=["Study Type","Sample Obtained"]

for i in range(relevant_paper_dic[index].shape[0]):

    #get text of each paper, preprocess it

    whole_text=relevant_paper_dic[index].text[i]    

    #for col in col_names:

    for col in cols:

        bert_returned=target_table_dic[index][col][i]

        if  bert_returned.lower() not in whole_text.lower():

            target_table_dic[index][col][i]=""

            

# select papers with qualified answers

con1 = target_table_dic[index]['Asymptomatic Transmission'].str.contains("%")

target_table_dic[index]=target_table_dic[index][con1]

target_table_dic[index]['Characteristic Related to Question 2'] = "-"



# Clean the table after manual checks

for i,row in target_table_dic[index].iterrows():

    answer=answer_question("What is the number of patients, cases, or samples?", row["Excerpt"])

    if "%" not in answer:

        row["Sample Size"]=answer

    else:

        row["Sample Size"]= ""

    if row["Sample Size"] == "20 , 21 , 22 , 23 , 24 , 25":

        row["Sample Size"]=""

        row["Age"]=""

    if row["Sample Size"]=="asymptomatic cases are likely under - reported ( 26 ) ( 27 ) ( 28 ) ( 29 )":

        row["Sample Size"]=""

    if "days" in row["Age"]:

        row["Age"]=""

target_table_dic[index]=target_table_dic[index][target_table_dic[index]['Asymptomatic Transmission']!="one - third of patients and sore throat was found in 14 . 0 %"]

target_table_dic[index]

target_table_dic[index].to_csv('/kaggle/working/table_'+ questions[index] +'.csv')
index=3

print(questions[index])

set(table_cols_dic[index]).difference(common_col_set)
col_questions=["What is the mean or median of the length in days of viral shedding after illness onset?",

               "What is the IQR or range of length in days of viral shedding after illness onset?",

               "What is the sample size or patients / cases number for the experiment of viral shedding?",

               "What is the age (years old) range of the patients or samples?",

               "What sample type obtained such as Anal, Blood, Broncho-alveolar lavage, Conjunctival, Fecal, GI tract, Lower respiratory tract, Nasal, Nasopharyngeal, Oropharyngeal, Pharyngeal, Rectal, Respiratory, Sputum, Throat, Urine?",

               "What is the study type or article research type like Systematic review, meta-analysis, Prospective observational study, Retrospective observational study, Observational study, Cross-sectional study, Case series, Expert review, Editorial, Simulation?"]



col_names=['Days','Range (Days)','Sample Size',"Age","Sample Obtained","Study Type"]

col_for_excerpt="Days"

may_not_have_num=['Sample Obtained',"Study Type"]

target_table_dic[index]=col_fill(col_questions,col_names,col_for_excerpt,may_not_have_num,index)

# Clean "Study Type","Sample Obtained"

cols=["Study Type","Sample Obtained"]

for i in range(relevant_paper_dic[index].shape[0]):

    #get text of each paper, preprocess it

    whole_text=relevant_paper_dic[index].text[i]    

    #for col in col_names:

    for col in cols:

        bert_returned=target_table_dic[index][col][i]

        if  bert_returned.lower() not in whole_text.lower():

            target_table_dic[index][col][i]=""

con1 = target_table_dic[index]["Days"] != ""

con2 = target_table_dic[index]["Range (Days)"] != ""

target_table_dic[index]=target_table_dic[index][ con1 | con2]

target_table_dic[index]=target_table_dic[index][target_table_dic[index]["Days"]!="76 . 7"]

target_table_dic[index].loc[target_table_dic[index]["Range (Days)"]=="13 days","Days"] ="13 days"

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="34 ( 44 . 2 % ) males","Sample Size"] ="64 patients"

target_table_dic[index]=target_table_dic[index][target_table_dic[index]["Range (Days)"]!="15 to 89"]

target_table_dic[index].loc[target_table_dic[index]["Days"]=="13 days","Days"] ="mean duration of 4 days"

target_table_dic[index].loc[target_table_dic[index]["Days"]=="13 days","Range (Days)"] =" IQR 3-7 days"

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="> 65 years","Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Age"]=="2 . 61 , 2 . 77","Age"] =""

target_table_dic[index]=target_table_dic[index][target_table_dic[index]["Days"]!="25"]

target_table_dic[index].loc[target_table_dic[index]["Age"]=="4 ( 3 - 7 ) days","Age"] =""

target_table_dic[index]

target_table_dic[index].to_csv('/kaggle/working/table_'+ questions[index] +'.csv')
# question index

index=4

print(questions[index])

set(table_cols_dic[index]).difference(common_col_set)
col_questions=["Cardiac , Dermatological , Hepatic ,Ocular , or Neurological manifestations?",

               "What is the frequency distribution of Symptoms",

               "What is the sample size or patients / cases number for the experiment or study for asymptomatical transmission?",

               #"What is the range, mean, median, or IQR of age?",

               #"What is the age range of the patients?",

               "What is the age range",

               "What sample type obtained such as Anal, Blood, Broncho-alveolar lavage, Conjunctival, Fecal, GI tract, Lower respiratory tract, Nasal, Nasopharyngeal, Oropharyngeal, Pharyngeal, Rectal, Respiratory, Sputum, Throat, Urine, or not found ?",

               "Systematic review, meta-analysis, Prospective observational study, Retrospective observational study, Observational study, Cross-sectional study, Case series, Expert review, Editorial, Simulation, or not found?"]



col_names=['Manifestation','Frequency of Symptoms','Sample Size',"Age","Sample Obtained","Study Type"]

col_for_excerpt='Manifestation'

may_not_have_num=["Manifestation",'Sample Obtained',"Study Type"]

a=col_fill(col_questions,col_names,col_for_excerpt,may_not_have_num,index)

# Clean "Study Type","Sample Obtained"

cols=["Study Type","Sample Obtained","Manifestation"]

for i in range(relevant_paper_dic[index].shape[0]):

    #get text of each paper, preprocess it

    whole_text=relevant_paper_dic[index].text[i]    

    #for col in col_names:

    for col in cols:

        bert_returned=target_table_dic[index][col][i]

        if  bert_returned.lower() not in whole_text.lower():

            target_table_dic[index][col][i]=""

con1 = target_table_dic[index]["Manifestation"] != ""

target_table_dic[index]=target_table_dic[index][ con1]# & con2]

target_table_dic[index].loc[-target_table_dic[index]["Age"].str.contains("years"),"Age"] =""

target_table_dic[index].loc[target_table_dic[index]["Sample Size"].str.contains("years"),"Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Manifestation"]=="haematological","Sample Size"] ="99 patients"

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="14 patients","Frequency of Symptoms"] ="Comorbidities were present in 51.8%, and the most frequent symptoms were fever (87%), cough (71%), chest pain/tightness (65%), and dyspnea (56%)"

target_table_dic[index].loc[-target_table_dic[index]["Frequency of Symptoms"].str.contains("%"),"Frequency of Symptoms"] =""

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="over 0 . 04ng / ml","Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="1514866","Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Study"]=="Cardiac involvement in COVID-19 patients: Risk factors, predictors, and complications: A review","Frequency of Symptoms"] ="patients that recovered from sars - cov reported that 68 % had hyperlipidemia , 4 % had cardiovascular system abnormalities , and 60 % had glucose metabolism disorders after recovery."

target_table_dic[index].loc[target_table_dic[index]["Study"]=="Cardiac involvement in COVID-19 patients: Risk factors, predictors, and complications: A review","Sample Size"] =""

target_table_dic[index]

target_table_dic[index].to_csv('/kaggle/working/table_'+ questions[index] +'.csv')
# question index

index=5

print(questions[index])

set(table_cols_dic[index]).difference(common_col_set)
col_questions=["What is the proportion (percentage) of all positive COVID19 patients who were asymptomatic?",

               "What is the sample size or patients / cases number for the experiment of symptom?",

               "What is the age range of the patients or samples?",

               "What sample type obtained such as Anal, Blood, Broncho-alveolar lavage, Conjunctival, Fecal, GI tract, Lower respiratory tract, Nasal, Nasopharyngeal, Oropharyngeal, Pharyngeal, Rectal, Respiratory, Sputum, Throat, Urine, or not found ?",

               "What is the study type or article research type like Systematic review, meta-analysis, Prospective observational study, Retrospective observational study, Observational study, Cross-sectional study, Case series, Expert review, Editorial, Simulation, or not found?"]



col_names=['Asymptomatic','Sample Size',"Age","Sample Obtained","Study Type"]

col_for_excerpt='Asymptomatic'

may_not_have_num=['Sample Obtained',"Study Type"]

target_table_dic[index]=col_fill(col_questions,col_names,col_for_excerpt,may_not_have_num,index)

# Clean "Study Type","Sample Obtained"

cols=["Study Type","Sample Obtained"]

for i in range(relevant_paper_dic[index].shape[0]):

    #get text of each paper, preprocess it

    whole_text=relevant_paper_dic[index].text[i]    

    #for col in col_names:

    for col in cols:

        bert_returned=target_table_dic[index][col][i]

        if  bert_returned.lower() not in whole_text.lower():

            target_table_dic[index][col][i]=""

con1 = target_table_dic[index]["Asymptomatic"].str.contains("%")

con2 = target_table_dic[index]["Study"] != ""

con3 = target_table_dic[index]["Excerpt"].str.contains("Asymptom")

con4 = target_table_dic[index]["Excerpt"].str.contains("asymptom")

target_table_dic[index]=target_table_dic[index][con1 & con2 & (con3 | con4)]

target_table_dic[index]['Characteristic Related to Question 2'] = "-"

target_table_dic[index].loc[target_table_dic[index]["Sample Size"].str.contains("%"),"Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Age"].str.contains("%"),"Age"] =""

target_table_dic[index].loc[target_table_dic[index]["Age"]=="−80°c","Sample Size"] ="17 symptomatic patients"

target_table_dic[index].loc[target_table_dic[index]["Age"]=="−80°c","Age"] =""

target_table_dic[index].loc[target_table_dic[index]["Sample Size"].str.contains("≤"),"Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="7","Sample Size"] =""

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="[ 7 ] [ 8 ]","Sample Size"] =""

target_table_dic[index].loc[-target_table_dic[index]["Age"].str.contains("year"),"Age"] =""

target_table_dic[index]

target_table_dic[index].to_csv('/kaggle/working/table_'+ questions[index] +'.csv')
# question index

index=6

print(questions[index])

set(table_cols_dic[index]).difference(common_col_set)
col_questions=["What proportion or percentage of pediatric patients were asymptomatic?",

               "What is the sample size or patients / cases number for the experiment or study for asymptomatical transmission?",

               "What is the age range of the patients?",

               "What sample obtained such as Anal, Blood, Broncho-alveolar lavage, Conjunctival, Fecal, GI tract, Lower respiratory tract, Nasal, Nasopharyngeal, Oropharyngeal, Pharyngeal, Rectal, Respiratory, Sputum, Throat, Urine, or not found ?",

               "What is the study type or article research type like Systematic review, meta-analysis, Prospective observational study, Retrospective observational study, Observational study, Cross-sectional study, Case series, Expert review, Editorial, Simulation, or not found?"]



col_names=['Aymptomatic','Sample Size',"Age","Sample Obtained","Study Type"]

col_for_excerpt='Aymptomatic'

may_not_have_num=['Sample Obtained',"Study Type"]

target_table_dic[index]=col_fill(col_questions,col_names,col_for_excerpt,may_not_have_num,index)

# Clean "Study Type","Sample Obtained"

cols=["Study Type","Sample Obtained"]

for i in range(relevant_paper_dic[index].shape[0]):

    #get text of each paper, preprocess it

    whole_text=relevant_paper_dic[index].text[i]    

    #for col in col_names:

    for col in cols:

        bert_returned=target_table_dic[index][col][i]

        if  bert_returned.lower() not in whole_text.lower():

            target_table_dic[index][col][i]=""

#con1 = target_table_dic[index]["Manifestation"] != ""

con2 = target_table_dic[index]["Aymptomatic"].str.contains("%")

con1 = target_table_dic[index]["Excerpt"].str.contains("pediatr")

con3 = target_table_dic[index]["Excerpt"].str.contains("Asymptom")

con4 = target_table_dic[index]["Excerpt"].str.contains("asymptom")

target_table_dic[index]=target_table_dic[index][con1 & con2 & (con3 | con4)]

target_table_dic[index]['Characteristic Related to Question 2'] = "-"

target_table_dic[index].loc[-target_table_dic[index]["Age"].str.contains("year"),"Age"] =""

target_table_dic[index]["Sample obtained"] =target_table_dic[index]["Sample Obtained"] 

target_table_dic[index].loc[target_table_dic[index]["Sample Size"]=="69","Sample Size"] ="2134 pediatric patients"

target_table_dic[index]

target_table_dic[index].to_csv('/kaggle/working/table_'+ questions[index] +'.csv')