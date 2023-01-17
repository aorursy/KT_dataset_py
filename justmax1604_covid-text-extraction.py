import os

import re

import json

from tqdm import tqdm

import pandas as pd

from collections import Counter

from stop_words import get_stop_words

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import numpy as np
# get the dataframe of containing those JSONs that correspond to correct dates and have pdf_json files

df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', usecols=["pdf_json_files", "publish_time"])

df_dates_correct = df.query('publish_time >= "2020-05-18" and publish_time < "2020-05-25"')

df_hasJSON = df_dates_correct.query('pdf_json_files == pdf_json_files')
# get the appropriate files from the folder containing pdf_json parces if they match the JSONs mentioned in the df_hasJSON, then extract appropriate data and write it in the docs list

# 

path = '../input/CORD-19-research-challenge/document_parses/pdf_json'

file_list = os.listdir(path)

docs = []

file_name_array = df_hasJSON['pdf_json_files'].to_numpy()

for file in file_list:

        file_path = f"../input/CORD-19-research-challenge/document_parses/pdf_json/{file}"

    

        if np.any(file_name_array[:] == file_path[36:]):

            j = json.load(open(file_path, "rb"))



            title = j["metadata"]["title"]

            authors = j["metadata"]["authors"]



            try:

                abstract = j["abstract"][0]["text"].lower()

            except:

                abstract = ""



            full_text = ""

            for text in j["body_text"]:

                full_text += text["text"].lower() + "\n\n"

            docs.append([file, title, authors, abstract, full_text])
# get a dataframe using docs list

# fetched a total of 2725 appropriate papers

df_papers = pd.DataFrame(docs, columns=["file_id", "title", "authors", "abstract", "full_text"])
# get a dataframe from df_papers that have "symptom" in full text

# fetched a total of 1327 appropriate papers

symptoms_df = df_papers[df_papers["full_text"].str.contains("symptom")]
# list of symptoms that will be used to tag the appropriate symptom words. This is not an exhaustive list, I included as many elements as I could find/think of

symptoms = [

    "weight loss","chills","shivering","convulsions","deformity","discharge","dizziness", "lymphopenia", "sneezing",

    "vertigo","fatigue","malaise","asthenia","hypothermia","jaundice","muscle weakness", "chest discomfort",

    "pyrexia","sweats","swelling","swollen","painful lymph node","weight gain","arrhythmia", "loss of smell", "loss of appetite", "loss of taste",

    "bradycardia","chest pain","claudication","palpitations","tachycardia","dry mouth","epistaxis", "dysgeusia", "hypersomnia", "taste loss",

    "halitosis","hearing loss","nasal discharge", "nasal inflammation", "otalgia","otorrhea","sore throat","toothache","tinnitus", "dysphonia",

    "trismus","abdominal pain","fever","bloating","belching","bleeding","bloody stool","melena","hematochezia", "burning sensation in the chest", 

    "constipation","diarrhea","dysphagia","dyspepsia","fecal incontinence","flatulence","heartburn", "chest tightness", "chest pressure",

    "nausea","odynophagia","proctalgia fugax","pyrosis","steatorrhea","vomiting","alopecia","hirsutism", "tachypnoea", "nasal obstruction",

    "hypertrichosis","abrasion","anasarca","bleeding into skin","petechia","purpura","ecchymosis", "bruising", 

    "blister","edema","itching","laceration","rash","urticaria","abnormal posturing","acalculia","agnosia","alexia",

    "amnesia","anomia","anosognosia","aphasia","apraxia","ataxia","cataplexy","confusion","dysarthria", "nasal congestion",

    "dysdiadochokinesia","dysgraphia","hallucination","headache","akinesia","bradykinesia","akathisia","athetosis",

    "ballismus","blepharospasm","chorea","dystonia","fasciculation","muscle cramps","myoclonus","opsoclonus","tic",

    "tremor","flapping tremor","insomnia","loss of consciousness","syncope","neck stiffness","opisthotonus",

    "paralysis","paresis","paresthesia","prosopagnosia","somnolence","abnormal vaginal bleeding", "neuralgia",

    "vaginal bleeding in early pregnancy", "miscarriage","vaginal bleeding in late pregnancy","amenorrhea", "body aches",

    "infertility","painful intercourse","pelvic pain","vaginal discharge","amaurosis fugax","amaurosis", "skin lesions",

    "blurred vision","double vision","exophthalmos","mydriasis","miosis","nystagmus","amusia","anhedonia",

    "anxiety","apathy","confabulation","depression","delusion","euphoria","homicidal ideation","irritability",

    "mania","paranoid ideation","suicidal ideation","apnea","hypopnea","cough","dyspnea","bradypnea","tachypnea",

    "orthopnea","platypnea","trepopnea","hemoptysis","pleuritic chest pain","sputum production","arthralgia",

    "back pain","sciatica","urologic","dysuria","hematospermia","hematuria","impotence","polyuria",

    "retrograde ejaculation","strangury","urethral discharge","urinary frequency","urinary incontinence","urinary retention", "anosmia", "myalgia", "rhinorrhea", "shortness of breath"]
# get a dataframe with only columns of interest

control_sentence_df = symptoms_df[["file_id", "full_text"]]

# empty list for populating data

control_data = []

# go through each sentence in full text, if the sentence contains "symptom", append the paper_id, sentence_id(paper_id +sentence number), and sentence text

for value in control_sentence_df.values:

    file_id = value[0]

    numSentences = 0

    text = value[1]

    text = text.replace('?', '.').replace('!', '.')

    sentences = re.split('[. ] |\n',text)

    for sentence in sentences:

        numSentences+=1

        if ("symptom" in sentence):

            sentence_id = file_id+'_sent'+str(numSentences)

            control_data.append([file_id,sentence_id,sentence])

# get a dataframe from control_data list and make a csv file            

control_data_df = pd.DataFrame(control_data, columns=['paper_id','sentence_id','sentence'])

control_data_df.to_csv('control_data.csv',index=False)



# fetched a total of 7206 appropriate sentences containing symptoms
sentences_df = control_data_df["sentence"]

# empty list for populating data

data = []

# initialize sentence number

numSent = 0

for sentence in sentences_df.values:

    sentence = sentence.replace(',', '').replace(')','').replace('(','').replace('/', ' ').replace('-', ' ').replace('"','').replace('.','')

    sentence_lst = sentence.split()

    tag_lst = []

    sentenceNum_lst = []

    # increment the sentence number to start counting from 1.

    numSent+=1

    sentenceNum = 'Sentence #' +str(numSent)

    # append the sentence number list with number string 

    sentenceNum_lst.append(sentenceNum)

    # populate the rest of the list with empty strings for the rest of the words in the sentence

    sentenceNum_lst.extend(['' for i in range ((len(sentence_lst)-1))])

    # lists of multiword symptoms

    multiword_symptoms = ([((symptom.split())) for symptom in symptoms if len(symptom.split())>1])

    multiword_sym_first = [word[0] for word in multiword_symptoms]

    multiword_sym_rest = ([word[1:] for word in multiword_symptoms])

    # list with True or False values, based on whether the multiword symptom string was part of the current sentence

    symp_in_sent_lst=[]

    # go through the list of multiword symptom strings and see if they are part of the current sentence

    for value in [(symptom) for symptom in symptoms if len(symptom.split())>1]:

        if value in sentence:

            symp_in_sent_lst.append(True)

        else:

            symp_in_sent_lst.append(False)

    # go through each word in the sentence_lst and place appropriate tags on words

    for i, word in enumerate(sentence_lst):

        # word has to be in symptoms list or multiword symptom first words list (sentence must contain other words/phrase to formulate multiword symptoms and not be the last word in the sentence)  

        if (word in symptoms or word[:-1] in symptoms or (word in multiword_sym_first and i!=len(sentence_lst) and any(symp_in_sent_lst))):

            tag_lst.append('B-SYM')

        # word has to be in the ending symptom word list and must be preceded by a word with a symptom tag, also sentence must contain other words/phrase to formulate multiword symptoms

        elif (any(symp_in_sent_lst) and word in list(np.hstack(multiword_sym_rest)) and (len(tag_lst)) != 0 and (tag_lst[(len(tag_lst))-1] == 'B-SYM' or tag_lst[(len(tag_lst))-1] == 'I-SYM')):

            tag_lst.append('I-SYM')

        else:

            tag_lst.append('O')

        

        # add the data to the list at appropriate corresponding indexes    

        data.append([sentenceNum_lst[i],sentence_lst[i],tag_lst[i]])



# get a datafframe from data list and create a csv

tag_df = pd.DataFrame(data, columns=['Sentence','Words','Tag'])

tag_df.to_csv('tag_sheet.csv',index=False)



# fetched a total of 213574 words and placed tags accodingly