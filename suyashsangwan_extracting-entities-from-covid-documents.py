import os

print(os.listdir("../input"))
import os

import re

import csv

import math

import json



import string

from string import punctuation



import pandas as pd

import numpy as np



import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords



import spacy

from spacy import displacy



!pip install scispacy

import scispacy

from scispacy.abbreviation import AbbreviationDetector

from scispacy.umls_linking import UmlsEntityLinker
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_jnlpba_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_craft_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import en_ner_jnlpba_md

import en_ner_craft_md

import en_ner_bc5cdr_md

import en_ner_bionlp13cg_md

import en_core_sci_lg



nlp0 = en_ner_jnlpba_md.load()

nlp1 = en_ner_craft_md.load()

nlp2 = en_ner_bc5cdr_md.load()

nlp3 = en_ner_bionlp13cg_md.load()

nlp4 = en_core_sci_lg.load()

!python -m spacy download en_core_web_md



import en_core_web_md

spacy_nlp = en_core_web_md.load()

# function to extract NER

def ner(nlp5, text):

    doc = nlp5(text)

    ent = ''

    ent = doc.ents

    return ent



# function to save NER in dataframe

def finding_entities(input_df,df_col_name):

    sent_craft_ent = []

    sent_jnlpba_ent = []

    sent_bc5cdr_ent = []

    sent_bionlp13cg_ent = []

    sent_sci_ent = []

    print(' Entering in the for loop','\n', 'It may take upto several minutes...... ')

    for i in range(0,len(input_df)):

        sent_jnlpba_ent.append(ner(nlp0, input_df[df_col_name].loc[i]))

        sent_craft_ent.append(ner(nlp1, input_df[df_col_name].loc[i]))

        sent_bc5cdr_ent.append(ner(nlp2, input_df[df_col_name].loc[i]))

        sent_bionlp13cg_ent.append(ner(nlp3, input_df[df_col_name].loc[i]))

        sent_sci_ent.append(ner(nlp4, input_df[df_col_name].loc[i]))

        print(i)



    print('writing the dataframe')

    input_df['sent_jnlpba_ent'] =  sent_jnlpba_ent

    input_df['sent_craft_ent'] = sent_craft_ent

    input_df['sent_bc5cdr_ent'] = sent_bc5cdr_ent

    input_df['sent_bionlp13cg_ent'] = sent_bionlp13cg_ent

    input_df['sent_sci_ent'] = sent_sci_ent

    return input_df



# functions to extract Spacy Named Entities 

def date(abstract):

 li=[]

 try:

    document = spacy_nlp(abstract)

    for ent in document.ents:

     if ent.label_=="DATE":

#       print(ent.label_, ent.text)

       li.append(ent.text)

    print(li)

    return li

 except:

    return li



def time(abstract):

 li=[]

 try:

    document = spacy_nlp(abstract)

    for ent in document.ents:

     if ent.label_=="TIME":

#       print(ent.label_, ent.text)

       li.append(ent.text)

    print(li)

    return li

 except:

    return li



def percent(abstract):

 li=[]

 try:

    document = spacy_nlp(abstract)

    for ent in document.ents:

     if ent.label_=="PERCENT":

#       print(ent.label_, ent.text)

       li.append(ent.text)

    print(li)

    return li

 except:

    return li



def quantity(abstract):

 li=[]

 try:

    document = spacy_nlp(abstract)

    for ent in document.ents:

     if ent.label_=="QUANTITY":

#       print(ent.label_, ent.text)

       li.append(ent.text)

    print(li)

    return li

 except:

    return li



def ordinal(abstract):

 li=[]

 try:

    document = spacy_nlp(abstract)

    for ent in document.ents:

     if ent.label_=="ORDINAL":

#       print(ent.label_, ent.text)

       li.append(ent.text)

    print(li)

    return li

 except:

    return li



def cardinal(abstract):

 li=[]

 try:

    document = spacy_nlp(abstract)

    for ent in document.ents:

     if ent.label_=="CARDINAL":

#       print(ent.label_, ent.text)

       li.append(ent.text)

    print(li)

    return li

 except:

    return li





#function to add spacy entities in dataframe

def adding_spacy_ner(input_df,df_col_name):

    input_df['Extracted_DATE_Entities'] = input_df[df_col_name].apply(date)

    print("........ DATE Entities Extracted Successfully..........")

    input_df['Extracted_TIME_Entities'] = input_df[df_col_name].apply(time)

    print("........ TIME Entities Extracted Successfully..........")

    input_df['Extracted_PERCENT_Entities'] = input_df[df_col_name].apply(percent)

    print("........ PERCENT Entities Extracted Successfully..........")

    input_df['Extracted_QUANTITY_Entities'] = input_df[df_col_name].apply(quantity)

    print("........ QUANTITY Entities Extracted Successfully..........")

    input_df['Extracted_ORDINAL_Entities'] = input_df[df_col_name].apply(ordinal)

    print("........ ORDINAL Entities Extracted Successfully..........")

    input_df['Extracted_CARDINAL_Entities'] = input_df[df_col_name].apply(cardinal)

    print("........ CARDINAL Entities Extracted Successfully..........")

    return input_df

#input_df = pd.read_csv("/kaggle/input/covid-19-dataset-filtering-and-sentence-extraction/Filtered_covid_documents_with_metadata.csv",sep="\t")



#input_df = finding_entities(input_df,'Abstract_and_Text')

#input_df = adding_spacy_ner(input_df,'Abstract_and_Text')



#input_df.to_csv("Extracted_entities_from_Filtered_covid_documents_with_metadata.csv", sep='\t')
input_df = pd.read_csv("/kaggle/input/covid-19-dataset-filtering-and-sentence-extraction/Extracted_sentences_from_filtered_covid_documents.csv",sep="\t")



input_df = finding_entities(input_df,'Sentence')

input_df = adding_spacy_ner(input_df,'Sentence')



input_df.to_csv("Extracted_entities_from_extracted_sentences_from_filtered_covid_documents.csv", sep='\t')