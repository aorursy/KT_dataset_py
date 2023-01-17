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
!pip install sense2vec
!pip install git+https://github.com/boudinfl/pke.git

!python -m nltk.downloader stopwords



!python -m nltk.downloader universal_tagset

!python -m spacy download en # download the english model
!pip install strsim
import time

import torch

import numpy

from transformers import T5ForConditionalGeneration,T5Tokenizer

import random

import spacy

import boto3

import zipfile

import os

import json

from sense2vec import Sense2Vec

import requests

from collections import OrderedDict

import string



import pke

import nltk

from nltk import FreqDist

nltk.download('brown')

nltk.download('stopwords')

nltk.download('popular')

from nltk.corpus import stopwords

from nltk.corpus import brown

from similarity.normalized_levenshtein import NormalizedLevenshtein



from nltk.tokenize import sent_tokenize

from flashtext import KeywordProcessor

!ls
def MCQs_available(word,s2v):

    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)

    if sense is not None:

        return True

    else:

        return False

def edits(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz '+string.punctuation

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)
def sense2vec_get_words(word,s2v):

    output = []



    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))

    word_preprocessed = word_preprocessed.lower()



    word_edits = edits(word_preprocessed)



    word = word.replace(" ", "_")



    sense = s2v.get_best_sense(word)

    most_similar = s2v.most_similar(sense, n=15)



    compare_list = [word_preprocessed]

    for each_word in most_similar:

        append_word = each_word[0].split("|")[0].replace("_", " ")

        append_word = append_word.strip()

        append_word_processed = append_word.lower()

        append_word_processed = append_word_processed.translate(append_word_processed.maketrans("","", string.punctuation))

        if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:

            output.append(append_word.title())

            compare_list.append(append_word_processed)





    out = list(OrderedDict.fromkeys(output))



    return out
# Return an array of options

def get_options(answer,s2v):

    distractors =[]



    try:

        distractors = sense2vec_get_words(answer,s2v)

        if len(distractors) > 0:

            print(" Sense2vec_distractors successful for word : ", answer)

            return distractors,"sense2vec"

    except:

        print (" Sense2vec_distractors failed for word : ",answer)





    return distractors,"None"
def tokenize_sentences(text):

    sentences = [sent_tokenize(text)]

    sentences = [y for x in sentences for y in x]

    # Remove any short sentences less than 20 letters.

    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]

    return sentences
def get_sentences_for_keyword(keywords, sentences):

    keyword_processor = KeywordProcessor()

    keyword_sentences = {}

    for word in keywords:

        word = word.strip()

        keyword_sentences[word] = []

        keyword_processor.add_keyword(word)

    for sentence in sentences:

        keywords_found = keyword_processor.extract_keywords(sentence)

        for key in keywords_found:

            keyword_sentences[key].append(sentence)



    for key in keyword_sentences.keys():

        values = keyword_sentences[key]

        values = sorted(values, key=len, reverse=True)

        keyword_sentences[key] = values



    delete_keys = []

    for k in keyword_sentences.keys():

        if len(keyword_sentences[k]) == 0:

            delete_keys.append(k)

    for del_key in delete_keys:

        del keyword_sentences[del_key]



    return keyword_sentences


def is_far(words_list,currentword,thresh,normalized_levenshtein):

    threshold = thresh

    score_list =[]

    for word in words_list:

        score_list.append(normalized_levenshtein.distance(word.lower(),currentword.lower()))

    if min(score_list)>=threshold:

        return True

    else:

        return False


def filter_phrases(phrase_keys,max,normalized_levenshtein ):

    filtered_phrases =[]

    if len(phrase_keys)>0:

        filtered_phrases.append(phrase_keys[0])

        for ph in phrase_keys[1:]:

            if is_far(filtered_phrases,ph,0.7,normalized_levenshtein ):

                filtered_phrases.append(ph)

            if len(filtered_phrases)>=max:

                break

    return filtered_phrases
def get_nouns_multipartite(text):

    out = []



    extractor = pke.unsupervised.MultipartiteRank()

    extractor.load_document(input=text, language='en')

    pos = {'PROPN', 'NOUN'}

    stoplist = list(string.punctuation)

    stoplist += stopwords.words('english')

    extractor.candidate_selection(pos=pos, stoplist=stoplist)

    # 4. build the Multipartite graph and rank candidates using random walk,

    #    alpha controls the weight adjustment mechanism, see TopicRank for

    #    threshold/method parameters.

    try:

        extractor.candidate_weighting(alpha=1.1,

                                      threshold=0.75,

                                      method='average')

    except:

        return out



    keyphrases = extractor.get_n_best(n=10)



    for key in keyphrases:

        out.append(key[0])



    return out
def get_phrases(doc):

    phrases={}

    for np in doc.noun_chunks:

        phrase =np.text

        len_phrase = len(phrase.split())

        if len_phrase > 1:

            if phrase not in phrases:

                phrases[phrase]=1

            else:

                phrases[phrase]=phrases[phrase]+1



    phrase_keys=list(phrases.keys())

    phrase_keys = sorted(phrase_keys, key= lambda x: len(x),reverse=True)

    phrase_keys=phrase_keys[:50]

    return phrase_keys


def get_keywords(nlp,text,max_keywords,s2v,fdist,normalized_levenshtein,no_of_sentences):

    doc = nlp(text)

    max_keywords = int(max_keywords)



    keywords = get_nouns_multipartite(text)

    keywords = sorted(keywords, key=lambda x: fdist[x])

    keywords = filter_phrases(keywords, max_keywords,normalized_levenshtein )



    phrase_keys = get_phrases(doc)

    filtered_phrases = filter_phrases(phrase_keys, max_keywords,normalized_levenshtein )



    total_phrases = keywords + filtered_phrases



    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2*no_of_sentences),normalized_levenshtein )





    answers = []

    for answer in total_phrases_filtered:

        if answer not in answers and MCQs_available(answer,s2v):

            answers.append(answer)



    answers = answers[:max_keywords]

    return answers


def generate_questions(keyword_sent_mapping,device,tokenizer,model,sense2vec,normalized_levenshtein):

    model.to(device)

    batch_text = []

    answers = keyword_sent_mapping.keys()

    for answer in answers:

        txt = keyword_sent_mapping[answer]

        context = "context: " + txt

        text = context + " " + "answer: " + answer + " </s>"

        batch_text.append(text)



    # print ("batch_text")

    # print (batch_text)

    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")





    print ("Running model for generation")

    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)



    with torch.no_grad():

        outs = model.generate(input_ids=input_ids,

                              attention_mask=attention_masks,

                              max_length=150)



    output_array ={}

    output_array["questions"] =[]



    for index, val in enumerate(answers):

        individual_question ={}

        out = outs[index, :]

        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)



        Question = dec.replace("question:", "")

        Question = Question.strip()

        individual_question["question_statement"] = Question

        individual_question["question_type"] = "MCQ"

        individual_question["answer"] = val

        individual_question["id"] = index+1

        individual_question["options"], individual_question["options_algorithm"] = get_options(val, sense2vec)



        individual_question["options"] =  filter_phrases(individual_question["options"], 10,normalized_levenshtein)

        index = 3

        individual_question["extra_options"]= individual_question["options"][index:]

        individual_question["options"] = individual_question["options"][:index]

        individual_question["context"] = keyword_sent_mapping[val]

        # individual_question["options"]=[]

        # individual_question["options_algorithm"] = ""

        if len(individual_question["options"])>0:

            output_array["questions"].append(individual_question)



        print("Context: ", keyword_sent_mapping[val])

        print("Generated Question: ")

        print(Question)

        print("Answer: ", val)



    return output_array


class PythonPredictor:

    def __init__(self):

        model_file_1 = "../input/s2v-old/s2v_old"

        

        print ("s2v model already exists.")

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        # model.eval()

        self.device = device

        self.model = model

        self.nlp = spacy.load('en_core_web_sm')



        self.s2v = Sense2Vec().from_disk('../input/s2v-old/s2v_old')



        self.fdist = FreqDist(brown.words())

        self.normalized_levenshtein = NormalizedLevenshtein()

        self.set_seed(42)

    def set_seed(self,seed):

        numpy.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(seed)

    def predict(self, payload):

        start = time.time()

        inp = {

            "input_text": payload.get("input_text"),

            "max_questions": payload.get("max_questions", 4)

        }



        text = inp['input_text']

        sentences = tokenize_sentences(text)

        joiner = " "

        modified_text = joiner.join(sentences)





        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )





        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)



        for k in keyword_sentence_mapping.keys():

            text_snippet = " ".join(keyword_sentence_mapping[k][:3])

            keyword_sentence_mapping[k] = text_snippet



   

        final_output = {}



        if len(keyword_sentence_mapping.keys()) == 0:

            return json.dumps(final_output)

        else:

            try:

                generated_questions = generate_questions(keyword_sentence_mapping,self.device,self.tokenizer,self.model,self.s2v,self.normalized_levenshtein)



            except:

                return final_output

            end = time.time()



            final_output["statement"] = modified_text

            final_output["questions"] = generated_questions["questions"]

            final_output["time_taken"] = end-start



            return final_output

    
if __name__ == "__main__":



    mcq = PythonPredictor()



    payload={

    "input_text" : "A double-walled sac called the pericardium encases the heart, which serves to protect the heart and anchor it inside the chest. Between the outer layer, the parietal pericardium, and the inner layer, the serous pericardium, runs pericardial fluid, which lubricates the heart during contractions and movements of the lungs and diaphragm.The heart's outer wall consists of three layers. The outermost wall layer, or epicardium, is the inner wall of the pericardium. The middle layer, or myocardium, contains the muscle that contracts. The inner layer, or endocardium, is the lining that contacts the blood.The tricuspid valve and the mitral valve make up the atrioventricular (AV) valves, which connect the atria and the ventricles. The pulmonary semi-lunar valve separates the right ventricle from the pulmonary artery, and the aortic valve separates the left ventricle from the aorta. The heartstrings, or chordae tendinae, anchor the valves to heart muscles."

    }



    out= mcq.predict(payload)
out["statement"]

out
out["questions"][0]
import gc

gc.collect()