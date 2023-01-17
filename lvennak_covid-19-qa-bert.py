!pip install tqdm

!pip install transformers



!mkdir /kaggle/working/sentence_wise_email/

!mkdir /kaggle/working/sentence_wise_email/module/

!mkdir /kaggle/working/sentence_wise_email/module/module_useT

!mkdir /kaggle/working/top_10_results/



path_to_results = '/kaggle/working/top_10_results/'

path_to_module_useT = '/kaggle/working/sentence_wise_email/module/module_useT'
!mkdir module_useT

# Download the module, and uncompress it to the destination folder.

!curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC /kaggle/working/sentence_wise_email/module/module_useT
# Paths to json files

path_1 = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

path_2 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

path_3 = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

path_4 = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'



# List of folder names

folder_names = ['biorxiv_medrxiv','comm_use_subset']

data_path = path_2
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import json

import nltk

nltk.download('punkt')

from math import log, sqrt

from collections import defaultdict

from copy import deepcopy

from tqdm import tqdm

from IPython.core.display import display, HTML

import tensorflow as tf

import tensorflow_hub as hub

import torch

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer
TRAINING = True
# Data conversion from dataframe to 2 lists

def df2list(dir_path = data_path):

    filenames = os.listdir(dir_path)

    print('Number of articles retrieved:', len(filenames))

    files = load_files(dir_path)

    df = generate_clean_df(files)

    corpus = list(df['title'] + ' ' + df['abstract'] + ' ' + df['text'])

    paper_id = list(df['paper_id'])

    return corpus, paper_id
# Extraction of data from json format to dataframe 

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



def load_files(dirname, filename=None):

    filenames = os.listdir(dirname)

    raw_files = []

    if filename:

        filename = dirname + filename

        raw_files = [json.load(open(filename, 'rb'))]

    else:

        for filename in tqdm(filenames):

            filename = dirname + filename

            file = json.load(open(filename, 'rb'))

            raw_files.append(file)

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

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



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df = clean_df.drop(columns=['authors','affiliations','bibliography',

                                      'raw_authors','raw_bibliography'])

    return clean_df



# It updates the vects_for_docs variable with vectors of all the documents.

def iterate_over_all_docs():

    for i in range(num_of_documents):

        if np.mod(i, 100) == 0:

            print('{0} of {1}'.format(str(i).zfill(len(str(num_of_documents))),num_of_documents))

        doc_text = corpus[i]

        token_list = get_tokenized_and_normalized_list(doc_text)

        vect = create_vector(token_list)

        vects_for_docs.append(vect)

    print('{0} of {1}'.format(num_of_documents, num_of_documents))





def create_vector_from_query(l1):

    vect = {}

    for token in l1:

        if token in vect:

            vect[token] += 1.0

        else:

            vect[token] = 1.0

    return vect





def generate_inverted_index():

    count1 = 0

    for vector in vects_for_docs:

        for word1 in vector:

            inverted_index[word1].append(count1)

        count1 += 1





def create_tf_idf_vector():

    vect_length = 0.0

    for vect in vects_for_docs:

        for word1 in vect:

            word_freq = vect[word1]

            temp = calc_tf_idf(word1, word_freq)

            vect[word1] = temp

            vect_length += temp ** 2



        vect_length = sqrt(vect_length)

        for word1 in vect:

            vect[word1] /= vect_length





def get_tf_idf_from_query_vect(query_vector1):

    vect_length = 0.0

    for word1 in query_vector1:

        word_freq = query_vector1[word1]

        if word1 in document_freq_vect:

            query_vector1[word1] = calc_tf_idf(word1, word_freq)

        else:

            query_vector1[word1] = log(1 + word_freq) * log(

                num_of_documents)

        vect_length += query_vector1[word1] ** 2

    vect_length = sqrt(vect_length)

    if vect_length != 0:

        for word1 in query_vector1:

            query_vector1[word1] /= vect_length





def calc_tf_idf(word1, word_freq):

    return log(1 + word_freq) * log(num_of_documents / document_freq_vect[word1])





def get_dot_product(vector1, vector2):

    if len(vector1) > len(vector2):  # this will ensure that len(dict1) < len(dict2)

        temp = vector1

        vector1 = vector2

        vector2 = temp

    keys1 = vector1.keys()

    keys2 = vector2.keys()

    sum = 0

    for i in keys1:

        if i in keys2:

            sum += vector1[i] * vector2[i]

    return sum





def get_tokenized_and_normalized_list(doc_text):

    tokens = nltk.word_tokenize(doc_text)

    ps = nltk.stem.PorterStemmer()

    stemmed = []

    for words in tokens:

        stemmed.append(ps.stem(words))

    return stemmed





def create_vector(l1):

    vect = {}  # this is a dictionary

    global document_freq_vect

    for token in l1:

        if token in vect:

            vect[token] += 1

        else:

            vect[token] = 1

            if token in document_freq_vect:

                document_freq_vect[token] += 1

            else:

                document_freq_vect[token] = 1

    return vect





def get_result_from_query_vect(query_vector1):

    parsed_list = []

    for i in range(num_of_documents - 0):

        dot_prod = get_dot_product(query_vector1, vects_for_docs[i])

        parsed_list.append((i, dot_prod))

        parsed_list = sorted(parsed_list, key=lambda x: x[1])

    return parsed_list



if TRAINING:

    

    corpus, paper_id = df2list()



    # Data processing

    inverted_index = defaultdict(list)

    num_of_documents = len(corpus)

    vects_for_docs = []  # we will need nos of docs number of vectors, each vector is a dictionary

    document_freq_vect = {}  # sort of equivalent to initializing the number of unique words to 0

    

    iterate_over_all_docs()

    generate_inverted_index()

    create_tf_idf_vector()
# Search of relevant articles

def find_relevant_articles(query, N=10, save_csv_file=False):

    query_list = get_tokenized_and_normalized_list(query)

    query_vector = create_vector_from_query(query_list)

    get_tf_idf_from_query_vect(query_vector)

    result_set = get_result_from_query_vect(query_vector)

    papers_info = {'query':query, 'query list':query_list, 'query vector':query_vector,

                   'id':[], 'title':[], 'abstract':[], 'text':[], 'weight':[], 'index':[]}

    for i in range(1,N+1):

        tup = result_set[-i]

        raw_file = load_files(data_path, filename=paper_id[tup[0]]+'.json')

        df = generate_clean_df(raw_file)

        papers_info['id'].append(df['paper_id'][0])

        papers_info['title'].append(df['title'][0])

        papers_info['abstract'].append(df['abstract'][0])

        papers_info['text'].append(df['text'][0])

        papers_info['weight'].append(tup[1])

        papers_info['index'].append(tup[0])

        print('{0}: has relevance weight {1:.6f} and json file is {2} '.format(

                str(tup[0]).zfill(len(str(num_of_documents))), tup[1], paper_id[tup[0]]))

        print(df['title'][0])

    if save_csv_file:

        names = []

        data = [[]]

        for i in range(N):

            names.append('Article '+str(i))

            data[0].append(corpus[result_set[-1-i][0]])   

        df = pd.DataFrame(data, columns = names)

        df.to_csv('Articles.csv', index=True)

    print('\nTop {0} Most Relevant Articles:'.format(N))

    for i in range(N):

        print('Paper #{0}: {1}\n'.format(i+1, papers_info['title'][i]))

    return papers_info
import pickle

import os



def save_pickles(papers_info, query):



    name_query = query.replace(" ", "_")

    name_query = name_query.replace(",", "")

    name_query = name_query.replace(".", "")

    name_query = name_query.replace("/", " ")

    name_path = os.path.join(path_to_results, name_query)

    

    if not os.path.exists(name_path):

        os.makedirs(name_path)

    

    with open(path_to_results + name_query + '/' + name_query + '_papers_info.pickle', 'wb') as f:

        pickle.dump(papers_info, f)

    

def load_pickles(query):

    

    name_query = query.replace(" ", "_")

    name_query = name_query.replace(",", "")

    name_query = name_query.replace(".", "")

    name_query = name_query.replace("/", " ")

    

    with open(path_to_results + name_query + '/' + name_query + '_papers_info.pickle', 'rb') as f1:

        file = pickle.load(f1)

        

    return file

if TRAINING:

    

    question_list = []



    question_list.append("Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water")

    question_list.append("How long is the incubation period for the virus")

    question_list.append("Can the virus be transmitted asymptomatically or during the incubation period")

    question_list.append("What is the quantity of asymptomatic shedding")

    question_list.append("How does temperature and humidity affect the tramsmission of 2019-nCoV")

    question_list.append("How long can 2019-nCoV remain viable on inanimate, environmental, or common surfaces")

    question_list.append("What types of inanimate or environmental surfaces affect transmission, survival, or  inactivation of 2019-nCov")

    question_list.append("Can the virus be found in nasal discharge, sputum, urine, fecal matter, or blood")

    question_list.append("What risk factors contribute to the severity of 2019-nCoV")

    question_list.append("How does hypertension affect patients")

    question_list.append("How does heart disease affect patients")

    question_list.append("How does copd affect patients")

    question_list.append("How does smoking affect 2019-nCoV patients")

    question_list.append("How does pregnancy affect patients")

    question_list.append("What are the case fatality rates for 2019-nCoV patients")

    question_list.append("What is the case fatality rate in Italy")

    question_list.append("What public health policies prevent or control the spread of 2019-nCoV")

    question_list.append("Can animals transmit 2019-nCoV")

    question_list.append("What animal did 2019-nCoV come from")

    question_list.append("What real-time genomic tracking tools exist")

    question_list.append("What regional genetic variations (mutations) exist")

    question_list.append("What effors are being done in asia to prevent further outbreaks")

    question_list.append("What drugs or therapies are being investigated")

    question_list.append("What clinical trials for hydroxychloroquine have been completed")

    question_list.append("What antiviral drug clinical trials have been completed")

    question_list.append("Are anti-inflammatory drugs recommended")

    question_list.append("Which non-pharmaceutical interventions limit tramsission")

    question_list.append("What are most important barriers to compliance")

    question_list.append("How does extracorporeal membrane oxygenation affect 2019-nCoV patients")

    question_list.append("What telemedicine and cybercare methods are most effective")

    question_list.append("How is artificial intelligence being used in real time health delivery")

    question_list.append("What adjunctive or supportive methods can help patients")

    question_list.append("What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV")

    question_list.append("What is being done to increase testing capacity or throughput")

    question_list.append("What point of care tests are exist or are being developed")

    question_list.append("What is the minimum viral load for detection")

    question_list.append("What markers are used to detect or track COVID-19")

    question_list.append('What collaborations are happening within the research community')

    question_list.append("What are the major ethical issues related pandemic outbreaks")

    question_list.append("How do pandemics affect the physical and/or psychological health of doctors and nurses")

    question_list.append("What strategies can help doctors and nurses cope with stress in a pandemic")

    question_list.append("What factors contribute to rumors and misinformation")

    question_list.append("What is the immune system response to 2019-nCoV")

    question_list.append("Can personal protective equipment prevent the transmission of 2019-nCoV")

    question_list.append("Can 2019-nCoV infect patients a second time")

    question_list.append("What is the weighted prevalence of sars-cov-2 or covid-19 in general population")



    for query in question_list:



        papers_info = find_relevant_articles(query=query)

        save_pickles(query=query, papers_info=papers_info)

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'



QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

QA_MODEL.to(torch_device)

QA_MODEL.eval()
def reconstructText(tokens, start=0, stop=-1):

    tokens = tokens[start: stop]

    if '[SEP]' in tokens:

        sepind = tokens.index('[SEP]')

        tokens = tokens[sepind+1:]

    txt = ' '.join(tokens)

    txt = txt.replace(' ##', '')

    txt = txt.replace('##', '')

    txt = txt.strip()

    txt = " ".join(txt.split())

    txt = txt.replace(' .', '.')

    txt = txt.replace('( ', '(')

    txt = txt.replace(' )', ')')

    txt = txt.replace(' - ', '-')

    txt_list = txt.split(' , ')

    txt = ''

    nTxtL = len(txt_list)

    if nTxtL == 1:

        return txt_list[0]

    newList =[]

    for i,t in enumerate(txt_list):

        if i < nTxtL -1:

            if t[-1].isdigit() and txt_list[i+1][0].isdigit():

                newList += [t,',']

            else:

                newList += [t, ', ']

        else:

            newList += [t]

    return ''.join(newList)



def BERTSQuADPrediction(document, question):

    ## we need to rewrite this function so that it chuncks the document into 250-300 word segments with

    ## 50 word overlaps on either end so that it can understand and check longer abstracts

    nWords = len(document.split())

    input_ids_all = QA_TOKENIZER.encode(question, document)

    tokens_all = QA_TOKENIZER.convert_ids_to_tokens(input_ids_all)

    overlapFac = 1.1

    if len(input_ids_all)*overlapFac > 2048:

        nSearchWords = int(np.ceil(nWords/5))

        quarter = int(np.ceil(nWords/4))

        docSplit = document.split()

        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 

                     ' '.join(docSplit[quarter-int(nSearchWords*overlapFac/2):quarter+int(quarter*overlapFac/2)]),

                     ' '.join(docSplit[quarter*2-int(nSearchWords*overlapFac/2):quarter*2+int(quarter*overlapFac/2)]),

                     ' '.join(docSplit[quarter*3-int(nSearchWords*overlapFac/2):quarter*3+int(quarter*overlapFac/2)]),

                     ' '.join(docSplit[-int(nSearchWords*overlapFac):])]

        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]        

        

    elif len(input_ids_all)*overlapFac > 1536:

        nSearchWords = int(np.ceil(nWords/4))

        third = int(np.ceil(nWords/3))

        docSplit = document.split()

        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 

                     ' '.join(docSplit[third-int(nSearchWords*overlapFac/2):third+int(nSearchWords*overlapFac/2)]),

                     ' '.join(docSplit[third*2-int(nSearchWords*overlapFac/2):third*2+int(nSearchWords*overlapFac/2)]),

                     ' '.join(docSplit[-int(nSearchWords*overlapFac):])]

        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]        

        

    elif len(input_ids_all)*overlapFac > 1024:

        nSearchWords = int(np.ceil(nWords/3))

        middle = int(np.ceil(nWords/2))

        docSplit = document.split()

        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 

                     ' '.join(docSplit[middle-int(nSearchWords*overlapFac/2):middle+int(nSearchWords*overlapFac/2)]),

                     ' '.join(docSplit[-int(nSearchWords*overlapFac):])]

        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]

    elif len(input_ids_all)*overlapFac > 512:

        nSearchWords = int(np.ceil(nWords/2))

        docSplit = document.split()

        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), ' '.join(docSplit[-int(nSearchWords*overlapFac):])]

        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]

    else:

        input_ids = [input_ids_all]

    absTooLong = False    

    

    answers = []

    cons = []

    for iptIds in input_ids:

        tokens = QA_TOKENIZER.convert_ids_to_tokens(iptIds)

        sep_index = iptIds.index(QA_TOKENIZER.sep_token_id)

        num_seg_a = sep_index + 1

        num_seg_b = len(iptIds) - num_seg_a

        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        assert len(segment_ids) == len(iptIds)

        n_ids = len(segment_ids)



        if n_ids < 512:

            start_scores, end_scores = QA_MODEL(torch.tensor([iptIds]).to(torch_device), 

                                     token_type_ids=torch.tensor([segment_ids]).to(torch_device))

        else:

            print('****** warning only considering first 512 tokens, document is '+str(nWords)+' words long.  There are '+str(n_ids)+ ' tokens')

            absTooLong = True

            start_scores, end_scores = QA_MODEL(torch.tensor([iptIds[:512]]).to(torch_device), 

                                     token_type_ids=torch.tensor([segment_ids[:512]]).to(torch_device))

        start_scores = start_scores[:,1:-1]

        end_scores = end_scores[:,1:-1]

        answer_start = torch.argmax(start_scores)

        answer_end = torch.argmax(end_scores)

        answer = reconstructText(tokens, answer_start, answer_end+2)

    

        if answer.startswith('. ') or answer.startswith(', '):

            answer = answer[2:]

            

        c = start_scores[0,answer_start].item()+end_scores[0,answer_end].item()

        answers.append(answer)

        cons.append(c)

    

    maxC = max(cons)

    iMaxC = [i for i, j in enumerate(cons) if j == maxC][0]

    confidence = cons[iMaxC]

    answer = answers[iMaxC]

    

    sep_index = tokens_all.index('[SEP]')

    full_txt_tokens = tokens_all[sep_index+1:]

    

    abs_returned = reconstructText(full_txt_tokens)



    ans={}

    ans['answer'] = answer

    if answer.startswith('[CLS]') or answer_end.item() < sep_index or answer.endswith('[SEP]'):

        ans['confidence'] = -1000000

    else:

        ans['confidence'] = confidence

    ans['abstract_bert'] = abs_returned

    ans['abs_too_long'] = absTooLong

    return ans
def create_hit_dictionary(papers_info):

    

    path = []

    for i in range (len(papers_info['id'])):

        path.append(data_path + papers_info['id'][i] + '.json')

    

    hits = []

    for i in range(len(path)):

        f = open(path[i])

        fil = f.read(-1)

        hits.append(fil)

    

    n_hits = len(hits)



    ## collect the relevant data in a hit dictionary

    hit_dictionary = {}

    for i in range(0, n_hits):

        doc_json = json.loads(hits[i])

        idx = str(doc_json['paper_id'])

        hit_dictionary[idx] = doc_json

        #print(doc_json)

        hit_dictionary[idx]['title'] = doc_json['metadata']["title"]

        hit_dictionary[idx]['authors'] = doc_json['metadata']["authors"]



    ## scrub the abstracts in prep for BERT-SQuAD

    for idx,v in hit_dictionary.items():

        abs_dirty = v['abstract']

        # looks like the abstract value can be an empty list

        v['abstract_paragraphs'] = []

        v['abstract_full'] = ''



        if abs_dirty:

            # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph

            # and a new entry that is full abstract text as both could be valuable for BERT derrived QA



            if isinstance(abs_dirty, list):

                for p in abs_dirty:

                    v['abstract_paragraphs'].append(p['text'])

                    v['abstract_full'] += p['text'] + ' \n\n'



            # looks like in some cases the abstract can be straight up text so we can actually leave that alone

            if isinstance(abs_dirty, str):

                v['abstract_paragraphs'].append(abs_dirty)

                v['abstract_full'] += abs_dirty + ' \n\n'

    return hit_dictionary
def searchAbstracts(hit_dictionary, question):

    abstractResults = {}

    for k,v in tqdm(hit_dictionary.items()):

        abstract = v['abstract_full']

        if abstract:

            ans = BERTSQuADPrediction(abstract, question)

            if ans['answer']:

                confidence = ans['confidence']

                abstractResults[confidence]={}

                abstractResults[confidence]['answer'] = ans['answer']

                abstractResults[confidence]['abstract_bert'] = ans['abstract_bert']

                abstractResults[confidence]['idx'] = k

                abstractResults[confidence]['abs_too_long'] = ans['abs_too_long']

                

    cList = list(abstractResults.keys())



    if cList:

        maxScore = max(cList)

        total = 0.0

        exp_scores = []

        for c in cList:

            s = np.exp(c-maxScore)

            exp_scores.append(s)

        total = sum(exp_scores)

        for i,c in enumerate(cList):

            abstractResults[exp_scores[i]/total] = abstractResults.pop(c)

    return abstractResults
def embed_useT(module):

    with tf.Graph().as_default():

        sentences = tf.compat.v1.placeholder(tf.string)

        embed = hub.Module(module)

        embeddings = embed(sentences)

        session = tf.compat.v1.train.MonitoredSession()

    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_useT(path_to_module_useT)
def displayResults(hit_dictionary, answers, question):

    

    question_HTML = '<div style="font-family: Times New Roman; font-size: 28px; padding-bottom:28px"><b>Query</b>: '+question+'</div>'



    confidence = list(answers.keys())

    confidence.sort(reverse=True)

    

    confidence = list(answers.keys())

    confidence.sort(reverse=True)

    



    for c in confidence:

        if c>0 and c <= 1 and len(answers[c]['answer']) != 0:

            if 'idx' not in  answers[c]:

                continue

            rowData = []

            idx = answers[c]['idx']

            title = hit_dictionary[idx]['title']

            authors = hit_dictionary[idx]['authors']



            

            full_abs = answers[c]['abstract_bert']

            bert_ans = answers[c]['answer']

            

            

            split_abs = full_abs.split(bert_ans)

            sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]

            if len(split_abs) == 1:

                sentance_end_pos = len(full_abs)

                sentance_end =''

            else:

                sentance_end_pos = split_abs[1].find('. ')+1

                if sentance_end_pos == 0:

                    sentance_end = split_abs[1]

                else:

                    sentance_end = split_abs[1][:sentance_end_pos]

                

            #sentance_full = sentance_beginning + bert_ans+ sentance_end

            answers[c]['full_answer'] = sentance_beginning+bert_ans+sentance_end

            answers[c]['sentence_beginning'] = sentance_beginning

            answers[c]['sentence_end'] = sentance_end

            answers[c]['title'] = title

        else:

            answers.pop(c)

    

    

    ## now rerank based on semantic similarity of the answers to the question

    cList = list(answers.keys())

    allAnswers = [answers[c]['full_answer'] for c in cList]

    

    messages = [question]+allAnswers

    

    encoding_matrix = embed_fn(messages)

    similarity_matrix = np.inner(encoding_matrix, encoding_matrix)

    rankings = similarity_matrix[1:,0]

    

    for i,c in enumerate(cList):

        answers[rankings[i]] = answers.pop(c)



    ## now form pandas dv

    confidence = list(answers.keys())

    confidence.sort(reverse=True)

    pandasData = []

    ranked_aswers = []

    for c in confidence:

        rowData=[]

        title = answers[c]['title']

        idx = answers[c]['idx']

        rowData += [idx]            

        sentance_html = '<div>' +answers[c]['sentence_beginning'] + " <font color='red'>"+answers[c]['answer']+"</font> "+answers[c]['sentence_end']+'</div>'

        

        rowData += [sentance_html, c]

        pandasData.append(rowData)

        ranked_aswers.append(' '.join([answers[c]['full_answer']]))

    

        pdata2 = pandasData

        

    display(HTML(question_HTML))

    

    df = pd.DataFrame(pdata2, columns = ['id', 'Answer', 'Confidence'])

        

    display(HTML(df.to_html(render_links=True, escape=False)))
query = "Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How long is the incubation period for the virus"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What is the quantity of asymptomatic shedding"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does temperature and humidity affect the tramsmission of 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How long can 2019-nCoV remain viable on inanimate, environmental, or common surfaces"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What types of inanimate or environmental surfaces affect transmission, survival, or  inactivation of 2019-nCov"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "Can the virus be found in nasal discharge, sputum, urine, fecal matter, or blood"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What risk factors contribute to the severity of 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does hypertension affect patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does heart disease affect patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does copd affect patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does smoking affect 2019-nCoV patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does pregnancy affect patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What are the case fatality rates for 2019-nCoV patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What is the case fatality rate in Italy"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What public health policies prevent or control the spread of 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "Can animals transmit 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What animal did 2019-nCoV come from"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What real-time genomic tracking tools exist"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What regional genetic variations (mutations) exist"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What effors are being done in asia to prevent further outbreaks"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What drugs or therapies are being investigated"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What clinical trials for hydroxychloroquine have been completed"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What antiviral drug clinical trials have been completed"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "Are anti-inflammatory drugs recommended"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "Which non-pharmaceutical interventions limit tramsission"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What are most important barriers to compliance"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How does extracorporeal membrane oxygenation affect 2019-nCoV patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What telemedicine and cybercare methods are most effective"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How is artificial intelligence being used in real time health delivery"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What adjunctive or supportive methods can help patients"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What is being done to increase testing capacity or throughput"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What point of care tests are exist or are being developed"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What is the minimum viral load for detection"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What markers are used to detect or track COVID-19"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = 'What collaborations are happening within the research community'

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What are the major ethical issues related pandemic outbreaks"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "How do pandemics affect the physical and/or psychological health of doctors and nurses"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What strategies can help doctors and nurses cope with stress in a pandemic"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What factors contribute to rumors and misinformation"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What is the immune system response to 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "Can personal protective equipment prevent the transmission of 2019-nCoV"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "Can 2019-nCoV infect patients a second time"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)
query = "What is the weighted prevalence of sars-cov-2 or covid-19 in general population"

papers_info = load_pickles(query)

hit_dictionary = create_hit_dictionary(papers_info=papers_info)

answers = searchAbstracts(hit_dictionary, query)

displayResults(hit_dictionary, answers, query)