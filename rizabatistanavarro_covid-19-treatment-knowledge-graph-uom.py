# For downloading resources (pre-trained models and dictionaries) from Google Drive

!conda install -y gdown



# For named entity recognition (NER)

!pip install transformers

!pip install seqeval

!pip install spacy-lookup



# For knowledge graph generation and querying using Grakn 

!pip install ijson

!pip install grakn-client



# For open information extraction (NER)

!pip uninstall -y typing

!pip uninstall -y allennlp

!pip install git+https://github.com/allenai/allennlp-models.git@a9be0f0236c1e6d16f17f5c0e27d35ad7f221a72#egg=allennlp_models 

!pip install git+https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm

!pip install git+https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz#egg=en_core_sci_lg

!pip install git+https://github.com/schlevik/tmdm



# For visualisation 

!pip install pyvis



import csv

import spacy

import gdown

import tarfile

import codecs

import json

import os

import string



from typing import Tuple, List

from collections import OrderedDict

from transformers import pipeline

from spacy_lookup import Entity

from gensim.models import Word2Vec



nlp = spacy.load('en_core_web_sm')
class Paper:

    paper_id = ''

    title = ''

    abstract = ''

    sections = OrderedDict()

    

    def __init__(self, json_object):

        if json_object is not None:

            self.paper_id = json_object['paper_id']

            self.title = json_object['metadata']['title']



            #paper abstract

            abstract_text = ''

            if 'abstract' in json_object:

              paper_abstract_array = json_object['abstract']

              for i in range(0,len(paper_abstract_array)):

                  span = paper_abstract_array[i]

                  abstract_text = abstract_text + span['text']

                  if i < len(paper_abstract_array)-1:

                      abstract_text = abstract_text + ' '

            self.abstract = abstract_text



            #paper sections

            body_sections = OrderedDict()

            if 'body_text' in json_object:

              body_text_array = json_object['body_text']

              for i in range(0,len(body_text_array)):

                  span = body_text_array[i]

                  section_heading = span['section']

                  if section_heading in body_sections:

                      section_text = body_sections[section_heading]

                      body_sections[section_heading] = section_text + ' ' + span['text']

                  else:

                      body_sections[section_heading] = span['text']

            self.sections = body_sections

        

    def get_whole_body_text(self, include_section_headings):

        body_text = ''

        for section in self.sections:

            if include_section_headings == True:

                body_text = body_text + section + '\n'

            body_text = body_text + self.sections[section] + '\n'

        return body_text



class Sentence:

  sentence_text = ''

  nes = set()

  paper_id = ''

  section = ''

    

class NE():

  begin = 0

  end = 0

  ne_type = ''

  

  def __init__(self, begin, end, ne_type):

    self.begin = begin

    self.end = end

    self.ne_type = ne_type



  def __hash__(self):

    return hash(str(self.begin) + '-'+ str(self.end) + ':' + self.ne_type)



  def __eq__(self, other):

    if isinstance(other, NE):

      if self.begin == other.begin and self.end == other.end and self.ne_type == other.ne_type:

        return True

      else:

        return False

    else:

      return False



  def __str__(self):

    return self.ne_type + ': ' + str(self.begin) + ', ' + str(self.end)



class NEToken:

  word = ''

  label = ''

  score = ''

  def __init__(self, word, label):

    self.word = word

    self.label = label



  def __str__(self):

    return self.label + '\t' + self.word
def load_abstracts(file_path):

    abstracts_dict = dict()

    csvfile = open(file_path, 'r')

    

    reader = csv.reader(csvfile, delimiter=',', quotechar='"')

    next(reader)

    i = 1

    for row in reader:

        abstract = Paper(None)

        abstract.paper_id = row[0]

        abstract.title = row[3]

        temp_abstract = row[8]

        if temp_abstract.startswith('Abstract '):

          temp_abstract = temp_abstract.replace('Abstract ', '', 1)

        abstract.abstract = temp_abstract

        abstracts_dict[row[0]] = abstract

        i = i+ 1

    csvfile.close()

    return abstracts_dict



papers = load_abstracts('../input/CORD-19-research-challenge/metadata.csv')



# Function for sentence splitting

def get_sentences(text):  

    doc = nlp(text.strip())

    return list(doc.sents)
# Entity types of interest

DRUG_LABEL = 'Chemical'

DISEASE_LABEL = 'Disease'



# Download and load the deep learning-based NER model trained using Huggingface Transformers

url = 'https://drive.google.com/uc?id=1TbSTVz-WqOZGkOzbMCTBWOAyPwfGrpC-'

output = 'cdr-model.tar.gz'

gdown.download(url, output, quiet=False)



my_tar = tarfile.open('cdr-model.tar.gz')

my_tar.extractall() 

my_tar.close()

os.remove('cdr-model.tar.gz')



CDR_MODEL_PATH = './cdr-model/'

cd_ner = pipeline('ner', model=CDR_MODEL_PATH + 'tf_model.h5', config=CDR_MODEL_PATH + 'config.json', tokenizer='allenai/scibert_scivocab_uncased', device=0)



# Getting the right labels in the model

def map_labels(model_dir_path):

  # load labels file

  labels_filename = model_dir_path + 'labels.txt'

  labels_file = open(labels_filename, 'r')

  labels = labels_file.readlines()

  labels_file.close()



  label_list = []

  for label in labels:

    label_list.append(label.strip())





  # load config file

  config_filename = model_dir_path + 'config.json'

  json_file = open(config_filename, 'r')

  json_object = json.loads(json_file.read())

  json_file.close()



  label_dict = json_object['label2id']

  return label_dict, label_list



cd_dict, cd_labels = map_labels(CDR_MODEL_PATH)



# Load dictionary from a flat list

def load_dictionary(file_path):

  entity_list = []

  file = open(file_path , 'r', encoding = 'utf-8')

  lines = file.readlines()

  for line in lines:

    entity_list.append(line.strip())

  file.close()

  return entity_list



url = 'https://drive.google.com/uc?id=1ATez8emmXUD4ifHxx9nDf5kqOMGD_Ybp'

output = 'Covid-19_Drugs.txt'

gdown.download(url, output, quiet=False)



url = 'https://drive.google.com/uc?id=1-6ki6GTjTJx0ppPqaIDPA4_qHlne1x2-'

output = 'Covid-19_Synonyms.txt'

gdown.download(url, output, quiet=False)



covid_names = load_dictionary('./Covid-19_Drugs.txt')

covid_drugs = load_dictionary('./Covid-19_Synonyms.txt')



disease_entity = Entity(keywords_list=covid_names, label=DISEASE_LABEL)

drug_entity = Entity(keywords_list=covid_drugs, label=DRUG_LABEL)
# Label tokens of recognised named entities using the BIO scheme

def get_tokens(ner_results, lookup_dict, labels_list):

  ne_toks = []

  for ner_result in ner_results:

    ne_token = NEToken(ner_result['word'], ner_result['entity'])

    ne_token.score = ner_result['score']

    entity_label = ner_result['entity']

    value = lookup_dict[entity_label]

    if value < len(labels_list):

      ne_token.label = labels_list[value]

    else:  

      ne_token.label = 'O'

    ne_toks.append(ne_token)

  return ne_toks



# Function for correcting NE labels: making sure an entity's first token is tagged as 'B'

# and propagating the label of an entity's first token to its subwords

def postprocess(tokens):

  new_toks = []

  previous_label = 'O'

  for token in tokens:

    if '[CLS]' not in token.word and '[SEP]' not in token.word:

      if token.label.startswith('I-') and previous_label=='O':

        token.label = 'B-' + token.label.split('-')[1]

      if token.word.startswith('##'):

        if '-' in previous_label:

          token.label = 'I-' + previous_label.split('-')[1]

        else:

          token.label = previous_label

      else:

        previous_label = token.label

      new_toks.append(token)

  

  #go backwards to ensure each tokens in a BI sequence have the same entity type

  previous_entity_type = ''

  for i in range(len(new_toks),0, -1):

    current_tok = new_toks[i-1]

    if current_tok.label.startswith('I-'):

      if previous_entity_type != 'O' and previous_entity_type != '':

        current_tok.label = 'I-' + previous_entity_type

        previous_entity_type = current_tok.label.split('-')[1]

      else:

        previous_entity_type = current_tok.label.split('-')[1]

    elif current_tok.label.startswith('B-') and previous_entity_type != 'O' and previous_entity_type != '':

      current_tok.label = 'B-' + previous_entity_type

      previous_entity_type = 'O'

    elif current_tok.label == 'O':

      previous_entity_type = 'O'



  return new_toks



# Function for consolidating consituent subwords (character n-grams) into a token

def consolidate(bert_tokenized):

  result = []



  for ne_token in bert_tokenized:

      if ne_token.word.startswith('##'):

          result[-1].word += ne_token.word[2:]

      else:

          result.append(ne_token)

  return result



# Function for merging entity labels coming from two different NE recognisers

def merge(toks1, toks2, to_debug):

  merged_toks = []

  if to_debug:

    print(len(toks1))

    print(len(toks2))

  if len(toks1) == len(toks2):

    for tok in toks1:

      merged_toks.append(tok)

    for i in range(0,len(toks2)):

      if merged_toks[i].label == 'O':

        merged_toks[i].label = toks2[i].label

  else:

    for tok in toks1:

      merged_toks.append(tok)

  if to_debug:

    for m in merged_toks:

      print(m.word, m.label)

  return merged_toks



def to_tuples(tokens):

  tuple_list = []

  for token in tokens:

    tuple_list.append((token.word, token.label))

  return tuple_list



# Function for obtaining character offsets of tokens of reconised named entities

def get_offsets(text: str, annotation: List[Tuple[str, str]], init_offset=0, return_last_match=False):

    result = []

    offset = 0

    text = text.lower()



    for token, label in annotation:

        try:

            start = text[offset:].index(token.lower()) + offset

            end = start + len(token)



        except ValueError:

            printable_text = ''.join(c for c in text[offset:] if c in string.printable)

            try:

                matched_index = printable_text.index(token.lower())

            except ValueError:

                return []

            start = matched_index + offset

            matched_string = printable_text[matched_index:matched_index + len(token)]

            char_iter = iter(text[start:])

            diff = 0

            for t in matched_string:

                original_char = next(char_iter)

                while t != original_char:

                    original_char = next(char_iter)

                    diff += 1



            end = start + len(token) + diff

            



        if not label == "O":

            tag, category = label.split('-')

            if tag == "B":

                result.append((start + init_offset, end + init_offset, category))

            elif tag == "I":

                old_start, *rest = result[-1]

                result[-1] = (old_start, end + init_offset, category)

        offset = end

    if return_last_match:

        return result, offset + init_offset

    else:

        return result





def get_offsets_from_sentences(text: str, annotation: List[List[Tuple[str, str]]]):

    offset = 0

    result = []

    for sent in annotation:

        sent_result, offset = get_offsets(text[offset:], sent, init_offset=offset, return_last_match=True)

        result.extend(sent_result)

    return result



# Recognise named entities in a given sentence using dictionaries

def apply_dictionary(nlp, text, entity_type, entity_label):

  resulting_tokens = []

  nlp.add_pipe(entity_type, first=False, name=entity_label, last=True)

  doc = nlp(text)

  previous_label = 'O'

  for token in doc:

    if token._.is_entity:

      if previous_label == 'O':

        resulting_tokens.append(NEToken(token.text, 'B-' + entity_label))

        previous_label = 'B-' + entity_label

      elif previous_label == 'B-' + entity_label or previous_label == 'I-' + entity_label:

        resulting_tokens.append(NEToken(token.text, 'I-' + entity_label))

        previous_label = 'I-' + entity_label

    else:

      resulting_tokens.append(NEToken(token.text, 'O'))

      previous_label = 'O'

  nlp.remove_pipe(entity_label)

  return resulting_tokens



# Recognise named entities in a given sentence using a deep learning-based model

def apply_ner_model(ner, text, label_dict, label_list, to_debug):

  results = ner(text)

  toks = get_tokens(results, label_dict, label_list)

  if to_debug: 

    print('ORIG RESULTS')

    for tok in toks:

      print(tok)

  clean_toks = postprocess(toks) 

  if to_debug:

    print('AFTER POSTPROCESSING')

    for clean_tok in clean_toks:

      print(clean_tok)

  consolidated_toks = consolidate(clean_toks)

  if to_debug:

    print('AFTER CONSOLIDATING')

    for consolidated_tok in consolidated_toks:

      print(consolidated_tok)

  return consolidated_toks



# Attach recognised entities to the sentence

def add_nes(sentence, ne_offsets):

    for ne_offset in ne_offsets:

      if ne_offset[1] - ne_offset[0] > 1:

        ne = NE(ne_offset[0], ne_offset[1], ne_offset[2])

        sentence.nes.add(ne)

    return sentence
# Load pre-trained BioWordVec embeddings

file_dict = {'pubmed_mesh_test.tar.gz':'https://drive.google.com/uc?id=17KLR5bPKtlLlMnrdmcYTAMtbIAPnubmy',

             'pubmed_mesh_test.trainables.syn1neg.npy.tar.gz':'https://drive.google.com/uc?id=17NfQzGJACGLD8zXpPTyaXrAYV3qlj0wl', 

             'pubmed_mesh_test.trainables.vectors_ngrams_lockf.npy.tar.gz':'https://drive.google.com/uc?id=1y2XgHAwm922oun5Vpojojwr-4rnEqb76',

             'pubmed_mesh_test.trainables.vectors_vocab_lockf.npy.tar.gz':'https://drive.google.com/uc?id=10Td9U8-R_b2QxSDQTGp5my48eqMZ6Fqz',

             'pubmed_mesh_test.wv.vectors.npy.tar.gz':'https://drive.google.com/uc?id=1Oc5-QT9qRz_SnNDGRFJLEfR7WqheSevf',

             'pubmed_mesh_test.wv.vectors_ngrams.npy.tar.gz':'https://drive.google.com/uc?id=1fncdQHoxMSMzfi1tVlEvvu57TDfZmoxa',

             'pubmed_mesh_test.wv.vectors_vocab.npy.tar.gz':'https://drive.google.com/uc?id=1EPSOZjtkQXLtqLc953ZNYlRUTAsJaD1t'}



for file in file_dict:

    url = file_dict[file]

    output = file

    gdown.download(url, output, quiet=False)



    my_tar = tarfile.open(file)

    my_tar.extractall('./title_abstract/') 

    my_tar.close()

    os.remove(file)



biowordvec_model = Word2Vec.load('./title_abstract/pubmed_mesh_test')
# Function for transforming terms from Medical Subject Headings (MeSH) into a more natural form

# For example: "Lung Inflammation, Experimental" will be transformed into "Experimental Lung Inflammation" as the latter is more likely to be used in papers.

def transform(name):

    result = name

    if result.count(', ') > 0:

        tokens = result.split(', ')

        result = ''

        for i in range(len(tokens)-1,-1,-1):

            result = result + tokens[i] + ' '

    return result.strip().lower()



# Load MeSH descriptors (main headings)

url = 'https://drive.google.com/uc?id=19PI1slvrMGvWoAv8iS7GJmevhsIxx5vM'

output = 'd2020.bin'

gdown.download(url, output, quiet=False)



mesh_file = open('./d2020.bin', 'r', encoding='utf-8')

mesh_lines = mesh_file.readlines()

mesh_file.close()

os.remove('d2020.bin')



uid_to_terms = dict()

term_to_uid = dict()



record_ctr = 0

cache = []

tokens = []



for line in mesh_lines:

    if line.startswith('*NEWRECORD'):

        cache = []

        record_ctr = record_ctr + 1

    else:

        tokens = line.strip().split(' = ')

        

    if line.startswith('MH ='):

        cache.append(transform(tokens[1]))

    elif line.startswith('PRINT ENTRY ='):

        subtokens = tokens[1].split('|')

        if transform(subtokens[0]) not in cache:

            cache.append(transform(subtokens[0]))

    elif line.startswith('ENTRY ='):

        subtokens = tokens[1].split('|')

        if transform(subtokens[0]) not in cache:

            cache.append(transform(subtokens[0]))

        

    elif line.startswith('UI ='):

        uid = tokens[1]

        uid_to_terms[uid] = cache

        for term in cache:

            if term not in term_to_uid:

                term_to_uid[term] = uid





# Load MeSH supplementary concept record terms

url = 'https://drive.google.com/uc?id=1Sk1nrpgDLOwu3V9217F3g3-itCauEcKo'

output = 'c2020.bin'

gdown.download(url, output, quiet=False)



mesh_file = open('./c2020.bin', 'r', encoding='utf-8')

mesh_lines = mesh_file.readlines()

mesh_file.close()

os.remove('c2020.bin')



uid_to_concepts = dict()



record_ctr = 0

cache = []

tokens = []



for line in mesh_lines:

    if line.startswith('*NEWRECORD'):

        cache = []

        record_ctr = record_ctr + 1

    else:

        tokens = line.strip().split(' = ')

        

    if line.startswith('NM ='):

        cache.append(transform(tokens[1]))

    elif line.startswith('N1 ='):

        if transform(tokens[1]) not in cache:

            cache.append(transform(tokens[1]))

    elif line.startswith('RN ='):

        if tokens[1] != '0' and tokens[1].startswith('EC')==False:

            cache.append(tokens[1])

    elif line.startswith('SY ='):

        subtokens = tokens[1].split('|')

        if transform(subtokens[0]) not in cache:

            cache.append(transform(subtokens[0]))

        

    elif line.startswith('UI ='):

        uid = tokens[1]

        uid_to_concepts[uid] = cache

        for concept in cache:

            if concept not in term_to_uid:

                term_to_uid[concept] = uid
# Function for getting the text span given the begin and end offsets within a sentence

def get_ne_span(sentence_text, ne):

    begin_offset = ne.begin

    end_offset = ne.end

    return (sentence_text[begin_offset:end_offset])



# Function for retrieving the unique ID

# First, exact string matching against MeSH entries is performed

# If no match is found, the most similar term--according to trained word embeddings--is returned

# However, a similar term is accepted only if similarity above or equal to 0.90

def get_closest_match(named_entity):

  named_entity = named_entity.lower()

  closest_match = 'NOT FOUND'

  word_vectors = biowordvec_model.wv

  if named_entity in term_to_uid:

    closest_match = term_to_uid[named_entity] + ':' + named_entity

  else:

    try:

      synonyms = word_vectors.most_similar(named_entity)

      for synonym in synonyms:

        if synonym[1] >= 0.90:

          if synonym[0] in term_to_uid:

            closest_match = term_to_uid[synonym[0]] + ':' + synonym[0]

            break

    except:

      closest_match = 'NOT FOUND'

  return closest_match
NUM_DOCS = 500

nlp = spacy.load('en_core_web_sm')

nlp.remove_pipe('ner')



all_docs_dict = dict()

oie_input_json_file = codecs.open('./oie_inputs.json', 'w+', encoding='utf-8')



ctr = 0

for p in papers:

    doc_dict = dict() 

    doc_dict['title'] = []

    doc_dict['body'] = []



    body_index = 0

    sentences = []

    paper = papers[p]

    # Sentence splitting:

    for s in get_sentences(paper.title):

      sentences.append(s.text)

    

    body_index = len(sentences)



    for s in get_sentences(paper.abstract):

      sentences.append(s.text)



    for s in get_sentences(paper.get_whole_body_text(True)):

      sentences.append(s.text)



    # For each sentence, apply the NER models

    sent_no = 0

    for text in sentences:

      #create Sentence 

      sentence = Sentence()

      sentence.paper_id = paper.paper_id

      sentence.sentence_text = text

      if sent_no < body_index:

        sentence.section = 'title'

      else:

        sentence.section = 'body'

      sentence.nes = set()



      to_debug = False

      #Machine learning-based model

      merged_toks = apply_ner_model(cd_ner, text, cd_dict, cd_labels, to_debug)

      tuples = to_tuples(merged_toks)

      ne_offsets = get_offsets(text, tuples, 0, False)

      new_sentence = add_nes(sentence, ne_offsets)



      #Dictionary-based matching

      disease_dict_toks = apply_dictionary(nlp, text, disease_entity, DISEASE_LABEL)

      drug_dict_toks = apply_dictionary(nlp, text, drug_entity, DRUG_LABEL)



      disease_dict_tuples = to_tuples(disease_dict_toks)

      drug_dict_tuples = to_tuples(drug_dict_toks)



      disease_dict_offsets = get_offsets(text, disease_dict_tuples, 0, False)

      drug_dict_offsets = get_offsets(text, drug_dict_tuples, 0, False)



      new_sentence = add_nes(new_sentence, disease_dict_offsets)

      new_sentence = add_nes(new_sentence, drug_dict_offsets)



      #Saving annotations in JSON for OIE

      sentence_object = dict()

      sentence_object['text'] = new_sentence.sentence_text

      ne_list = []

      for ne in new_sentence.nes:

        ne_span = get_ne_span(new_sentence.sentence_text, ne)

        most_similar_span = get_closest_match(ne_span)

        ne_list.append((ne.begin, ne.end, ne.ne_type, most_similar_span))

      if len(ne_list) > 0:

        sentence_object['named_entities'] = ne_list



      if new_sentence.section in doc_dict:

        section_object = doc_dict[new_sentence.section]

        section_object.append(sentence_object)

        doc_dict[new_sentence.section] = section_object

      else:

        section_object = [sentence_object]

        doc_dict[new_sentence.section] = section_object

      

      sent_no = sent_no + 1

    all_docs_dict[paper.paper_id] = doc_dict

    ctr = ctr + 1

    if ctr >= NUM_DOCS:

        break



json_string = json.dumps(all_docs_dict, ensure_ascii=False, indent = 4).encode('utf-8')

oie_input_json_file.write(json_string.decode())

oie_input_json_file.close()

def chunks(lst, n):

    """Yield successive n-sized chunks from lst."""

    for i in range(0, len(lst), n):

        yield lst[i:i + n]



from collections import defaultdict

from tqdm.notebook import tqdm



from tmdm.allennlp.oie import get_oie_provider

from tmdm.util import load_file, save_file



import math



from tmdm.pipe.tokenizer import IDTokenizer

from scispacy.custom_tokenizer import combined_rule_tokenizer

from tmdm.util import OneSentSentencizer



import en_core_sci_lg

from tmdm.main import default_one_sent_getter



nlp = en_core_sci_lg.load(disable=['ner','parser'])

nlp.tokenizer = IDTokenizer(combined_rule_tokenizer(nlp), getter=default_one_sent_getter)

nlp.add_pipe(OneSentSentencizer())



# how many to annotate with OIE in parallel

batch_size = 100

# only process those sentences with named entities in them

data_prep = [

        (f"{k}/{i}",  # id

         l['text'])  # data

        for k, v in all_docs_dict.items() for i, l in enumerate(v['body']) if 'named_entities' in l

]

print("Number of sentences to process: ", len(data_prep))



outputs = defaultdict(dict)



print("Tokenizing...")

# tokenize sentences to process

docs = list(tqdm(nlp.pipe(data_prep), total=len(data_prep)))

# cuda=0: run on GPU. -1 for CPU

p = get_oie_provider("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", cuda=0)



print("Annotating...")

for batch in tqdm(chunks(docs, batch_size), total=math.ceil(len(docs) / batch_size)):

    annotations = p.annotate_batch(batch)

    for doc, annotations in zip(batch, annotations):

        doc._.oies = annotations
for doc in tqdm(docs):

    idx, sent_id = doc._.id.rsplit("/", 1)

    outputs[idx][int(sent_id)] = doc._._oies



    

ner_file = 'oie_inputs.json'



data_ner = load_file(f'{ner_file}')

final = {}

for id, article in tqdm(data_ner.items()):

    for i, sent in enumerate(article['body']):

        if "named_entities" in sent:

            try:

                oies = outputs[id][i]

            except Exception as e:

                raise e

                oies = [[], []]

            if oies != [[], []]:

                predicates = (sum(1 for _, _, x in oies[0] if x.startswith("V")))

                if len(sent['named_entities']) >= 2:

                    final[f'{id}/{i}'] = {

                        'ner': sent['named_entities'],

                        'oie': oies,

                        'text': sent['text']

                    }

import glob

import os



import sys

from typing import List



from tmdm.main import tmdm_one_sent_pipeline, change_getter

from tmdm.model.oie import Argument

from tmdm.util import load_file, save_file



import loguru





def is_predicate_between_arguments(predicate, arguments):

    is_any_before = any(a[0].i < predicate[0].i for a in arguments)

    is_any_after = any(a[0].i > predicate[0].i for a in arguments)

    is_inbetween = is_any_before and is_any_after

    return is_inbetween





def get_args_with_nes_and_el(oie):

    args_with_ne = [x for x in oie.arguments if len([x for x in x._.get_nes() if "NOT FOUND" not in x.label_])]

    return args_with_ne



def getter(input_data):

    idx, d = input_data

    return idx, d['text']



nlp = en_core_sci_lg.load(disable=['ner', 'parser'])

nlp.tokenizer = IDTokenizer(combined_rule_tokenizer(nlp), getter=default_one_sent_getter)

nlp.add_pipe(OneSentSentencizer())



change_getter(nlp, getter)

print("Tokenizing...")

docs = list(tqdm(nlp.pipe(list(final.items())), total=len(final)))

print(f"Tokenized {len(docs)} docs.")

print("Annotating...")



failed_ner_align = 0

failed_oie_align = 0

filtered = []



for doc in tqdm(docs):

    if doc:

        id = doc._.id

        d = final[id]

        ne = [(s, e, f"{l}|{el}") for s, e, l, el in d['ner']]

        try:

            doc._.oies = d['oie']

        except:

            failed_oie_align += 1

        try:

            doc._.nes = ne

        except:

            failed_ner_align += 1

    if doc._.nes and doc._.oies:

        for oie in doc._.oies:

            args_with_ne: List[Argument] = get_args_with_nes_and_el(oie)

            if len(args_with_ne) > 1 and is_predicate_between_arguments(oie, args_with_ne):

                res = {

                    "text": str(doc),

                    "predicate": str(oie)

                }

                for argument in args_with_ne:

                    nes = [(str(ne), *str(ne.label_).rsplit("|", 1)) for ne in argument._.get_nes() if

                           "NOT FOUND" not in ne.label_]

                    res[f'ARG-{argument.order}'] = {

                        "type": argument.label_,

                        "text": str(argument),

                        "ner": [{

                            "text": text,

                            "type": ner_type,

                            "id": el_id,

                        } for text, ner_type, el_id in nes]

                    }

                filtered.append(res)



print(f"{failed_ner_align} NER misaligned.")

print(f"{failed_oie_align} OIE misaligned.")

print(f"Have {len(filtered)} extractions after filtering.")



new_d = []

for datum in filtered:

         new_datum = {}

         new_datum['text'] = datum['text']

         new_datum['predicate'] = datum['predicate']

         for i,(k, v) in enumerate(((k,v) for k,v in datum.items() if k.startswith("ARG"))):

             new_datum[f'ARG{i}'] = {}

             new_datum[f'ARG{i}']['type'] = v['type']

             new_datum[f'ARG{i}']['text'] = v['text']

             new_datum[f'ARG{i}']['ner_text'] = '; '.join(f"{n['type']}: {n['text']}" for n in v['ner'])

             new_datum[f'ARG{i}']['ner_id'] = '; '.join(n['id'] for n in v['ner'])

             new_datum[f'ARG{i}']['ner_type'] = '; '.join(n['type'] for n in v['ner'])

             new_datum[f"ARG{i}"]['ner_nested'] = v['ner']

             new_d.append(new_datum)





def filter_identical(data):

        ctr = 0

        new_data = []

        for datum in data:

            nes = [v['ner_text'] for k,v in datum.items() if k.startswith('ARG')]

            nes = [n for n in nes if n]

            if  not len(set(nes)) > 1:

                ctr += 1

            else:

                new_data.append(datum)

        return(new_data)

    



new_data = filter_identical(new_d)

save_file(new_data, f"oie_outputs.json")
nlp = spacy.load('en_core_web_sm')

from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer(language='english')





# Predicates extracted by OpenIE are normalised

# First, we use lemmatisation to obtain the lemmatised form of the predicate; if the predicate is a phrase, the root is used

# The stem (obtained using Snowball Stemming) of the lemmatised form is then taken as the normalised predicate

def normalise_predicate(original_predicate):

  normalised = ''

  preprocessed = nlp(original_predicate)

  for w in preprocessed:

    if w.dep_ == 'ROOT':

      lemma = w.lemma_

      stem = stemmer.stem(lemma)

      normalised = stem

      break

  return normalised
pair_frequency_dict = dict()

pair_predicate_dict = dict()

all_predicates = []



for item in new_data:

    predicate = normalise_predicate(item['predicate'])

    arg0 = item['ARG0']

    arg1 = item['ARG1']

      

    arg0_types = arg0['ner_type'].split(';')

    arg1_types = arg1['ner_type'].split(';')



    arg0_ids = arg0['ner_id'].split(';')

    arg1_ids = arg1['ner_id'].split(';')



    for i in range(0,len(arg0_types)):

        for j in range(0,len(arg1_types)):

            ne_types = [arg0_types[i].strip(), arg1_types[j].strip()]

            if DISEASE_LABEL in ne_types and DRUG_LABEL in ne_types:

                arg0_id = arg0_ids[i].split(':')[0].strip()

                arg1_id = arg1_ids[j].split(':')[0].strip()

                pair_str = arg0_id + '-' + arg1_id

                if arg0_id != arg1_id:

                    if predicate not in all_predicates:

                        all_predicates.append(predicate)

                    if pair_str in pair_predicate_dict:

                        pair_predicates = pair_predicate_dict[pair_str]

                        if predicate not in pair_predicates:

                            pair_predicates[predicate] = 1

                            pair_predicate_dict[pair_str] = pair_predicates

                        else:

                            predicate_count = pair_predicates[predicate]

                            predicate_count = predicate_count + 1

                            pair_predicates[predicate] = predicate_count

                            pair_predicate_dict[pair_str] = pair_predicates

                    else:

                        pair_predicates = dict()

                        pair_predicates[predicate] = 1

                        pair_predicate_dict[pair_str] = pair_predicates



                    if pair_str in pair_frequency_dict:

                        pair_count = pair_frequency_dict[pair_str]

                        pair_count = pair_count + 1

                        pair_frequency_dict[pair_str] = pair_count

                    else:

                        pair_frequency_dict[pair_str] = 1





sorted_freq = sorted(pair_frequency_dict.items(), key=lambda x: x[1], reverse=True)



sorted_pair_predicate_dict = dict()

for pair in pair_predicate_dict:

    pair_predicates = pair_predicate_dict[pair]

    sorted_predicates = sorted(pair_predicates.items(), key=lambda x: x[1], reverse=True)

    sorted_pair_predicate_dict[pair] = sorted_predicates
url = 'https://drive.google.com/uc?id=1YsyvBkAcz9b5YyWQxTNOB3HTwaxlG1J5'

output = './Treatment_Predicates_Clustering.txt'

gdown.download(url, output, quiet=False)

treatment_predicates = load_dictionary(output)



url = 'https://drive.google.com/uc?id=19umZ5Uy__4ozafP71xlrbIBQBMcnN13D'

output = './mesh_to_dosage.json'

gdown.download(url, output, quiet=False)

dosage_file = open(output, 'r')

dosage_dict = json.load(dosage_file)

dosage_file.close()



# Function for getting the preferred name of a MeSH entry, given a MeSH identifier

def get_name(mesh_id):

    preferred_name = ''

    if mesh_id in uid_to_terms:

        preferred_name = uid_to_terms[mesh_id][0]

    elif mesh_id in uid_to_concepts:

        preferred_name = uid_to_concepts[mesh_id][0]

    return preferred_name





chemical_dict = dict()

condition_dict = dict()



chemical_json_array = []

condition_json_array = []

treatment_json_array = []



MIN_PAIR_FREQ = 2

MIN_PRED_FREQ = 2



for freq in sorted_freq:

    if freq[1] < MIN_PAIR_FREQ:

        break

    mesh_ids = freq[0].split('-')

    chem_id = ''

    disease_id = ''

    chem_found = False

    disease_found = False

    for mesh_id in mesh_ids:

        if mesh_id in dosage_dict:

            if dosage_dict[mesh_id] != 'Not applicable' and dosage_dict[mesh_id]!= '':

                chem_id = mesh_id

                chem_found = True

        else:

            disease_id = mesh_id

            disease_found = True

    

    if chem_found == True and disease_found == True:

        preds = sorted_pair_predicate_dict[freq[0]]

        for pred in preds:

            if pred[1] < MIN_PRED_FREQ:

                break

            if pred[0] in treatment_predicates:

                if chem_id not in chemical_dict:

                    chem_name = get_name(chem_id)

                    chemical_dict[chem_id] = chem_name

                    chem_object = dict()

                    chem_object['mesh_id'] = chem_id

                    chem_object['chemical_name'] = chem_name

                    dosage_form = dosage_dict[chem_id]

                    chem_object['dosage_form'] = dosage_form

                    chemical_json_array.append(chem_object)

                if disease_id not in condition_dict:

                    disease_name = get_name(disease_id)

                    condition_dict[disease_id] = disease_name

                    disease_object = dict()

                    disease_object['mesh_id'] = disease_id

                    disease_object['condition_name'] = disease_name

                    condition_json_array.append(disease_object)



                treatment_object = dict()

                treatment_object['medication_id'] = chem_id

                treatment_object['disorder_id'] = disease_id

                treatment_json_array.append(treatment_object)



#write the json files to disk

with open('chemicals.json', 'w') as outfile: 

    json.dump(chemical_json_array, outfile) 



with open('conditions.json', 'w') as outfile: 

    json.dump(condition_json_array, outfile) 



with open('treatments.json', 'w') as outfile: 

    json.dump(treatment_json_array, outfile) 
from grakn.client import GraknClient



with GraknClient(uri="167.99.203.112:48555") as client:

    with client.session(keyspace = "cord") as session:

        with session.transaction().write() as transaction:

            #delete existing treatment relations

            delete_query = 'match $c isa chemical; $d isa condition; $t (medication: $c, disorder: $d) isa treatment; delete $t;' 

            transaction.query(delete_query)

            

            #delete existing mesh-id attribute values

            delete_query = 'match $c isa chemical, has mesh-id $mi; delete $mi;' 

            transaction.query(delete_query)

            

            #delete existing chemical-name attribute values 

            delete_query = 'match $c isa chemical, has chemical-name $cn; delete $cn;' 

            transaction.query(delete_query)

            

            #delete existing dosage-from attribute values

            delete_query = 'match $c isa chemical, has dosage-form $df; delete $df;' 

            transaction.query(delete_query)

            

            #delete existing mesh-id attribute values

            delete_query = 'match $c isa condition, has mesh-id $mi; delete $mi;' 

            transaction.query(delete_query)

            

            #delete existing condition-name attribute values 

            delete_query = 'match $c isa condition, has condition-name $cn; delete $cn;' 

            transaction.query(delete_query)

            

            #delete existing chemicals

            delete_query = 'match $c isa chemical; delete $c;' 

            transaction.query(delete_query)

            

            #delete existing conditions

            delete_query = 'match $c isa condition; delete $c;' 

            transaction.query(delete_query)



            transaction.commit()

            

            
import ijson



def condition_template(condition):

    graql_insert_query = 'insert $condition isa condition, has condition-name "' + condition["condition_name"] + '"'

    graql_insert_query += ', has mesh-id "' + condition["mesh_id"] + '"'

    graql_insert_query += ";"

    return graql_insert_query



def chemical_template(chemical):

    graql_insert_query = 'insert $chemical isa chemical, has chemical-name "' + chemical["chemical_name"] + '"'

    graql_insert_query += ', has mesh-id "' + chemical["mesh_id"] + '"'

    graql_insert_query += ', has dosage-form "' + chemical["dosage_form"] + '"'

    graql_insert_query += ";"

    return graql_insert_query



def treatment_template(treatment):

    graql_insert_query = 'match $disorder isa condition, has mesh-id "' + treatment["disorder_id"] + '";'

    graql_insert_query += ' $medication isa chemical, has mesh-id "' + treatment["medication_id"] + '";'    

    graql_insert_query += " insert (disorder: $disorder , medication: $medication) isa treatment;"

    return graql_insert_query



def parse_data_to_dictionaries(input):

    items = []

    with open(input["data_path"] + ".json") as data:

        for item in ijson.items(data, "item"):

            items.append(item)

    return items



def load_data_into_grakn(input, session):

    items = parse_data_to_dictionaries(input)



    for item in items:

        with session.transaction().write() as transaction:

            graql_insert_query = input["template"](item)

            print("Executing Graql Query: " + graql_insert_query)

            transaction.query(graql_insert_query)

            transaction.commit()



    print("\nInserted " + str(len(items)) + " items from [ " + input["data_path"] + "] into Grakn.\n")





def build_covid19_graph(inputs):

    with GraknClient(uri="167.99.203.112:48555") as client:

        with client.session(keyspace = "cord") as session:

            for input in inputs:

                print("Loading from [" + input["data_path"] + "] into Grakn ...")

                load_data_into_grakn(input, session)



inputs = [

    {

        "data_path": "conditions",

        "template": condition_template

    },

    {

        "data_path": "chemicals",

        "template": chemical_template

    },

    {

        "data_path": "treatments",

        "template": treatment_template

    }    

]



build_covid19_graph(inputs)
# Function that can be called to query the knowledge graph

def query_graph(drug='', disease=''):

    answers = []

    chem_name = '$chemname'

    chem_name_query = ''

    if drug != '':

        chem_name_query = '  $chemname "' + drug.lower() + '";'

        

    cond_name = '$condname'

    cond_name_query = ''

    if disease != '':

        cond_name_query = '  $condname "' + disease.lower() + '";'

    

    with GraknClient(uri="167.99.203.112:48555") as client:

        with client.session(keyspace = "cord") as session:

            with session.transaction().read() as transaction:

                query = [

                    'match ',

                    '  $chem isa chemical, has mesh-id $chemid, has chemical-name $chemname, has dosage-form $doseform;',

                    '  $dose-form contains "Oral";',

                    chem_name_query ,

                    '  $cond isa condition, has mesh-id $condid, has condition-name $condname;',

                    cond_name_query ,

                    '  (medication: $chem, disorder: $cond) isa treatment;',

                    'get $chemid, $chemname, $doseform, $condid, $condname;'

                ]



                print("\nQuery:\n", "\n".join(query))

                query = "".join(query)

                iterator = transaction.query(query)



                for answer in iterator:

                    chemname = answer.get("chemname")

                    chemid = answer.get("chemid")

                    doseform = answer.get("doseform")

                    condname = answer.get("condname")

                    condid = answer.get("condid")

                    answers.append((chemid.value(), chemname.value(), doseform.value(), condid.value(), condname.value()))



    return answers

# This will return the entire knowledge graph

answers = query_graph()

print(answers)



# This will return diseases treated by the given drug

#answers = query_graph(drug='oxygen')



# This will return drugs that have been investigated to treat the given disease

#answers = query_graph(disease='cough')
import networkx as nx



# Function for transforming the results of querying the knowledge graph above, into a graph

def to_graph(answers):

    graph = nx.Graph()

    for answer in answers:

        graph.add_node(answer[0], name=answer[1], type='oral_drug')

        graph.add_node(answer[3], name=answer[4], type='condition')

        graph.add_edge(answer[0], answer[3])

    return graph

import os

import shutil

import tempfile

from pyvis.network import Network

from typing import List



class PyVisPrinter:

    """Class to visualise a (serialized) dataset entry."""



    def __init__(self, path=None):

        self.path = tempfile.mkdtemp(prefix='vis-', dir='.') or path

        

    def clean(self):

        shutil.rmtree(self.path)



    def print_graph(self, graph: nx.Graph, filename):



        vis = Network(bgcolor="#222222",

                      width='100%',

                      font_color="white", notebook=False)

        

        for idx, (node, data) in enumerate(graph.nodes(data=True)):

            vis.add_node(

                node,

                title=data['name'],

                label=data['name'],

                color='yellow' if data['type'] == 'condition' else 'green' if data['type'] == 'oral_drug' else 'blue'

            )



        for i, (source, target) in enumerate(graph.edges()):

            if source in vis.get_nodes() and target in vis.get_nodes():

                vis.add_edge(source, target)

            else:

                print("source or target not in graph!")



        name = os.path.join(self.path, filename)

        return vis

    



graph = to_graph(answers)

p = PyVisPrinter()

v = p.print_graph(graph, 'cord_graph.html')

v.show('cord_graph.html')