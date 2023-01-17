# Set-up: uncomment and run selection for  

! pip install scispacy

! pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz



# NOT WORKING: nlp.load('en_core_sci_sm') 

# Seting manually

! mkdir /kaggle/working/scispacy-models

! wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz 

! mv en_core_sci_sm-0.2.4.tar.gz /kaggle/working/scispacy-models 

! tar xvfz /kaggle/working/scispacy-models/en_core_sci_sm-0.2.4.tar.gz -C /kaggle/working/scispacy-models



# Model main folder for the output:

! mkdir /kaggle/working/models
import argparse

import sys

import pandas as pd

import utils as utils

import os

import pickle

import time

from gensim.corpora.dictionary import Dictionary

from gensim.corpora import MmCorpus

from gensim.models import AuthorTopicModel, LdaModel, LdaMulticore

from recommender import Recommender

from commands import Commands
def load_model(path_to_model, model_type):

    if model_type == 'lda':

        return LdaModel.load(path_to_model)

    elif model_type == 'lmc':

        return LdaMulticore(path_to_model)

    else:

        return AuthorTopicModel.load(path_to_model)





def load_docind(path_to_docind):

    with open(path_to_docind, 'r', encoding='utf-8') as f:

        docind = [doci for doci in f.readlines()]

    return docind





def save_model(path_to_model, model, corpus, docind, authors=None):

    name = os.path.basename(path_to_model)

    if not os.path.exists(path_to_model):

        os.mkdir(path_to_model)

    print("Saving model ({})".format(path_to_model))

    model.save(path_to_model + "/" + name)

    MmCorpus.serialize(path_to_model + '/corpus.mm', corpus)

    with open(path_to_model + '/corpus.mm.docind', 'w', encoding='utf-8') as f:

        for doci in docind:

            f.write(doci+"\n")

    if authors is None:

        pickle.dump(authors, open(path_to_model + '/authors.pkl', 'wb'))

DEBUG = False



def create_model(args):

    

    # load data and filter data

    # Select interesting fields from metadata file

    fields = ['cord_uid','title', 'authors', 'publish_time', 'abstract', 'journal','url', 'has_pdf_parse',

              'has_pmc_xml_parse', 'pmcid', 'full_text_file', 'sha']



    # Extract selected fields from metadata file into dataframe

    df_mdata = pd.read_csv(args['data'], skipinitialspace=True, index_col='cord_uid', usecols=fields)



    # WARNING: cord_uid is described as unique, but c4u0gxp5 is repeated. So I remove one of this

    df_mdata = df_mdata.loc[~df_mdata.index.duplicated(keep='first')]

    df_mdata['publish_time'] = pd.to_datetime(df_mdata['publish_time'], errors="coerce")

    df_mdata['publish_year'] = df_mdata['publish_time'].dt.year

    df_mdata = df_mdata[df_mdata['abstract'].notna()]

    df_mdata = df_mdata[df_mdata['authors'].notna()]

    # df_mdata = df_mdata[df_mdata['sha'].notna()]

    df_mdata['authors'] = df_mdata['authors'].apply(lambda row: str(row).split('; '))



    if args['only_covid']:

        print("Filtering corpus: Remove papers from metadata")

        df_mdata = utils.filter_covid(df_mdata)

    relevant_time = df_mdata.publish_year.between(args['above_year'], args['below_year'])

    df_mdata = df_mdata[relevant_time]



    if DEBUG:

        df_mdata = df_mdata.head(n=50)



    # prepare data

    model = None

    if args['load_model'] is not None:

        print("Loading pretrained model")

        model = load_model(args['load_model'], args['model'])



    # authors

    author2docs = None

    if args['load_authors'] is not None:

        print("Loading precomputed author2doc")

        author2docs = pickle.load(args['load_authors'])



    # corpus

    if args['load_corpus'] is not None:

        print("Loading precomputed corpus")

        corpus = MmCorpus(args['load_corpus'])

        docind = load_docind(args['load_corpus'] + '.docind')

        id2word = model.id2word

    else:

        print("Preprocessing metadata as corpus")

        texts, author2docs, docind = utils.preprocess_corpus(df_mdata, model_name=args['spacy_model'])

        print("Number of abstracts loaded:", len(texts))



        # Create Dictionary and transform corpus

        id2word = Dictionary(texts)

        id2word.filter_extremes(no_below=args['no_below'], no_above=args['no_above'])

        corpus = [id2word.doc2bow(text) for text in texts]

    print("Topic Model using vocabulary size of", len(id2word))



    # Model

    start = time.time()

    if args['load_model'] is None and args['model'] == 'atm':

        print("Building ATM model")

        model = AuthorTopicModel(corpus, num_topics=args['num_topics'], id2word=id2word, author2doc=author2docs)

    elif args['load_model'] is None and args['model'] == 'lda':

        print("Building LDA model")

        model = LdaModel(corpus, num_topics=args['num_topics'], id2word=id2word)

    elif args['load_model'] is None and args['model'] == 'lmc':

        print("Building LMC model")

        model = LdaMulticore(corpus, num_topics=args['num_topic'], id2word=id2word, workers=4)

    elif args['load_model'] is None:

        print("Selected model is not implemented")

        sys.exit(0)

    end = time.time()

    print('Model building time: ', round((end - start) / 60, 2))



    if args['save_model'] is not None:

        save_model(args['save_model'], model, corpus, docind, author2docs)



    # build the recommender system

    print('Preparing recommender')

    start = time.time()

    recommender = Recommender(model, df_mdata, corpus, args['model'])

    end = time.time()

    print('Recommender preparation time: ', round((end - start) / 60, 2))



    # Prompt

    if not args["no_interaction"]:

        print('Starting command interface')

        cmd = Commands()

        cmd.init(recommender)

        cmd.cmdloop()
config = {'data': '/kaggle/input/CORD-19-research-challenge/metadata.csv',

          'model': 'lda',

          'load_model': None,

          'load_corpus': None,

          'load_authors': None,

          'num_topics': 170,

          'no_below': 5,

          'no_above': 50,

          'only_covid': False,

          'below_year': 2020,

          'above_year': 1950,

          'save_model': '/kaggle/working/models/lda.topics-170.fr-5-50.all-abstracts.model',

          'no_interaction': True,

          'encoding': 'utf-8',

          'spacy_model': '/kaggle/working/scispacy-models/en_core_sci_sm-0.2.4/en_core_sci_sm/en_core_sci_sm-0.2.4'}
create_model(config)