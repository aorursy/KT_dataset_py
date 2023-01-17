!pip install git+git://github.com/javiersastre/ask_jc.git
import os

import sys

import pandas as pd



from ask_jc.paper.paper_sentence_extractor import PaperSentenceExtractor

from ask_jc.paper.paper_sentence_dataframe import PaperSentenceDataFrame

from ask_jc.indexer.bm25_indexer import Bm25Indexer

from ask_jc.searcher.bm25_searcher import Bm25Searcher
pd.set_option('display.max_colwidth', 0) # Do not truncate contents of tables when printing them



recompute_index = False # do not recompute index if already present

max_docs = 0 # max_docs 0 means there is no limit on the maximum amount of documents to return per query

threshold = 0.1 # keep documents that have some score at least slightly above 0

# datasets = ['biorxiv_medrxiv', 'comm_use_subset', 'custom_license', 'noncomm_use_subset'] # Kaggle runs out of RAM with all datasets

datasets = ['biorxiv_medrxiv', 'custom_license'] # As an example, we used here these 2 datasets only so the notebook can run in Kaggle

cord19_folder = os.path.join('..', 'input', 'CORD-19-research-challenge')

cord19index_folder = os.path.join('..', 'input', 'cord19index')

output_folder = os.path.join('..', 'output')

paper_folders = [os.path.join(cord19_folder, folder, folder, 'pdf_json') for folder in datasets]

sentences_folder = os.path.join(output_folder, 'sentences')

sentence_folders = [os.path.join(sentences_folder, folder) for folder in datasets]

sentence_file = os.path.join(cord19index_folder, 'sentences.gzip') # use precomputed sentences dataframe if exists

if recompute_index or not os.path.exists(sentence_file):

    sentence_file = os.path.join(output_folder, 'sentences.gzip') # otherwise generate it in output folder

index_file = os.path.join(cord19index_folder, 'bm25_index.pkl') # use precomputed BM25 index if exists

if recompute_index or not os.path.exists(index_file):

    index_file = os.path.join(output_folder, 'bm25_index.pkl') # otherwise generate it in output folder

metadata_file = os.path.join(cord19_folder, 'metadata.csv')

metadata = pd.read_csv(metadata_file)
if recompute_index or not os.path.exists(sentence_file):

    extractor = PaperSentenceExtractor()

    for p, s in zip(paper_folders, sentence_folders):

        extractor.extract_and_save(p, s)

    del(extractor)
if recompute_index or not os.path.exists(sentence_file):

    sentence_df = PaperSentenceDataFrame()

    sentence_df.load_folder(sentences_folder)

    sentence_df.save_pickle(sentence_file)

    del(sentence_df)
if recompute_index or not os.path.exists(index_file):

    indexer = Bm25Indexer()

    indexer.index_dataframe(sentence_file)

    indexer.save_index(index_file)

    del(indexer) # Delete indexer once the index is built, since we will use the searcher from now on on the saved index
searcher = Bm25Searcher()

searcher.load_sentences_and_index(sentence_file, index_file)
hits = searcher.search('example query', max_docs=max_docs, threshold=0.1)

hits
#note that the number of documents found will appear at the bottom to help you customize which terms you'd like to include in final search

#POPULATION

hitsP = searcher.search('covid-19 coronavirus', max_docs=max_docs, threshold=threshold)

hitsP
#INTERVENTION

hitsI = searcher.search('molecular based assays serological tests RNA-based antibodies', max_docs=max_docs, threshold=threshold)

hitsI
#COMPARISON

hitsC = searcher.search('reverse transcription-polymerase chain reaction (RT-PCR) immunoassay chest xray', max_docs=max_docs, threshold=threshold)

hitsC
#OUTCOME

hitsO = searcher.search('rapid point-of-care detection diagnosis', max_docs=max_docs, threshold=threshold)

hitsO
#TYPE OF STUDY 

hitsT = searcher.search('comparison', max_docs=max_docs, threshold=threshold)

hitsT
def get_common_paper_results(hits_list, metadata):

    unique_ids_list = [hits[['paper_id']].drop_duplicates() for hits in hits_list]

    common_paper_ids = pd.concat(unique_ids_list, axis=1, join='inner').iloc[:, 0].to_frame()

    common_hits_list = [hits.loc[hits['paper_id'].isin(common_paper_ids['paper_id'])] for hits in hits_list]

    common_hits_list = [common_hits.merge(metadata, how='left', left_on='paper_id', right_on='sha') for common_hits in common_hits_list]

    common_hits_list = [common_hits[['doi', 'title','authors', 'journal','publish_time','abstract', 'paper_id', 'sentence']] for common_hits in common_hits_list]

    return common_hits_list
#identify which PICOT terms to use for final search, this only uses (P)atient, (I)ntervention,(C)omparison

final_search = get_common_paper_results([hitsP,hitsI,hitsC], metadata)
#use PICOT terms passed in final_search to pass to original queries to explore the sentences used to link the common articles

resultsP, resultsI, resultsC = get_common_paper_results([hitsP, hitsI, hitsC], metadata)
#save each result to a csv to manually review each sentence and have the list of paper references

resultsP.to_csv('resultsP.csv')

resultsI.to_csv('resultsI.csv')

resultsC.to_csv('resultsC.csv')
for paper_id in resultsP[['paper_id']]: # For each paper in the intersection

    print(resultsP[['doi', 'title','authors', 'journal','publish_time','abstract', 'paper_id']]) # Print paper reference

    print(resultsP['sentence']) # Print P query sentences

    print(resultsI['sentence']) # Print I query sentences

    print(resultsC['sentence']) # Print C query sentences
