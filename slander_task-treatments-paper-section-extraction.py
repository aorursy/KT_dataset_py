# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Load metadata and filter to papers containing covid-19 terms in title or abstract

text_files = glob.glob('../input/coronawhy-plus/v6_text/*tsv')

metadata_df = pd.read_csv('../input/coronawhy/clean_metadata.csv', index_col=0)

metadata_df.loc[:, 'title_abstract'] = metadata_df.loc[:, 'title'].str.lower() + ' ' + metadata_df.loc[:, 'abstract'].str.lower()
metadata_df.loc[:, 'title_abstract'] = metadata_df.loc[:, 'title_abstract'].fillna('')

covid19_df = metadata_df.loc[metadata_df.title_abstract.str.contains('covid-19|sars-cov-2|2019-ncov|sars coronavirus 2|2019 novel coronavirus')]

print(covid19_df.shape)

covid19_sha_list = covid19_df.sha.tolist()
"""
Load annotation dataframe snapshot.
"""

annot_df = pd.read_csv('../input/covid-pc-task-study-design-annotation-200412/Study_Design_Annotation_Snapshot_4_12_20.csv')

"""
Edit columns to make more usable.
"""

#Rename columns
annot_df_old_columns = annot_df.columns.tolist()
print(annot_df_old_columns)

annot_df_new_columns = [
    'assignee',
    'cord_uid',
    'sha',
    'title',
    'url',
    'in_silico',
    'in_vitro',
    'in_vivo',
    'system_review_ma_rct',
    'rct',
    'non_rct',
    'historical_comparator',
    'descriptive_study',
    'system_review_ma_non_rct',
    'other'
                       ]
annot_df_edit = annot_df.copy()
annot_df_edit.columns = annot_df_new_columns

#Drop irrelevant columns
annot_df_edit = annot_df_edit.drop(['assignee','title', 'url', 'other'], axis=1)

#Generate a sum column to separate non-annotated papers from annotated
annot_df_edit['sum'] = annot_df_edit.sum(axis=1)
print("Breakdown of paper annotations:")
print(annot_df_edit['sum'].value_counts())
annot_df_annotated = annot_df_edit.loc[annot_df_edit['sum'] != 0]
ann_shas = annot_df_annotated.sha.tolist()
ann_shas = [i.split(';')[0] for i in ann_shas]

print("Annotated sha ids:")
print(len(ann_shas))

ann_cord_uids = annot_df_annotated.cord_uid.tolist()
print("Annotated cord_uids:")
print(len(ann_cord_uids))
"""
Functions to extract section headers and the text within sections from the v6_text.json in the coronawhy-plus dataset produced my Mike Honey

"""

def generate_section_dict(text_df, paper_sha_list, sections):
    """
    Returns a dict of {paper_id: section: sentences} for specified paper id and specified sections
    """
    
    text_df_with_ids = text_df.loc[text_df.paper_id.isin(paper_sha_list)]
    text_df_of_sections = text_df_with_ids.loc[text_df_with_ids.section.isin(sections)]
    section_dict = text_df_of_sections.groupby('paper_id')['sentence'].apply(list).to_dict()

    #concatenate sentences
    section_concat_sentences_dict = {k: " ".join(v) for k,v in section_dict.items()}
    
    return section_concat_sentences_dict
   
def process_text_tsv_files(text_files, paper_sha_list, sections_oi):

    """
    Iterate through v6_text.json files, extracting papers by sha_id and specified sections.
    
    """
    master_dict = {}
      
    for text_file in text_files:
        print("Processing %s..." % text_file)
        text_df = pd.read_csv(text_file, sep='\t')
        text_df.loc[:, 'sentence'] = text_df.loc[:, 'sentence'].astype(str)
        text_df.loc[:, 'section'] = text_df.loc[:, 'section'].astype(str)

        for section in sections_oi:
            print("Extracting %s section..." % section)
            
            if section not in master_dict.keys():
                master_dict[section] = {}
                
            tmp_dict = generate_section_dict(text_df, paper_sha_list, [section])
            for k, v in tmp_dict.items():
                master_dict[section].setdefault(k, []).append(v)
        
    return master_dict

def extract_paper_section_headers(text_files, paper_sha_ids):
    master_dict = {}
    for text_file in text_files:
        print("Processing %s..." % text_file)
        text_df = pd.read_csv(text_file, sep='\t')
        
        text_df_for_ids = text_df.loc[text_df.paper_id.isin(paper_sha_ids)]
        section_dict = text_df_for_ids.groupby('paper_id')['section'].apply(list).to_dict()
        section_dict_unique = {k: list(set(v)) for k,v in section_dict.items()}
        
        for paper_id, sections in section_dict_unique.items():
            for section in sections:
                master_dict.setdefault(paper_id, set([])).add(section)
    
    return master_dict

def extract_paper_sections(text_files, paper_sha_ids):
    
    master_dict = {}
    for text_file in text_files:
        print("Processing %s..." % text_file)
        text_df = pd.read_csv(text_file, sep='\t')
        
        text_df_for_ids = text_df.loc[text_df.paper_id.isin(paper_sha_ids)]
        
        sha_text_dict = dict(tuple(text_df_for_ids.groupby('paper_id')))
        for paper_sha, paper_df in sha_text_dict.items():
            if paper_sha not in master_dict:
                master_dict[paper_sha] = {}
                
            section_sentences_dict = dict(tuple(paper_df[['section', 'sentence']].groupby('section')))
            for section, sentence_df in section_sentences_dict.items():
                sentence_df.loc[:, 'sentence'] = sentence_df.sentence.fillna('')
                sentences = sentence_df.sentence.tolist()
                
                current_sentences = master_dict[paper_sha].get(section, '')
                new_sentences = current_sentences + ' ' + ' '.join(sentences)
                master_dict[paper_sha][section] = new_sentences
                
    return master_dict

#covid19_paper_sections = extract_paper_section_headers(text_files, covid19_sha_list)

annot_paper_sections = extract_paper_section_headers(text_files, ann_shas)
"""
Section counts
"""

from collections import defaultdict
import re

section_counts_dict = defaultdict(int)
#for sha, sections in covid19_paper_sections.items():
for sha, sections in annot_paper_sections.items():
    for section in sections:
        section_counts_dict[section] += 1
        
"""
Include:
methods
results
statistics

Exclude:
introduction
discussion
funding
conclusions

"""
unique_sections = [str(i) for i in section_counts_dict.keys()]
unique_sections
"""
Functions to generate regex match patterns from synonymous words/phrases for filtering subject headers

"""

def extract_regex_pattern(section_list, pattern):
    r = re.compile(pattern)
    extracted_list = list(filter(r.match, section_list))
    remaining_list = list(set(section_list) - set(extracted_list))
    
    return remaining_list, extracted_list

def construct_regex_match_pattern(terms):
    terms = ['.*%s.*' % i for i in terms]
    pattern = '|'.join(terms)
    return pattern

#Examples
#figure and table references
r = re.compile(".*figref|.*tabref|.*figure")
remaining_list, extracted_list = extract_regex_pattern(unique_sections, r)
print("Figure and table references:")
print(extracted_list)

#author
r = re.compile(".*author")
remaining_list, extracted_list = extract_regex_pattern(unique_sections, r)
print("Sections containing author:")
print(extracted_list)
"""
These are the sections names currently being filtered out.

"""

exclusion_regex_pattern_terms = [
    'discussion', 
    'conclusion', 
    'conflicts of interest',
    'conflict of interest',
    'fund', 
    'ideas and opinions', 
    'article in press', 
    'legal aspects', 
    'acknowledgement',
    'acknowledgment',
    'declaration of', 
    'implications of all the available evidence', 
    'research in context', 
    'author',
    'interpretation',
    'competing interests',
    'references',
    'article in press',
    'disclaimer',
    'contributions',
    'disclosure',
    'references',
    'looking ahead',
    'summarizing the findings',
    'literature',
    'history',
    'future',
    'historic',
    'editor',
    'contributors',
    'license'
]
exc_terms = construct_regex_match_pattern(exclusion_regex_pattern_terms)
incl_sections, excl_sections = extract_regex_pattern(unique_sections, exc_terms)

with open('excl_paper_sections.txt', 'w') as f:
    for excl_section in excl_sections:
        f.write('%s\n' % excl_section)

#Write sorted counts of all sections (covid19 papers) to text file
with open('annot_paper_sections_by_freq.txt', 'w') as f:
    for i in sorted(section_counts_dict.items(), key=lambda k_v: k_v[1], reverse=True):
        f.write('%s\t%s\n' % (i[0], i[1]))
#covid19_shas_with_sections = covid19_paper_sections.keys()
annot_shas_with_sections = annot_paper_sections.keys()
#sample_papers_without_sections_sha_list = list(set(covid19_sha_list) - set(covid19_shas_with_sections))[:20]
sample_papers_without_sections_sha_list = list(set(ann_shas) - set(annot_shas_with_sections))[:20]
sample_papers_without_sections_sha_list
papers_without_sections_dict = extract_paper_section_headers(text_files, sample_papers_without_sections_sha_list)
#sha_section_texts = extract_paper_sections(text_files, covid19_shas_with_sections)
sha_section_texts = extract_paper_sections(text_files, annot_shas_with_sections)
"""
Generate a JSON containing corpus papers:
{
    paper_sha: concatenated text from all relevant sections
}
"""

sha_filtered_and_concat_texts = {}

for sha, section_texts in sha_section_texts.items():
    
    all_section_texts = []
    for section, text in section_texts.items():
        if section not in excl_sections:
            all_section_texts.append(text)
        
    sha_filtered_and_concat_texts[sha] = ' '.join(all_section_texts)

print(len(sha_filtered_and_concat_texts))
#with open('covid19_corpus_paper_sections_200410.json', 'w') as f:
with open('ann_corpus_paper_sections_200417.json', 'w') as f:
    json.dump(sha_section_texts, f)
sample_papers_without_sections_sha_list = [i.split(';')[0] for i in sample_papers_without_sections_sha_list]

papers_without_sections_text_dict = extract_paper_sections(text_files, sample_papers_without_sections_sha_list)
papers_without_sections_text_dict
"""
The aggregate file is too big to load into the notebook directly, but it can be parsed with ijson


"""


!pip install ijson

import ijson
"""

v6_text.json individual files

ijson items:
Keys:
dict_keys(['paper_id', 'language', 'section', 'sentence', 'lemma', 'UMLS', 'GGP', 'SO', 'TAXON', 'CHEBI', 'GO', 'CL', 'DNA', 'CELL_TYPE', 'CELL_LINE', 'RNA', 
'PROTEIN', 'DISEASE', 'CHEMICAL', 'CANCER', 'ORGAN', 'TISSUE', 'ORGANISM', 'CELL', 'AMINO_ACID', 'GENE_OR_GENE_PRODUCT', 'SIMPLE_CHEMICAL', 'ANATOMICAL_SYSTEM', 
'IMMATERIAL_ANATOMICAL_ENTITY', 'MULTI-TISSUE_STRUCTURE', 'DEVELOPING_ANATOMICAL_STRUCTURE', 'ORGANISM_SUBDIVISION', 'CELLULAR_COMPONENT', 'PATHOLOGICAL_FORMATION', 
'ORGANISM_SUBSTANCE', 'sentence_id'])

Example item:
{'paper_id': '566b5c62fc77292ebe09295d59e7fbf6fc914260', 'language': 'en', 'section': 'survey methodology', 
'sentence': 'Older children who obtained parental consent were given diaries with simplified language to fill in on their own (see Table S1 for more details).', 
'lemma': ['old', 'child', 'who', 'obtain', 'parental', 'consent', 'be', 'give', 'diary', 'with', 'simplify', 'language', 'to', 'fill', 'in', 'on', '-PRON-', 'own', '(', 'see', 'table', 's1', 'for', 'more', 'detail', ')', '.'], 
'UMLS': ['Child', 'parent', 'Consent', 'Diaries', 'Programming Languages'], 
'GGP': [], 'SO': [], 'TAXON': [], 'CHEBI': [], 'GO': [], 'CL': [], 'DNA': [], 'CELL_TYPE': [], 'CELL_LINE': [], 'RNA': [], 
'PROTEIN': [], 'DISEASE': [], 'CHEMICAL': [], 'CANCER': [], 'ORGAN': [], 'TISSUE': [], 'ORGANISM': ['children'], 'CELL': [], 'AMINO_ACID': [], 'GENE_OR_GENE_PRODUCT': [], 
'SIMPLE_CHEMICAL': [], 'ANATOMICAL_SYSTEM': [], 'IMMATERIAL_ANATOMICAL_ENTITY': [], 'MULTI-TISSUE_STRUCTURE': [], 'DEVELOPING_ANATOMICAL_STRUCTURE': [], 
'ORGANISM_SUBDIVISION': [], 'CELLULAR_COMPONENT': [], 'PATHOLOGICAL_FORMATION': [], 'ORGANISM_SUBSTANCE': [], 'sentence_id': '566b51354014260'}

"""

aggregate_text_file = '../input/coronawhy/v6_text.json'

with open(aggregate_text_file) as f:
    objects = ijson.items(f, "item")
    for obj in objects:
        print(obj['paper_id'])

