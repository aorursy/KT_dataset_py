# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
"""
Using v6_text files produced by Mike Honey.

Each row is a sentence with a section label.

"""
text_files = glob.glob('../input/coronawhy-plus/v6_text/*tsv')

"""
Use 999 paper ids from annotation table for Task - Treatments team.
"""

annotation_task_table = '../input/annotation-treatment-task/_AnnotationSample_Task-Treatment - AnnotationSample_2020_04_02.tsv'

annot_task_df = pd.read_csv(annotation_task_table, sep='\t', index_col=0)

#Skip first uid, which is header (displaced by instruction in table)
paper_uid_list = annot_task_df.index.tolist()[1:]

#Skip first sha
paper_sha_list = annot_task_df.iloc[:, 0].tolist()[1:]

#Some papers have multiple shas, delimited by ';'.  Taking the first one for now, will come back if I cannot locate the text for it.
edit_paper_sha_list = [i.split(';')[0] for i in paper_sha_list]

def generate_section_dict(text_df, paper_sha_list, sections):
    
    text_df_with_ids = text_df.loc[text_df.paper_id.isin(paper_sha_list)]
    text_df_of_sections = text_df_with_ids.loc[text_df_with_ids.section.isin(sections)]
    section_dict = text_df_of_sections.groupby('paper_id')['sentence'].apply(list).to_dict()

    #concatenate sentences

    section_concat_sentences_dict = {k: " ".join(v) for k,v in section_dict.items()}
    
    return section_concat_sentences_dict
   
def process_text_tsv_files(text_files, paper_sha_list, sections_oi):
    
    master_dict = {}
      
    for text_file in text_files:
        print("Processing %s..." % text_file)
        text_df = pd.read_csv(text_file, sep='\t')
        text_df.loc[:, 'sentence'] = text_df.loc[:, 'sentence'].astype(str)
        
        for section in sections_oi:
            print("Extracting %s section..." % section)
            
            if section not in master_dict.keys():
                master_dict[section] = {}
                
            tmp_dict = generate_section_dict(text_df, paper_sha_list, [section])
            for k, v in tmp_dict.items():
                master_dict[section].setdefault(k, []).append(v)
        
    return master_dict

sections_oi = ['methods', 'results']

test_dict = process_text_tsv_files(text_files, edit_paper_sha_list, sections_oi)
results_ids = (set(list(test_dict['results'])))
methods_ids = set(list(test_dict['methods']))
ids_with_no_results = list(set(edit_paper_sha_list) - set(results_ids))
ids_with_no_methods = list(set(edit_paper_sha_list) - set(methods_ids))
"""
Get paper section headers

 if section not in master_dict.keys():
                master_dict[section] = {}
                
            tmp_dict = generate_section_dict(text_df, paper_sha_list, [section])
            for k, v in tmp_dict.items():
                master_dict[section].setdefault(k, []).append(v)
"""

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
sections_of_papers_with_no_results = extract_paper_section_headers(text_files, ids_with_no_results)
sections_of_papers_with_no_methods = extract_paper_section_headers(text_files, ids_with_no_methods)

sha_id_title_dict = {k: v for k,v in zip(annot_task_df.iloc[:, 0], annot_task_df.iloc[:, 1])}
edited_sha_id_title_dict = {k.split(';')[0] : v for k, v in sha_id_title_dict.items()}
with open('sections_of_papers_with_no_results.txt', 'w') as f:

    for paper_id, sections in sections_of_papers_with_no_results.items():
        f.write("%s\t%s\n" % (paper_id, edited_sha_id_title_dict[paper_id]))
        for section in sections:
            f.write("%s\n" % section)
        f.write('\n')

with open('sections_of_papers_with_no_methods.txt', 'w') as f:

    for paper_id, sections in sections_of_papers_with_no_methods.items():
        f.write("%s\t%s\n" % (paper_id, edited_sha_id_title_dict[paper_id]))
        for section in sections:
            f.write("%s\n" % section)
        f.write('\n')
