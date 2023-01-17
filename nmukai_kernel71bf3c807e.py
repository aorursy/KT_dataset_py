import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import glob
import json

from IPython.utils import io
import os
import gc

import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import minibatch

!pip install spacy download en_core_web_sm
# read metadata
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}metadata.csv'
meta_df = pd.read_csv(metadata_path)

# examine columns
print(meta_df.shape)
print(meta_df.info())

# examine cord_uid dup
cord_uid_dup = meta_df.cord_uid.duplicated(keep = False)
print('number of papers having duplicated cord_uid: ', len(meta_df[cord_uid_dup]))

# print duplicated
if len(meta_df[cord_uid_dup]) > 0:
    print(meta_df[['cord_uid', 'sha', 'title']][cord_uid_dup])
# drop rows having duplicated cord_uid value(keep the first row among duplication)
cord_uid_dup_keep_first = meta_df.cord_uid.duplicated(keep = 'first')
meta_df = meta_df[~cord_uid_dup_keep_first]
print(meta_df.shape)
# duplicated sha, Nan excluded
has_sha = ~meta_df.sha.isnull()
sha_dup = meta_df.sha.duplicated(keep = False) & has_sha
print('number of papers having duplicated sha: ', len(meta_df[sha_dup]))

# duplicated pmcid, Nan excluded
has_pmcid = ~meta_df.pmcid.isnull()
pmcid_dup = meta_df.pmcid.duplicated(keep = False) & has_pmcid
print('number of papers having duplicated pmcid: ', len(meta_df[pmcid_dup]))

# print duplicated
if len(meta_df[sha_dup]) > 0:
    print(meta_df[['cord_uid', 'sha', 'title']][sha_dup])
    
if len(meta_df[pmcid_dup]) > 0:
    print(meta_df[['cord_uid', 'pmcid', 'title']][pmcid_dup])
# drop rows having duplicated sha value(keep the first row among duplication)
sha_dup_keep_first = meta_df.sha.duplicated(keep = 'first') & has_sha
meta_df = meta_df[~sha_dup_keep_first]
print(meta_df.shape)
# compare no of sha and no of pdf_parse(should be same)
has_pdf = meta_df.has_pdf_parse == True
print('number of papers having sha', len(meta_df[has_sha]))
print('number of papers having pdf_parse', len(meta_df[has_pdf]))

# compare no of pmcid and no of pmc_xml_parse(might be different)
has_pmc_xml = meta_df.has_pmc_xml_parse == True
print('number of papers having pmcid', len(meta_df[has_pmcid]))
print('number of papers having pmc_xml_parse', len(meta_df[has_pmc_xml]))

# number of rows having coresponding json files(to be joined)
print('number of papers having pdf_parse or pmc_xml_parse: ',
     len(meta_df[has_pdf | has_pmc_xml]))

print('number of papers not having pdf_parse but pmc_xml_parse: ',
     len(meta_df[~has_pdf & has_pmc_xml]))
# pdf_json file path
pdf_json = glob.glob(f'{root_path}/**/pdf_json/*.json', recursive=True)
print('number of pdf_json files: ' ,len(pdf_json))

# pdf_json file basename(sha)
aa = [os.path.basename(x) for x in pdf_json]
pdf_json_sha = [os.path.splitext(x)[0] for x in aa]
print(pdf_json_sha[0:5])

# pmc_json file path
pmc_json = glob.glob(f'{root_path}/**/pmc_json/*.json', recursive=True)
print('number of pmc_json files: ' ,len(pmc_json))

# pmc_json file basename(pmcid)
aa = [os.path.basename(x) for x in pmc_json]
pmc_json_pmcid = [os.path.splitext(x)[0] for x in aa]
print(pmc_json_pmcid[0:5])

# pdf_json + pmc_json
all_json = pdf_json
all_json.extend(pmc_json)
print('number of pmc_json / pdf_json files: ', len(all_json))
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.title = content['metadata']['title']
            self.abstract = []
            self.body_text = []
            self.introduction = []
            self.conclusion = []

            # Abstract
            if 'abstract' in content:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            # Introduction
            introduction_synonyms = ['introduction', 'introductions']
            for entry in content['body_text']:
                section_title = ''.join(x.lower() for x in entry['section'] if x.isalpha())
                if any(r in section_title for r in introduction_synonyms) :
                    self.introduction.append(entry['text'])
            # Conclusion
            conclusion_synonyms = ['conclusion', 'conclusions',
                                  'conclusions and perspectives', 
                                  'conclusions and future perspectives', 
                                  'conclusions and future directions',
                                  'summary']
            for entry in content['body_text']:
                section_title = ''.join(x.lower() for x in entry['section'] if x.isalpha())
                if any(r in section_title for r in conclusion_synonyms) :
                    self.conclusion.append(entry['text'])  
    
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            self.introduction = '\n'.join(self.introduction)
            self.conclusion = '\n'.join(self.conclusion)

    def __repr__(self):
        return f'{self.paper_id}:{self.abstract[:20]}{self.body_text[:20]}'
first_row = FileReader(all_json[0])
print(first_row)
paper_dict = {'paper_id': [], 'title':[], 'abstract': [], 
              'body_text': [], 'introduction': [], 'conclusion': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    paper_dict['paper_id'].append(content.paper_id)
    paper_dict['title'].append(content.title)
    paper_dict['abstract'].append(content.abstract)
    paper_dict['body_text'].append(content.body_text)
    paper_dict['introduction'].append(content.introduction)
    paper_dict['conclusion'].append(content.conclusion)
paper_df = pd.DataFrame(paper_dict)
paper_df = paper_df.replace("", np.nan)
print(paper_df.info())
# join meta_df(left) and pdf_json_df(right)
df1 = pd.merge(meta_df, paper_df, left_on = 'sha', right_on = 'paper_id', how = 'left')

# keep joined rows
df1 = df1[~df1['paper_id'].isnull()]
to_be_joined = len(meta_df[~(meta_df['sha'].isnull())])
actually_joined = len(df1)
print('number of sha(pdf_json paper_id) in metadata:   ', to_be_joined)
print('number of pdf_json files found:                 ', actually_joined)
print('number of pdf_json files not found:             ', to_be_joined - actually_joined)

print(df1.shape)
print(df1.columns)
# remove title, abstract extracted from json_file
df1.drop(columns = ['title_y', 'abstract_y'], inplace = True)

# rename title, abstract originateed from metadata
df1.rename(columns = {'title_x': 'title', 'abstract_x': 'abstract'}, inplace = True)

# add new column joined
df1['pdf_json_joined'] = True
df1['pmc_json_joined'] = False

print(df1.shape)
print(df1.columns)
# remaining meta_df rows having pmcid
rem_df = meta_df[~(meta_df['cord_uid'].isin(df1['cord_uid'])) &
                ~(meta_df['pmcid'].isnull())]

# join remaining meta_df(left) and pmc_json_df(right)
df2 = pd.merge(rem_df, 
               paper_df, left_on = 'pmcid', 
               right_on = 'paper_id', how = 'left')

# keep joined rows
df2 = df2[~(df2['paper_id'].isnull())]

to_be_joined = len(rem_df)
actually_joined = len(df2)
print('number of pmcid(pmc_json paper_id) in remaining metadata: ', to_be_joined)
print('number of pmc_json files found:                 ', actually_joined)
print('number of pmc_json files not found:             ', to_be_joined - actually_joined)

print(df2.shape)
print(df2.columns)
# remove title, abstract extracted from json_file
df2.drop(columns = ['title_y', 'abstract_y'], inplace = True)

# rename title, abstract originateed from metadata
df2.rename(columns = {'title_x': 'title', 'abstract_x': 'abstract'}, inplace = True)

# add new column joined
df2['pdf_json_joined'] = False
df2['pmc_json_joined'] = True

print(df2.shape)
print(df2.columns)
# still remeining DF
df3 = meta_df[~(meta_df.cord_uid.isin(df1.cord_uid)) &
              ~(meta_df.cord_uid.isin(df2.cord_uid))]
df3['paper_id'] = np.nan
df3['body_text'] = np.nan
df3['introduction'] = np.nan
df3['conclusion'] = np.nan
df3['pdf_json_joined'] = False
df3['pmc_json_joined'] = False

# conatenate df1, df2, df3
df4 = pd.concat([df1, df2, df3])
meta_paper_df = meta_df['cord_uid']
meta_paper_df = pd.merge(meta_paper_df, df4, on = 'cord_uid', how = 'left')

print(meta_paper_df.info())
nlp = spacy.load('en_core_web_sm')
nlp.pipeline
npi_patterns = {'LOCK_DOWN':[nlp('lock down'),
                            nlp('lockdown')],
                'SCHOOL_CLOSURE': [nlp('school closure')],
                'HOUSEHOLD_QUARANTINE':[nlp('household quarantine'),
                                         nlp('stay at home')],
                'RESTRICT_ACTIVITY':[nlp('refrain from socail activity'),
                                    nlp('restrict social activity'),
                                    nlp('shut restaurant'),
                                    nlp('shut bar')],
                'RESTRICT_TRAVEL': [nlp('travel restriction'), 
                        nlp('restrict travel'),
                        nlp('entry ban'),
                        nlp('ban entry'),
                        nlp('travel ban'),
                        nlp('ban travel')],
                'PUBLIC_COMMNICATION': [nlp('public campaign'),
                   nlp('public communication'),
                   nlp('change behavior'),
                   nlp('social distance'),
                   nlp('wash hand'),
                   nlp('face mask')]}

npi_matcher = PhraseMatcher(nlp.vocab)
for key, value in npi_patterns.items():
    print(key)
    print(value)
    npi_matcher.add(key, None, *value)
def find_npi(text, batch_size = 1000):
    ''' 
    tokenize, phrasematch and make dataframe
    argument: list of texts
    return:   dataframe 'idx',  'name', 'text' 'doc'
    '''
    batches = list(minibatch(text, size = batch_size))
    dict_x = {'idx': [], 'name':[], 'text':[], 'doc':[]}    

    for batch_no, batch in enumerate(batches):
        if batch_no % (len(batches) // 10) == 0:
            print('Processing batch_no: ', batch_no, ' of ', len(batches))
        
        docs = list(nlp.pipe(batch, disable = ['tageer', 'parser', 'ner']))
    
        for idx, doc in enumerate(docs):
            matches = npi_matcher(doc)
            for match_id, start, end in matches:
                # print(idx, doc.vocab.strings[match_id], doc[start:end].text)
                dict_x['idx'].append(batch_no * batch_size + idx)
                dict_x['name'].append(doc.vocab.strings[match_id])
                dict_x['text'].append(doc[start:end].text)
                dict_x['doc'].append(doc)

        # memory release
        del docs
        gc.collect()
    
    # dataframe
    df_x = pd.DataFrame(dict_x)
    return(df_x)
title_df = find_npi(meta_paper_df['title'].astype(str), batch_size = 1000)

# remove duplication
title_df = title_df.drop_duplicates(subset = ['idx', 'name'])

print('number of papers found:', len(set(title_df.idx)))
print('NPI categorys are: ', set(title_df.name))
abstract_df = find_npi(meta_paper_df['abstract'].astype(str), batch_size = 1000)

# remove duplication
abstract_df = abstract_df.drop_duplicates(subset = ['idx', 'name'])

print('number of papers found:', len(set(abstract_df.idx)))
print('NPI categorys are: ', set(abstract_df.name))
introduction_df = find_npi(meta_paper_df.introduction.astype(str), batch_size = 50)

# remove duplication
introduction_df = introduction_df.drop_duplicates(subset = ['idx', 'name'])

print('number of papers found:', len(set(introduction_df.idx)))
print('NPI categorys are: ', set(introduction_df.name))
conclusion_df = find_npi(meta_paper_df.conclusion.astype(str), batch_size = 1000)

# remove duplication
conclusion_df = conclusion_df.drop_duplicates(subset = ['idx', 'name'])
print('number of papers found:', len(set(conclusion_df.idx)))
print('NPI categorys are: ', set(conclusion_df.name))
# collect phaseMatching result (where NPI phrases are in the paper)
npi_df = pd.DataFrame({'cord_uid': meta_paper_df.cord_uid})
npi_df['title_matched'] = False
npi_df.title_matched[[idx for idx in title_df.idx]] = True

npi_df['abstract_matched'] = False
npi_df.abstract_matched[[idx for idx in abstract_df.idx]] = True

npi_df['introduction_matched'] = False
npi_df.introduction_matched[[idx for idx in introduction_df.idx]] = True

npi_df['conclusion_matched'] = False
npi_df.conclusion_matched[[idx for idx in conclusion_df.idx]] = True
 
# collect phaseMatching result(which NPI category phrases are in the paper)
result_df = pd.concat([title_df, abstract_df, introduction_df, conclusion_df])

for cate in [k for k in npi_patterns.keys()]:
    npi_df[cate] = False
    cate_matched_df = result_df[result_df.name == cate]
    npi_df[cate][[idx for idx in cate_matched_df.idx]] = True
# join meta_paper, npi
npi_df = pd.merge(meta_paper_df, npi_df, 
                  on = 'cord_uid', how = 'left')

# keep rows having NPI phrases
npi_df = npi_df[npi_df.title_matched | 
                     npi_df.abstract_matched |
                     npi_df.introduction_matched | 
                     npi_df.conclusion_matched]

print(npi_df.shape)
print(npi_df.columns)

# write csv file
npi_path = 'npi_data.csv'
npi_df.to_csv(npi_path)

npi_phrase_path = 'npi_phrase.csv'
result_df.to_csv(npi_phrase_path)
result_df.head()
# make count dataframe
c_dict = {'category': [], 'title_matched':[], 'abstract_matched':[],
         'introduction_matched':[], 'conclusion_matched':[]}
for cate in [cate for cate in npi_patterns.keys()]:
    c_dict['category'].append(cate)
    for mat in ['title_matched', 'abstract_matched',
                'introduction_matched', 'conclusion_matched']:
        c_dict[mat].append(len(npi_df[npi_df[cate] & npi_df[mat]]))

count_df = pd.DataFrame(c_dict)
count_df
fig = plt.figure()
count_df.plot.barh(x = 'category', 
                   y = ['title_matched', 'abstract_matched',
                       'introduction_matched', 'conclusion_matched'])
plt.title('number of papers having NPI phrases')
fig.savefig("npi_papers.jpeg")