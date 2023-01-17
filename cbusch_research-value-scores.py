#!pip install fuzzywuzzy
#!pip install nltk
#!pip install plotly
#!pip install swifter
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm_notebook as tqdm
import re
from functools import reduce
import operator
from fuzzywuzzy import fuzz
#import swifter
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
def get_pos_tag(text,pos_type=('NNS')):
    if type(text)!=str:
        text = str(text)
    tokenized = nltk.word_tokenize(text)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] in pos_type)]
    if len(nouns):
        #print(text,nltk.pos_tag(tokenized))
        return text
    else:
        return None


def get_tags(text='',
             type_of_extractor='type_of_study', # 'type_of_study', 'animal_model' , 'cohort','is_significant'
             text_useless_on_its_own=('study'),
             stop_words=set(stopwords.words('english')),
             *args, **kwargs
            ):
    
    regex_map={
            'type_of_study':'(?:\S+\s)(?:study|trial|retrospective|prospective|case-control|ecological)',
            'animal_model': '(?:\S+\s)(?:animal)',
            'human_model':  '(?:\S+\s)(?:human|person|person|man|men|woman|women|child|children|patient|patients)',
            'lab_model':    '(?:\S+\s)(?:lab|laboratory|culture|specimen|microbiology|molecula)',
            'cohort':       '[0-9]+\s[a-zA-Z]+',
            'is_significant': '(is|not)\sstatistically significant'
             }
        
    if not regex_map.get(type_of_extractor,None):
        print("Invalid Implementation",type_of_extractor)
        return None
    
    regex_filter = regex_map[type_of_extractor]
    selected_words = re.findall(regex_filter,text.lower())
    
    if type_of_extractor in ['type_of_study','animal_model','human_model','lab_model']:
        value = pd.Series(selected_words).apply(
                                lambda x: pd.Series(x.split(" "))
                                        ).apply( 
                                lambda x:
                                    x.apply(
                                        (lambda y: y if y not in stop_words  else None )
                                        )                                
                                    ).apply(
                                        lambda x: ' '.join(x.dropna()),
                                        axis=1
                                    ).apply(
                                        lambda x: x if x not in text_useless_on_its_own else None
                                        ).dropna().tolist()
        
        
            

    if type_of_extractor in ['cohort']:
        selected_words_parsed = pd.Series(selected_words).apply(
                                lambda x: get_pos_tag(x,('NNS'))
                            ).apply(
#this is to include only specific words
                                lambda x: { x.split(" ")[1]:x.split(" ")[0]} if ( x and 
                                                                                 len(x.split(" "))==2 and 
                                                                                 x.split(" ")[1] in kwargs.get('inclusion_keyword',[])) else None
#                                 lambda x: { x.split(" ")[1]:x.split(" ")[0]} if ( x and 
#                                                                                  len(x.split(" "))==2 ) else None
            
                    ).dropna()

        if len(selected_words_parsed): 
            value = reduce(lambda a, b: dict(a, **b), selected_words_parsed)
        else:
            value = {}

    if type_of_extractor == 'is_significant':
        value = pd.Series(selected_words).value_counts().to_dict()
        
    return value
def get_keywords(list_of_files):
    
    df = pd.DataFrame()

    for txt_file in tqdm(list_of_files):
        txt_file_text = open(txt_file, "r").read()
        
        author_info=get_authors_info(txt_file)

        df = df.append({
                'paper_id': re.findall("[\w-]+\.",txt_file)[0][:-1],
                'path': txt_file,
                'study': get_tags(text=txt_file_text,
                                 type_of_extractor='type_of_study',
                                 stop_words=  set(stopwords.words('english')).union( ['present','one','recent','largest',
                                                         'first','different','previous','prior',
                                                         'this','our','current','another','pilot',
                                                         '"this','"a','"our'
                                                        ]
                                                    )
                                ),
                'subject_animal': get_tags(text=txt_file_text,
                                 type_of_extractor='animal_model',
                                 stop_words=  set(stopwords.words('english'))
                                 ),
                'subject_human': get_tags(text=txt_file_text,
                                 type_of_extractor='human_model',
                                 stop_words=  set(stopwords.words('english'))
                                 ),        
                'subject_lab': get_tags(text=txt_file_text,
                                 type_of_extractor='lab_model',
                                 stop_words=  set(stopwords.words('english'))
                                 ),   
                'cohort_info': get_tags(text=txt_file_text,
                                 type_of_extractor='cohort',
                                 stop_words=  set(),
                                 inclusion_keyword =  ('people','patient','patients','children','infant', 'females','males',
                                                         'women','men','subjects','animals','control','controls','cases',
                                                         'mice','dogs','calves','cats','samples','groups','individuals','participants',
                                                         'adult','adults','candidate','candidates')
                                ),  
                'is_significant': get_tags(text=txt_file_text,
                                 type_of_extractor='is_significant',
                                ),
                **author_info
                },ignore_index=True)

    df=df.set_index('paper_id')
    return df
study_scoring_df = pd.read_csv("../input/study-scoring/study_scoring.csv")
study_scoring_df.head()

def get_study_method_match(study_name):
    #print("study_scoring_df.study_method",study_scoring_df.study_method)
#    print("study_name",study_name)
    best_match_index = np.argmax(study_scoring_df.study_method.apply(lambda x: 
                                    fuzz.token_sort_ratio(x.lower(), study_name.lower())))
    
    return study_scoring_df.value[best_match_index]


def get_cohort_info_score(df_row):
    try:
        data= df_row['cohort_info']
        values = list(data.values())

        if len(values):
            if '000' in values:
                return 3
            try:
                values = pd.Series(values).apply(int)
            except:
                print("unable to convert values to int")

            max_value = max(values)

            if max_value <= 10:
                return 1
            elif max_value <=100:
                return 2
            elif max_value >100:
                return 3
        else:
            return 0
    except:
        return 0 

    
def get_significant_score(df_row):    
    try:
        data= df_row['is_significant']
        is_count = data.get('is',0)
        is_not_count = data.get('not',0)
    except:
        print('significant data issue',df_row['is_significant'],df_row['path'])        
        return 0     
    
    weight = (is_count-is_not_count)

    if weight>3:
        return 3
    elif weight <-3:
        return -3
    else:
        return weight


def get_study_type(df_row):
    try:
        data= df_row['study']
        if len(data)>0:
            return max(pd.Series(data).apply(lambda x: get_study_method_match(x)))
        else:
            return 0  
    except:
        print('study type issue',df_row['study'],df_row['path'])        
        return 0
    
def get_study_subject_type(df_row):

    try:
        data= df_row[['subject_animal','subject_human','subject_lab']]
        data = data.apply(lambda x: len(x))

        if data['subject_animal']:
            return 2    
        if data['subject_human'] >  data['subject_lab'] :
            return 3
        elif data['subject_lab'] > 0:
            return 1
        else:
            return 0
    except:
        return 0


def get_author_score(df_row):    
    try:
        data= df_row['Rank']
        if data < 20:
            return 3
        elif data < 100:
            return 2    
        elif data < 150:
            return 1
        else:
            return 0 
    except:
        return 0
    
def get_weights(df_row):
    weights =   {
             
                'weight_subject_type': get_study_subject_type(df_row), 
                'weight_cohort_info': get_cohort_info_score(df_row),
                'weight_significance': get_significant_score(df_row),       
                'weight_study_type':  get_study_type(df_row),
                'weight_affliation': get_author_score(df_row)
            }

    weights['cummulative_sum'] = sum(weights.values())
    
    return pd.Series(weights)
# for test
# df = get_keywords(get_filelist(dir_path,type_of_file='.json')[0:40])        
# df = df.join(df.swifter.apply(lambda x: get_weights(x),axis=1))
# df
def get_authors_info(json_filepath):
    paper_info = json.loads(open(json_filepath,"r").read())
    extracted_paper_info = {
        'paper_id' : paper_info['paper_id'],
        'metadata':  paper_info['metadata']['title'],
        'num_authors': len(paper_info['metadata']['authors']),
        'path_to_json': json_filepath,

    }

    def get_author_info(author_json,prefix_to_key=''):

        author_details={key: paper_info['metadata']['authors'][0].get(key,None) for key in ('first','middle','last')}
        affiliation_temp=paper_info['metadata']['authors'][0].get('affiliation',None)
        if affiliation_temp:
            author_details['institution'] = affiliation_temp.get('institution')
            author_details['location'] = affiliation_temp.get('location')

        first_person_keys = list(author_details.keys())
        for key in first_person_keys:
            author_details[prefix_to_key+'_'+key] = author_details.pop(key)    

        return author_details

    if len(paper_info['metadata']['authors'])>0:
        author_info= get_author_info(   paper_info['metadata']['authors'][0],
                                        prefix_to_key='first_person'
                                    )
        if author_info:
            extracted_paper_info.update(author_info)


#     if len(paper_info['metadata']['authors'])>1:
#         author_info= get_author_info(   paper_info['metadata']['authors'][-1],
#                                         prefix_to_key='last_person'
#                                     )
#         if author_info:
#             extracted_paper_info.update(author_info)        

    return extracted_paper_info
def get_filelist(dir_path,type_of_file='.json'):
    
    txt_files=[]
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            if type_of_file in filename:
                txt_files.append(f'{dirpath}/{filename}')
    print("total files:",len(txt_files))
    return txt_files
twur_df = pd.read_csv('../input/twur-top-200/twur.csv')
twur_df.head()
twur_kag_xwalk_df = pd.read_csv("../input/twur-kaggle-crosswalk/twur-kag-xwalk.csv")
twur_kag_xwalk_df.head()
# Loop over all the folders to extract keywords, and calculate repective weights
#!mkdir -p output
test = False
save_to_file = True

df_main= pd.DataFrame()

for sources in tqdm(['biorxiv_medrxiv','comm_use_subset','custom_license','noncomm_use_subset']):
    dir_path = f'../input/CORD-19-research-challenge/{sources}/{sources}/'
    output_file = f'../weights_{sources}.csv'
    
    if test:
        df = get_keywords(get_filelist(dir_path,type_of_file='.json')[0:40])        
    else:
        df = get_keywords(get_filelist(dir_path,type_of_file='.json'))
        
    if 'first_person_institution' not in df.columns:
        df['first_person_institution'] = 'nan'   

    # join to twur without crosswalk
    #df = pd.merge(
    #    df.reset_index(),
    #    twur_kag_df[['Rank','TNAME']],
    #    left_on='first_person_institution',
    #    right_on='TNAME',
    #    how='left' 
    #).set_index('paper_id')  

    # join to twur using xwalk
    df = pd.merge((
        pd.merge(df.reset_index(), twur_kag_xwalk_df, how='left', left_on='first_person_institution', right_on='kaggle_name')),
                  twur_df, how='left', left_on='twur_name', right_on='Name').set_index('paper_id')
    
    # using swifter
    #df = df.join(df.swifter.apply(lambda x: get_weights(x),axis=1))
    
    # without swifter
    df = df.join(df.apply(lambda x: get_weights(x),axis=1))

    
    display(df.shape)
    display(df.head())
    
    df_main = df_main.append(df.reset_index(),ignore_index=True)
    
#     if test:
#         break

    if save_to_file:
        df.to_csv(output_file)

df_main = df_main.set_index('paper_id')
df_main
df_main.shape
df_main.head()
df_main.to_csv("main_output.csv")
plt.hist(df_main.cummulative_sum)
plt.show()
for weight_col in list(filter(lambda x: x if 'weight_' in x else None, df_main.columns )):
    print(weight_col)
    plt.hist(df_main[weight_col])
    plt.show()
    print("---"*10)
value_count_dict = {}
for weight_col in list(filter(lambda x: x if 'weight_' in x else None, df_main.columns )):
    value_count_dict[weight_col] = df_main[weight_col].value_counts().to_dict()
print(value_count_dict)

import plotly.graph_objects as go
import plotly.express as px
mapping = {
   'weight_study_type'  :   ['Unknown','Good','Better','Best'],
   'weight_subject_type':   ['Unknown','Lab','Animal','Human'],
   'weight_cohort_info'  :   ['Unknown','Good','Better','Best'],
   'weight_affliation':     ['200+ or unknown','100-200','21-100','Top 20'],    
   'weight_significance':   ['Unknown','Weak','Good','Strong'] 
} 
for weight_type in mapping.keys(): 
    keys = list(value_count_dict[weight_type].keys())
    y = list(value_count_dict[weight_type].values())
    sorted_key_index = np.argsort(list(value_count_dict[weight_type].keys()))
    x = [mapping[weight_type][i] for i in keys]
    if weight_type=='weight_significance':
        x = mapping[weight_type]
        w = value_count_dict[weight_type]
        y = [ 
             w.get(0,0),
             w.get(-3,0)+w.get(-2,0)+w.get(-1,0)  ,
             w.get(1,0)+w.get(2,0),
             w.get(3,0)
            ]
        fig = px.bar(x=x, 
                     y=y, 
                     labels={'x':weight_type, 'y':'count'})
    elif weight_type=='weight_affliation' :
        fig = px.bar(x=np.array(x)[sorted_key_index.tolist()].tolist()[1:], 
             y=np.array(y)[sorted_key_index.tolist()].tolist()[1:], 
             labels={'x':weight_type, 'y':'count'})
    else:
        fig = px.bar(x=np.array(x)[sorted_key_index.tolist()].tolist(), 
                     y=np.array(y)[sorted_key_index.tolist()].tolist(), 
                     labels={'x':weight_type, 'y':'count'})
    fig.show()

