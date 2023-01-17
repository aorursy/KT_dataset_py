# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        1==1

        #print(os.path.join(dirname, filename))

import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

# Any results you write to the current directory are saved as output.
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'

filenames = os.listdir(biorxiv_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []

all_files_loc=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        

        if filename[-4:]=='json':



            filename1=os.path.join(dirname, filename)

            #file = json.load(open(filename1, 'rb'))

            #all_files.append(file)

            all_files_loc.append(filename1)

# for filename in filenames:

#     filename = biorxiv_dir + filename

#     file = json.load(open(filename, 'rb'))

#     all_files.append(file)

#     all_files_loc.append(filename)
len(all_files_loc)
print(all_files[:3])

print(all_files_loc[:3])
def file_author_extract(file):

    if len(file['metadata']['authors'])>0:

        authors_detail=pd.concat((pd.DataFrame(file['metadata']['authors']),pd.DataFrame(pd.DataFrame((file['metadata']['authors']))['affiliation'].values.tolist())),axis=1)

        authors_detail.rename(columns = {'location':'ref_location'},inplace = True)

        return authors_detail

    else :

        return pd.DataFrame()

def get_author_text(file,filename):



    authors_detail=file_author_extract(file)

    

    authors_detail['title']=file['metadata']['title']

    authors_detail['location']=filename

    authors_detail['paper_id']=file['paper_id']

    return authors_detail

def get_ref_text(file,filename):



    ref_details=pd.DataFrame(file['bib_entries']).T.reset_index()

    ref_details.rename(columns = {'index':'ref'},inplace = True)

    all_ref=pd.DataFrame(file['body_text'])['cite_spans'].values.tolist()

    extract_ref=[k for k in [k['ref_id'] for  j in all_ref  for k in j if len(j)>0 ] if k != None]

    base_ref=pd.DataFrame({'ref':extract_ref})

    base_ref['ref_cnt']=1

    base_ref=base_ref.groupby('ref').count().reset_index()

    ref_details_merge=ref_details.merge(base_ref,on='ref',how='left')

    ref_details_merge['ref_cnt']=ref_details_merge['ref_cnt'].fillna(1)

    ref_details_merge.rename(columns = {'title':'ref_title'},inplace = True)

    ref_details_merge['title']=file['metadata']['title']

    ref_details_merge['location']=filename

    ref_details_merge['paper_id']=file['paper_id']

    return ref_details_merge

def get_fig_ref_text(file,filename):



    figure_ref_all=pd.DataFrame(file['ref_entries']).T

    figure_ref_all['title']=file['metadata']['title']

    figure_ref_all['location']=filename

    figure_ref_all['paper_id']=file['paper_id']

    return figure_ref_all

def file_abstract_extract(file):

    if len(file['abstract'])>0:

        return [file['abstract'][0]['text']]

    else :

        return ['None']

def get_text(file,filename):

    all_text=pd.DataFrame(file['body_text'])['text'].drop_duplicates().values.tolist()

    text_article = ' '.join([str(elem) for elem in all_text])

    all_text=pd.DataFrame({'abstract':file_abstract_extract(file),

    'body_text':[text_article]})

    all_text['title']=file['metadata']['title']

    all_text['location']=filename

    all_text['paper_id']=file['paper_id']

    return all_text



# get_author_text_list=[]

# get_ref_text_list=[]

# get_fig_ref_text_list=[]

# get_text_list=[]



# for k in range(len(all_files_loc)):

#     file  = json.load(open(all_files_loc[k], 'rb'))

#     get_author_text_list.append(get_author_text(file))

#     get_ref_text_list.append(get_ref_text(file))

#     get_fig_ref_text_list.append(get_fig_ref_text(file))

#     get_text_list.append(get_text(file))
def convert_to_onedf(list_append,rec=None):

    df=pd.DataFrame()

    j=-1

    jk=0

    pandas_list=[]

    print('length of list is '+ str(len(list_append)))

    for k in list_append:

        j=j+1

        if j%800 == 0:

            pandas_list.append(pd.DataFrame())

            jk=jk+1

            print(jk)

        if len(k)>0 :

            

            pandas_list[jk-1]=pandas_list[jk-1].append(k, sort=True) 

    if len(pandas_list)>1:

        return convert_to_onedf(pandas_list,len(pandas_list))

    return pandas_list[0]
from concurrent.futures import ThreadPoolExecutor

import time

def process_on_set(videos, num_workers,function):

    def process_file(i):

        filename_2 = videos[i]

        file  = json.load(open(filename_2, 'rb'))

        y_pred = function(file,filename_2)

        return y_pred



    with ThreadPoolExecutor(max_workers=num_workers) as ex:

        predictions = ex.map(process_file, range(len(videos)))



    return list(predictions)




# for k in range(len(all_files_loc)):

#     file  = json.load(open(all_files_loc[k], 'rb'))

#     get_author_text_list.append(get_author_text(file))

#     get_ref_text_list.append(get_ref_text(file))

#     get_fig_ref_text_list.append(get_fig_ref_text(file))

#     get_text_list.append(get_text(file))

    

start_time = time.time()

get_author_text_list = process_on_set(all_files_loc, num_workers=4,function=get_author_text)

elapsed = time.time() - start_time

print("Elapsed %f sec. Average per doc: %f sec." % (elapsed, elapsed / len(all_files_loc)))                                           



start_time = time.time()

get_author_text_df=convert_to_onedf(get_author_text_list).reset_index(drop=True)

get_author_text_df.to_pickle('get_author_text_df.pkl')

elapsed = time.time() - start_time

print("Elapsed %f sec. file %s " % (elapsed, 'get_author_text_df'))    



start_time = time.time()

get_ref_text_list = process_on_set(all_files_loc, num_workers=4,function=get_ref_text)

elapsed = time.time() - start_time



print("Elapsed %f sec. Average per doc: %f sec." % (elapsed, elapsed / len(all_files_loc)))      



start_time = time.time()

get_ref_text_df=convert_to_onedf(get_ref_text_list).reset_index(drop=True)

get_ref_text_df.to_pickle('get_ref_text_df.pkl')

elapsed = time.time() - start_time

print("Elapsed %f sec. file %s " % (elapsed, 'get_ref_text_df'))  





start_time = time.time()

get_fig_ref_text_list = process_on_set(all_files_loc, num_workers=4,function=get_fig_ref_text)

elapsed = time.time() - start_time

print("Elapsed %f sec. Average per doc: %f sec." % (elapsed, elapsed / len(all_files_loc)))      



start_time = time.time()

get_fig_ref_text_df=convert_to_onedf(get_fig_ref_text_list).reset_index(drop=True)

get_fig_ref_text_df.to_pickle('get_fig_ref_text_df.pkl')

elapsed = time.time() - start_time

print("Elapsed %f sec. file %s " % (elapsed, 'get_fig_ref_text_df'))  





start_time = time.time()

get_text_list = process_on_set(all_files_loc, num_workers=4,function=get_text)

elapsed = time.time() - start_time

print("Elapsed %f sec. Average per doc: %f sec." % (elapsed, elapsed / len(all_files_loc)))      



start_time = time.time()

get_text_df=convert_to_onedf(get_text_list).reset_index(drop=True)

get_text_df.to_pickle('get_text_df.pkl')

elapsed = time.time() - start_time

print("Elapsed %f sec. file %s " % (elapsed, 'get_text_df'))  

# get_author_text_df=convert_to_onedf(get_author_text_list)

# get_author_text_df.to_pickle('/kaggle/working/get_author_text_df.pkl')

# get_ref_text_df=convert_to_onedf(get_ref_text_list)

# get_ref_text_df.to_pickle('/kaggle/working/get_ref_text_df.pkl')

# get_fig_ref_text_df=convert_to_onedf(get_fig_ref_text_list)

# get_fig_ref_text_df.to_pickle('/kaggle/working/get_fig_ref_text_df.pkl')

# get_text_df=convert_to_onedf(get_text_list)

# get_text_df.to_pickle('/kaggle/working/get_text_df.pkl')


# start_time = time.time()

# get_text_list = process_on_set(all_files_loc[:22], num_workers=4,function=get_text)

# elapsed = time.time() - start_time

# print("Elapsed %f sec. Average per doc: %f sec." % (elapsed, elapsed / len(all_files_loc)))      



# start_time = time.time()

# get_text_df=convert_to_onedf(get_text_list).reset_index(drop=True)

# get_text_df.to_pickle('get_text_df.pkl')

# elapsed = time.time() - start_time

# print("Elapsed %f sec. file %s " % (elapsed, 'get_text_df'))  

int_lt=list(set(get_ref_text_df['title'].str.lower().unique().tolist()) & set(get_ref_text_df['ref_title'].str.lower().unique().tolist()))
len(int_lt)
get_ref_map=get_ref_text_df.groupby(['title','ref_title']).sum()['ref_cnt'].reset_index()
get_ref_map.dropna().sort_values('ref_cnt',ascending=False)