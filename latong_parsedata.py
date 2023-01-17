import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from  collections import OrderedDict


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
meta=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
meta.head()
meta=meta[((meta['has_pdf_parse']==True) |(meta['has_pmc_xml_parse']==True))]
meta_sm=meta[['cord_uid','sha','pmcid','title','abstract','publish_time','url']]
meta_sm.drop_duplicates(subset ="title", keep = False, inplace = True)
meta_sm.loc[meta_sm.publish_time=='2020-12-31'] = "2020-03-31"
meta_sm.head()
meta_sm.shape
sys.path.insert(0, "../")

root_path = '/kaggle/input/CORD-19-research-challenge/'
#build a dataframe to store parsed data

df = {"paper_id": [], "text_body": []}
df = pd.DataFrame.from_dict(df)
df
collect_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

from tqdm import tqdm

for i,file_name in enumerate (tqdm(collect_json)):
    
    row = {"paper_id": None, "text_body": None}
  
    with open(file_name) as json_data:
            
        data = json.load(json_data,object_pairs_hook=OrderedDict)
        
        row['paper_id']=data['paper_id']
        
        body_list = []
       
        for _ in range(len(data['body_text'])):
            try:
                body_list.append(data['body_text'][_]['text'])
            except:
                pass

        body = "\n ".join(body_list)
        
        row['text_body']=body 
        df = df.append(row, ignore_index=True)
  
#merge metadata df with parsed json file based on sha_id
merge1=pd.merge(meta_sm, df, left_on='sha', right_on=['paper_id'])
merge1.head()
len(merge1)
#merge metadata set with parsed json file based on pcmid
merge2=pd.merge(meta_sm, df, left_on='pmcid', right_on=['paper_id'])
merge2.head()
len(merge2)
#combine merged sha_id and pcmid dataset, remove the duplicate values based on file name
merge_final= merge2.append(merge1, ignore_index=True)
merge_final.drop_duplicates(subset ="title", keep = False, inplace = True)
len(merge_final)
#remove articles that are not related to COVID-19 based on publish time
corona=merge_final[(merge_final['publish_time']>'2019-11-01') & (merge_final['text_body'].str.contains('nCoV|Cov|COVID|covid|SARS-CoV-2|sars-cov-2'))]
corona.shape
coro=corona.reset_index(drop=True)
coro.head()
coro.to_csv("coro.csv",index=False)
#clean txt ...