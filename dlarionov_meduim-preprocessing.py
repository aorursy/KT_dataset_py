import numpy as np
import pandas as pd
# pd.set_option('display.max_colwidth', -1)

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
from langdetect import detect

def get_lang(row):
    try:
        return detect(row)
    except:
        return None
from bs4 import BeautifulSoup

def parse_html(html):
    soup = BeautifulSoup(html, 'lxml')       
    
    article = soup.find('div', class_='postArticle-content')    
    content = article.getText(separator=' ')
    imgs = len(article.select('img'))
    hrefs = len(article.select('a'))    
    
    ul = soup.find('ul', class_='tags')
    tags = []
    if (ul):
        tags = [a.text.replace(' ','') for a in ul.find_all('a')]    
    
    return content, imgs, hrefs, tags
def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result
def process_file(path_to_in_file):    
    with open(path_to_in_file, encoding='utf-8') as in_file:
        rows = []
        for line in in_file:
            json_data = read_json_line(line)
            
            html = json_data['content'].replace('\n', ' ').replace('\r', ' ')            
            content, img_cnt, href_cnt, tags = parse_html(html)
            
            rows.append([
                pd.to_datetime(json_data['published']['$date']),
                json_data['title'],
                json_data['url'].split('//')[-1].split('/')[0],
                json_data['meta_tags']['description'],                
                json_data['meta_tags']['author'],
                json_data['meta_tags']['twitter:data1'].split(' ')[0],                          
                tags,
                content,
                img_cnt,
                href_cnt
            ])
            
            # if (len(rows)==1000): break # dev mode
        
        columns = [
            'ts',
            'title',
            'domain',
            'description',             
            'author', 
            'read_time', 
            'tags',
            'content',
            'img_cnt',
            'href_cnt'
        ]
        df = pd.DataFrame(rows, columns=columns)
        
        #df['title'] = df['title'].map(lambda x: x.split('–')[0])
        
        df['read_time'] = df['read_time'].astype(np.int8)        
        df['content_len'] = df['content'].map(len).astype(np.int32)
        df['title_len'] = df['title'].map(len).astype(np.int32)
        df['desc_len'] = df['description'].map(len).astype(np.int32)        
        
        df = process_ts(df)
        df = process_lang(df)
        df = process_tags(df)
        
        return df

def process_ts(df):
    df['mm'] = df['ts'].apply(lambda ts: ts.month).astype(np.int8)
    df['yyyy'] = df['ts'].apply(lambda ts: ts.year).astype(np.int16)
    df['yyyymm'] = df['ts'].apply(lambda ts: 100 * ts.year + ts.month).astype(np.int32)
    df['hour'] = df['ts'].apply(lambda ts: ts.hour).astype(np.int8)
    df['dayofweek'] = df['ts'].apply(lambda ts: ts.dayofweek).astype(np.int8)
    df['weekend'] = df['ts'].apply(lambda ts: ts.dayofweek > 5).astype(np.int8)
    df['morning'] = df['ts'].apply(lambda ts: (ts.hour >= 7) & (ts.hour < 12)).astype(np.int8)
    df['day'] = df['ts'].apply(lambda ts: (ts.hour >= 12) & (ts.hour < 18)).astype(np.int8)
    df['evening'] = df['ts'].apply(lambda ts: (ts.hour >= 18) & (ts.hour < 23)).astype(np.int8)
    df['night'] = df['ts'].apply(lambda ts: (ts.hour >= 23) | (ts.hour < 7)).astype(np.int8) # or!
    return df

def process_lang(df):
    df['lang'] = df['description'].map(get_lang)
    #df.loc[~df.lang.isin(['en','pt','fr','es','de','it']), 'lang'] = 'rare'
    return df

def process_tags(df):
    df['tags_cnt'] = df['tags'].map(len).astype(np.int8)
    df['tags_str'] = df['tags'].map(' '.join)
    return df
%%time
train = process_file(path_to_in_file='../input/train.json')
test = process_file(path_to_in_file='../input/test.json')

train['target'] = pd.read_csv('../input/train_log1p_recommends.csv')['log_recommends']
test['id'] = pd.read_csv('../input/sample_submission.csv')['id']

train.to_pickle('train.data')
test.to_pickle('test.data') 
train = pd.read_pickle('train.data')
test = pd.read_pickle('test.data')
train.tail().T
test.head().T