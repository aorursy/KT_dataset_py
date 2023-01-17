from argparse import Namespace
import pandas as pd
import os
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import seaborn as sns
import numpy as np
sns.set_style("darkgrid")
from tqdm import notebook
from typing import Callable
import pickle
import shutil
args = Namespace(
    data_dir = '../input/pan-data/fake_news_spreader/',
    en_data_dir = '../input/pan-data/fake_news_spreader/en',
    es_data_dir = '../input/pan-data/fake_news_spreader/es',
)


def read_xml(xml_loc, lang):
    """reads an author's xml data
    
    Args:
        xml_loc : location of xml_file
        lang : either 'en' or 'es', for sanity check
    """
    tree = ET.parse(xml_loc)
    root = tree.getroot()
    
    assert root.attrib['lang'] == lang
    for c in root:
        for child in c: 
            yield(child.text)

def create_author_to_document_df(
    data_paths,
    lang
):
    """Creats a dict where author is mapped to his documents
    Args:
        data_paths Tuple[file_name, file_loc]
        lang: one of 'en' or 'es' for sanity check
    Returns:
        data_df: pd.DataFrame containing the author, his comments
    """
    author_to_doc = dict()
    
    author_list = []
    doc_list = []
    for file_name,file_loc in data_paths:
        author = file_name.split('.')[0]
        docs = read_xml(file_loc, lang)
        
        author_list.append(author)
        doc_list.append( list(docs) )
        
    data_df = pd.DataFrame(
        {
            'author': author_list,
            'doc_list': doc_list,
        }
    )
    return data_df

def read_truth_file(loc: str):
    """reads truth file
    Args:
        loc: full path to truth file
    Returns:
        truth_df : pd.DataFrame containing author,truth columns
    """
    
    with open(loc,'r') as file:
        lines = [line.strip().split(':::') for line in file]
    author_list = [i[0] for i in lines]
    truth = [ int(i[1]) for i in lines]
    truth_df = pd.DataFrame(
        {
            'author':author_list,
            'truth': truth,
        }
    )
    return truth_df

def make_fake_news_data_df(xml_data_dir, lang):
    """creates a pd.DataFrame of author, his comments, truth label"""
    data_paths = [
        (file,os.path.join(xml_data_dir,file)) 
        for file in os.listdir(xml_data_dir) if file[-1]!='t'
    ]
    truth_path = [
        os.path.join(xml_data_dir,file) 
        for file in os.listdir(xml_data_dir) if file[-1]=='t'
    ][0]
    
    author_documents_df = create_author_to_document_df(
        data_paths, lang,
    )

    truth_df = read_truth_file(truth_path)

    data_df = pd.merge(author_documents_df, truth_df, on = 'author')
    
    return data_df, author_documents_df, truth_df
en_data_df ,_ , _ = make_fake_news_data_df(args.en_data_dir, 'en' )
# en_data_df.rename(
#     columns={"truth":"label"},
#     inplace=True,
# )
en_data_df['label'] = en_data_df['truth'].map({
    0:'Harmless', 1:'Fake News Spreader'
})
en_data_df
# notice that it contains duplicates
en_data_df[en_data_df.truth==0].iloc[0].doc_list
#contains duplicates
len(set(en_data_df[en_data_df.truth==0].iloc[0].doc_list)) != len(en_data_df[en_data_df.truth==0].iloc[0].doc_list)
en_data_df[en_data_df.truth==1].iloc[0].doc_list
ax = sns.countplot(x="label", data=en_data_df,palette="plasma")
ax.set_title('English Data Countplot', fontsize=20)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
plt.show()
es_data_df ,_ , _ = make_fake_news_data_df(args.es_data_dir, 'es' )
es_data_df['label'] = es_data_df['truth'].map({
    0:'Harmless', 1:'Fake News Spreader'
})
es_data_df
es_data_df[es_data_df.truth==0].iloc[0].doc_list
es_data_df[es_data_df.truth==1].iloc[0].doc_list
ax = sns.countplot(x="label", data=es_data_df,palette="plasma")
ax.set_title('English Data Countplot', fontsize=20)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
plt.show()
_en_df = en_data_df.copy()
_en_df['language']='English'

_es_df = es_data_df.copy()
_es_df['language']='Spanish'


combined_df = pd.concat([_en_df,_es_df ])
with sns.plotting_context('paper',font_scale=1.7):
    fig, ax = plt.subplots(figsize=(7,4))

    sns.countplot(ax=ax, x="label", data=combined_df, hue='language',palette="plasma")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
    plt.show()

    fig.savefig("data_countplot.pdf", bbox_inches="tight") 
_temp = combined_df.copy()
_temp['num_docs'] = _temp.doc_list.map(len)
# All authors have same number of documents
len(set(_temp['num_docs'])) == 1
_temp['num_unique_docs'] = _temp.doc_list.map( lambda x: len(set(x)) )
_temp[['author','num_unique_docs','num_docs']]
_temp['num_unique_docs'].describe()
_temp[_temp.language == 'English']['num_unique_docs'].describe()
_temp[_temp.language == 'Spanish']['num_unique_docs'].describe()
_temp['tweet_len'] = _temp.doc_list.map( lambda x: [ len(tweet.strip().split()) for tweet in x ] )    
_temp['max_tweet_len'] = _temp.tweet_len.map(max)
_temp['min_tweet_len'] = _temp.tweet_len.map(min)
_temp['max_tweet_len'].describe()
_temp[_temp.language == 'English' ]['max_tweet_len'].describe()
_temp[_temp.language == 'Spanish' ]['max_tweet_len'].describe()
_temp['min_tweet_len'].describe()
_temp[_temp.language == 'English' ]['min_tweet_len'].describe()
_temp[_temp.language == 'Spanish' ]['min_tweet_len'].describe()
_temp[_temp['max_tweet_len'] == 86]['doc_list'].values[0]
_temp[_temp['min_tweet_len'] == 1]['doc_list'].values[0]