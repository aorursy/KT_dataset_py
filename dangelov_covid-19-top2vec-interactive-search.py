!pip install top2vec==1.0.6
import numpy as np 
import pandas as pd 
import json
import os
import ipywidgets as widgets
from IPython.display import clear_output, display
from top2vec import Top2Vec
metadata_df = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")
metadata_df.head()
def preproccess_papers():

    dataset_dir = "../input/CORD-19-research-challenge/"
    comm_dir = dataset_dir+"comm_use_subset/comm_use_subset/"
    noncomm_dir = dataset_dir+"noncomm_use_subset/noncomm_use_subset/"
    custom_dir = dataset_dir+"custom_license/custom_license/"
    biorxiv_dir = dataset_dir+"biorxiv_medrxiv/biorxiv_medrxiv/"
    directories_to_process = [comm_dir,noncomm_dir, custom_dir, biorxiv_dir]

    papers_with_text = list(metadata_df[metadata_df.has_full_text==True].sha)

    paper_ids = []
    titles = []
    abstracts = []
    sections = []
    body_texts = []

    for directory in directories_to_process:

        filenames = os.listdir(directory)

        for filename in filenames:

          file = json.load(open(directory+filename, 'rb'))

          #check if file contains text
          if file["paper_id"] in papers_with_text:

            section = []
            text = []

            for bod in file["body_text"]:
              section.append(bod["section"])
              text.append(bod["text"])

            res_df = pd.DataFrame({"section":section, "text":text}).groupby("section")["text"].apply(' '.join).reset_index()

            for index, row in res_df.iterrows():

              # metadata
              paper_ids.append(file["paper_id"])

              if(len(file["abstract"])):
                abstracts.append(file["abstract"][0]["text"])
              else:
                abstracts.append("")

              titles.append(file["metadata"]["title"])

              # add section and text
              sections.append(row.section)
              body_texts.append(row.text)

    return pd.DataFrame({"id":paper_ids, "title": titles, "abstract": abstracts, "section": sections, "text": body_texts})
# papers_df = preproccess_papers()
# papers_df.head()
def filter_short(papers_df):
    papers_df["token_counts"] = papers_df["text"].str.split().map(len)
    papers_df = papers_df[papers_df.token_counts>200].reset_index(drop=True)
    papers_df.drop('token_counts', axis=1, inplace=True)
    
    return papers_df
   
# papers_df = filter_short(papers_df)
# papers_df.head()
# top2vec = Top2Vec(documents=papers_df.text, speed="learn", workers=4)
top2vec = Top2Vec.load("../input/covid19top2vec/covid19_deep_learn_top2vec")
papers_df = pd.read_feather("../input/covid19top2vec/covid19_papers_processed.feather")
keywords_select_st = widgets.Label('Enter keywords seperated by space: ')
display(keywords_select_st)

keywords_input_st = widgets.Text()
display(keywords_input_st)

keywords_neg_select_st = widgets.Label('Enter negative keywords seperated by space: ')
display(keywords_neg_select_st)

keywords_neg_input_st = widgets.Text()
display(keywords_neg_input_st)

doc_num_select_st = widgets.Label('Choose number of topics: ')
display(doc_num_select_st)

doc_num_input_st = widgets.Text(value='5')
display(doc_num_input_st)

def display_similar_topics(*args):
    
    clear_output()
    display(keywords_select_st)
    display(keywords_input_st)
    display(keywords_neg_select_st)
    display(keywords_neg_input_st)
    display(doc_num_select_st)
    display(doc_num_input_st)
    display(keyword_btn_st)
    
    try:
        topic_words, word_scores, topic_scores, topic_nums = top2vec.search_topics(keywords=keywords_input_st.value.split(),num_topics=int(doc_num_input_st.value), keywords_neg=keywords_neg_input_st.value.split())
        for topic in topic_nums:
            top2vec.generate_topic_wordcloud(topic, background_color="black")
        
    except Exception as e:
        print(e)
        
keyword_btn_st = widgets.Button(description="show topics")
display(keyword_btn_st)
keyword_btn_st.on_click(display_similar_topics)
topic_num_select = widgets.Label('Select topic number: ')
display(topic_num_select)

topic_input = widgets.Text()
display(topic_input)

doc_num_select = widgets.Label('Choose number of documents: ')
display(doc_num_select)

doc_num_input = widgets.Text(value='10')
display(doc_num_input)

def display_topics(*args):
    
    clear_output()
    display(topic_num_select)
    display(topic_input)
    display(doc_num_select)
    display(doc_num_input)
    display(topic_btn)

    documents, document_scores, document_nums = top2vec.search_documents_by_topic(topic_num=int(topic_input.value), num_docs=int(doc_num_input.value))
    
    result_df = papers_df.loc[document_nums]
    result_df["document_scores"] = document_scores
    
    for index,row in result_df.iterrows():
        print(f"Document: {index}, Score: {row.document_scores}")
        print(f"Section: {row.section}")
        print(f"Title: {row.title}")
        print("-----------")
        print(row.text)
        print("-----------")
        print()

topic_btn = widgets.Button(description="show documents")
display(topic_btn)
topic_btn.on_click(display_topics)
keywords_select_kw = widgets.Label('Enter keywords seperated by space: ')
display(keywords_select_kw)

keywords_input_kw = widgets.Text()
display(keywords_input_kw)

keywords_neg_select_kw = widgets.Label('Enter negative keywords seperated by space: ')
display(keywords_neg_select_kw)

keywords_neg_input_kw = widgets.Text()
display(keywords_neg_input_kw)

doc_num_select_kw = widgets.Label('Choose number of documents: ')
display(doc_num_select_kw)

doc_num_input_kw = widgets.Text(value='10')
display(doc_num_input_kw)

def display_keywords(*args):
    
    clear_output()
    display(keywords_select_kw)
    display(keywords_input_kw)
    display(keywords_neg_select_kw)
    display(keywords_neg_input_kw)
    display(doc_num_select_kw)
    display(doc_num_input_kw)
    display(keyword_btn_kw)
    
    try:
        documents, document_scores, document_nums = top2vec.search_documents_by_keyword(keywords=keywords_input_kw.value.split(), num_docs=int(doc_num_input_kw.value), keywords_neg=keywords_neg_input_kw.value.split())
        result_df = papers_df.loc[document_nums]
        result_df["document_scores"] = document_scores

        for index,row in result_df.iterrows():
            print(f"Document: {index}, Score: {row.document_scores}")
            print(f"Section: {row.section}")
            print(f"Title: {row.title}")
            print("-----------")
            print(row.text)
            print("-----------")
            print()
           
    except Exception as e:
        print(e)
        

keyword_btn_kw = widgets.Button(description="show documents")
display(keyword_btn_kw)
keyword_btn_kw.on_click(display_keywords)
keywords_select_sw = widgets.Label('Enter keywords seperated by space: ')
display(keywords_select_sw)

keywords_input_sw = widgets.Text()
display(keywords_input_sw)

keywords_neg_select_sw = widgets.Label('Enter negative keywords seperated by space: ')
display(keywords_neg_select_sw)

keywords_neg_input_sw = widgets.Text()
display(keywords_neg_input_sw)


doc_num_select_sw = widgets.Label('Choose number of words: ')
display(doc_num_select_sw)

doc_num_input_sw = widgets.Text(value='20')
display(doc_num_input_sw)

def display_similar_words(*args):
    
    clear_output()
    display(keywords_select_sw)
    display(keywords_input_sw)
    display(keywords_neg_select_sw)
    display(keywords_neg_input_sw)
    display(doc_num_select_sw)
    display(doc_num_input_sw)
    display(sim_word_btn_sw)
    
    try:            
        words, word_scores = top2vec.similar_words(keywords=keywords_input_sw.value.split(), keywords_neg=keywords_neg_input_sw.value.split(), num_words=int(doc_num_input_sw.value))
        for word, score in zip(words, word_scores):
            print(f"{word} {score}")
   
    except Exception as e:
        print(e)
        
sim_word_btn_sw = widgets.Button(description="show similar words")
display(sim_word_btn_sw)
sim_word_btn_sw.on_click(display_similar_words)