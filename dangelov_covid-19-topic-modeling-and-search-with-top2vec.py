!pip install top2vec==1.0.6
import numpy as np 

import pandas as pd 

import json

import os

from top2vec import Top2Vec
metadata_df = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

metadata_df.head()
dataset_dir = "../input/CORD-19-research-challenge/"

comm_dir = dataset_dir+"comm_use_subset/comm_use_subset/pdf_json/"

noncomm_dir = dataset_dir+"noncomm_use_subset/noncomm_use_subset/pdf_json/"

custom_dir = dataset_dir+"custom_license/custom_license/pdf_json/"

biorxiv_dir = dataset_dir+"biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/"

directories_to_process = [comm_dir,noncomm_dir, custom_dir, biorxiv_dir]



papers_with_text = list(metadata_df[metadata_df.has_pdf_parse==True].sha)



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

            

papers_df = pd.DataFrame({"id":paper_ids, "title": titles, "abstract": abstracts, "section": sections, "text": body_texts})
papers_df.head()
papers_df["token_counts"] = papers_df["text"].str.split().map(len)

papers_df = papers_df[papers_df.token_counts>200].reset_index(drop=True)

papers_df.drop('token_counts', axis=1, inplace=True)

papers_df.head()
top2vec = Top2Vec.load("../input/covid19top2vec/covid19_deep_learn_top2vec")
papers_df = pd.read_feather("../input/covid19top2vec/covid19_papers_processed.feather")
top2vec.get_num_topics()
topic_words, word_scores, topic_nums = top2vec.get_topics(399)
for topic in topic_nums[180:190]:

    top2vec.generate_topic_wordcloud(topic, background_color="black")
topic_words, word_scores, topic_scores, topic_nums = top2vec.search_topics(keywords=["covid", "infect"],num_topics=10)

for topic in topic_nums:

    top2vec.generate_topic_wordcloud(topic, background_color="black")
documents, document_scores, document_nums = top2vec.search_documents_by_topic(topic_num=344, num_docs=2)

    

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
documents, document_scores, document_nums = top2vec.search_documents_by_keyword(keywords=["covid", "model"], num_docs=2)

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

words, word_scores = top2vec.similar_words(keywords=["chloroquine"], num_words=20)

for word, score in zip(words, word_scores):

    print(f"{word} {score}")