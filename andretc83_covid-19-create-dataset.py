import numpy as np

import pandas as pd

import os

import json

import glob

import sys
json_filenames = glob.glob(f'/kaggle/input/**/*.json', recursive=True)
def return_corona_df(json_filenames, source):

    features = {"doc_id": [None], "source": [None], "title": [None], "authors": [None],

                  "abstract": [None], "text_body": [None], "bibliography": [None]}

    df = pd.DataFrame.from_dict(features)

    

    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None, "authors": [None],

              "abstract": None, "text_body": None, "bibliography": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']

            

            authors = ", ".join([author['first'] + " " + author['last'] \

                                 for author in data['metadata']['authors'] if data['metadata']['authors']])

            row['authors'] = authors



            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract 

            

            body_list = [d['text'] for d in data['body_text']]

            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            bibliography = "\n ".join([bib['title'] + "," + bib['venue'] + "," + str(bib['year']) \

                                      for bib in data['bib_entries'].values()])

            row['bibliography'] = bibliography

            

            if source == 'b':

                row['source'] = "BIORXIV"

            elif source == "c":

                row['source'] = "COMMON_USE_SUB"

            elif source == "n":

                row['source'] = "NON_COMMON_USE"

            elif source == "p":

                row['source'] = "PMC_CUSTOM_LICENSE"

            

            df = df.append(row, ignore_index=True)

    

    return df
corona_df = return_corona_df(json_filenames, 'b')
corona_df.shape
corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')