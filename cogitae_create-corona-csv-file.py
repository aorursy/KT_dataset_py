import numpy as np

import pandas as pd

import os

import json

import glob

import sys

sys.path.insert(0, "../")



root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'
# Get all the files saved into a list and then iterate over them like below to extract relevant information

# hold this information in a dataframe and then move forward from there. 
# Just set up a quick blank dataframe to hold all these medical papers. 



corona_features = {"doc_id": [None], "source": [None], "title": [None],

                  "abstract": [None], "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)
corona_df
# Cool so dataframe now set up, lets grab all the json file names. 



# For this we can use the very handy glob library



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)



len(json_filenames)
# Now we just iterate over the files and populate the data frame. 
def return_corona_df(json_filenames, df, source):



    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            doc_id = data['paper_id']

            row['doc_id'] = doc_id

            row['title'] = data['metadata']['title']



            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [abst['text'] for abst in data['abstract']]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. 

            body_list = [bt['text'] for bt in data['body_text']]

            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Now just add to the dataframe. 

            

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

    

    

    
corona_df = return_corona_df(json_filenames, corona_df, 'b')
corona_df.shape
corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')