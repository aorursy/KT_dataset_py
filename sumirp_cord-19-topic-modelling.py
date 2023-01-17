# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import json

import glob

import sys

sys.path.insert(0, "../")



root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'



# Any results you write to the current directory are saved as output.

# Foe creating csv file I referred to Create Corona .csv File by Frank Mitchell
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
# Now we just iterate over the files and populate the data frame. 
def return_corona_df(json_filenames, df, source):



    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']



            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. For some reason I am getting an index error

            # In one of the Json files, so rather than have it wrapped in a lovely list

            # comprehension I've had to use a for loop like a neanderthal. 

            

            # Needless to say this bug will be revisited and conquered. 

            

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                except:

                    pass



            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Now just add to the dataframe. 

            

            if source == 'b':

                row['source'] = "biorxiv"

            elif source == "c":

                row['source'] = "common_use_sub"

            elif source == "n":

                row['source'] = "non_common_use"

            elif source == "p":

                row['source'] = "pmc_custom_license"

            

            df = df.append(row, ignore_index=True)

    

    return df
corona_df = return_corona_df(json_filenames, corona_df, 'b')
corona_df.shape
corona_df.head()