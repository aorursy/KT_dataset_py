import numpy as np

import pandas as pd

import os

import glob

import sys

sys.path.insert(0, "../")



from parse_functions import return_corona_df



# Get all the files saved into a list and then iterate over them like below to extract relevant information

# hold this information in a dataframe and then move forward from there. 
# Just set up a quick blank dataframe to hold all these medical papers. 



corona_features = {"doc_id": [None], "source": [None], "title": [None],

                  "abstract": [None], "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)
corona_df
# Cool so dataframe now set up, lets grab all the json file names. 



# For this we can use the very handy glob library



root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)



# Now we just iterate over the files and populate the data frame. 
corona_df = return_corona_df(json_filenames, corona_df, 'b')
corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')