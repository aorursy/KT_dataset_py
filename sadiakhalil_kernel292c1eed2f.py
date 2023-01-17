# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, json

file_type = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        # separate the files by their extensions

        file_type.append(filename.split(".")[-1])



# Any results you write to the current directory are saved as output.

file_type_set = set(file_type)

print('==input dataset == \n', file_type_set)



file_type_list = list(file_type_set)

for f in file_type_list:

    f_count = file_type.count(f)

    print(f'{f}: {f_count}')
for root, folders, filenames in os.walk('/kaggle/input'):

    print('folders: ', folders)
f_readme = open('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.readme', 'r') 

print(f_readme.read())

        
metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")

pd.set_option("display.max_rows", 5)

metadata.head()
json_folder = "/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license"

json_file = os.listdir(json_folder)[0]

print (json_file)

json_file_path = os.path.join(json_folder, json_file)

print (json_file_path)

with open(json_file_path) as file:

    json_data = json.load(file)
json_data_df = pd.io.json.json_normalize(json_data) 

json_data_df
print(f'Total files in json folder: {len(os.listdir(json_folder))}')
# lets add a tool to monitor the processing progress

from tqdm import tqdm



# Lets look at all the 

json_files = list(os.listdir(json_folder))[0:100] # will explore all 1426 files later

pmc_custom_license_df = pd.DataFrame()



for j_file in tqdm(json_files):

    json_files_path = os.path.join(json_folder, j_file)

    #print (json_files_path)

    with open(json_files_path) as json_file:

        json_data = json.load(json_file)

    json_data_df = pd.io.json.json_normalize(json_data)

    pmc_custom_license_df = pmc_custom_license_df.append(json_data_df)    
#pmc_custom_license_df.reset_index(drop=True)

#pmc_custom_license_df.set_index("bib_entries.BIBREF0.ref_id", inplace = True) 

pmc_custom_license_df.head()
pmc_custom_license_df['abstract_text'] = pmc_custom_license_df['abstract'].apply(lambda x: x[0]['text'] if x else "")

pd.set_option('display.max_colwidth', 500)

pmc_custom_license_df[['abstract', 'abstract_text']].head()