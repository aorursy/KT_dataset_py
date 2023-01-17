import numpy as np

import pandas as pd



import os

import json
count = 0

file_exts = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        count += 1

        file_ext = filename.split(".")[-1]

        file_exts.append(file_ext)



file_ext_set = set(file_exts)



print(f"Files: {count}")

print(f"Files extensions: {file_ext_set}\n\n=====================\nFiles extension count:\n=====================")

file_ext_list = list(file_ext_set)

for fe in file_ext_list:

    fe_count = file_exts.count(fe)

    print(f"{fe}: {fe_count}")
count = 0

for root, folders, filenames in os.walk('/kaggle/input'):

    print(root, folders)
json_folder_path = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json"

json_file_name = os.listdir(json_folder_path)[0]

print(json_file_name)

json_path = os.path.join(json_folder_path, json_file_name)



with open(json_path) as json_file:

    json_data = json.load(json_file)
json_data_df = pd.io.json.json_normalize(json_data)
json_data_df
print(f"Files in folder: {len(os.listdir(json_folder_path))}")
from tqdm import tqdm



# to process all files, uncomment the next line and comment the line below

# list_of_files = list(os.listdir(json_folder_path))

list_of_files = list(os.listdir(json_folder_path))[0:500]

pmc_custom_license_df = pd.DataFrame()



for file in tqdm(list_of_files):

    json_path = os.path.join(json_folder_path, file)

    with open(json_path) as json_file:

        json_data = json.load(json_file)

    json_data_df = pd.io.json.json_normalize(json_data)

    pmc_custom_license_df = pmc_custom_license_df.append(json_data_df)
pmc_custom_license_df.head()
pmc_custom_license_df['abstract_text'] = pmc_custom_license_df['abstract'].apply(lambda x: x[0]['text'] if x else "")
pd.set_option('display.max_colwidth', 500)

pmc_custom_license_df[['abstract', 'abstract_text']].head()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=14)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(pmc_custom_license_df['abstract_text'], title = 'Comm use subset - papers abstract - frequent words  (500 samples)')
show_wordcloud(pmc_custom_license_df['bib_entries.BIBREF0.title'], title = 'Comm use subset - papers title - frequent words (500 samples)')

pmc_custom_license_df.loc[((pmc_custom_license_df['bib_entries.BIBREF0.venue']=="") | ((pmc_custom_license_df['bib_entries.BIBREF0.venue'].isna()))), 'bib_entries.BIBREF0.venue'] = "Not identified"

import seaborn as sns

def plot_count(feature, title, df, size=1, show_percents=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[0:20], palette='Set3')

    g.set_title("Number of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=10)

    if(show_percents):

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(100*height/total),

                    ha="center") 

    ax.set_xticklabels(ax.get_xticklabels());

    plt.show()    

plot_count('bib_entries.BIBREF0.venue', 'Comm use subset - Top 20 Journals (500 samples)', pmc_custom_license_df, 3.5)