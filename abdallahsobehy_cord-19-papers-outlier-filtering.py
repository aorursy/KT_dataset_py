# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os  

import json

import seaborn as sb



# This import helps visualizing in a more appealing way e.g. data_frame.head()

import glob

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Regular Expressions library

import re

# Import natural language processing library

import nltk

from nltk.corpus import stopwords

# nltk.download("stopwords")

from nltk.stem.porter import PorterStemmer



ps_obj = PorterStemmer()



root_path = "/kaggle/input/CORD-19-research-challenge/"

##

# Stems words of input string, removes non-alphabetical characters

# \param str_ input string to be normalized

# \return normalized string

#

def normalize_str(str_):

    # Keep only alphabets and remove any other numbers or symbols replacing them with spaces

    str_ = re.sub('[^a-zA-Z]', ' ', str_)

    # all to lower case

    str_ = str_lower()

    # split in list format to remove stop words

    str_ = str_.split()

    # Remove stop words

    str_ = [ps_ob(word) for word in str_ if word not in set(stopwords.words('english'))]

    # Revert to strig format

    str_ = ' '.join(str_)

    return str_
# Get all Json files

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
meta_file = root_path + "metadata.csv"

# Read CSV file as pandas dataframe, dtype to specify the data type for the specified keys

meta_df = pd.read_csv(meta_file, dtype={

'pubmed_id': str,

'Microsoft Academic Paper ID': str, 

'doi': str, 'arxiv_id': str

    })

# This function returns the first n (n=5 by default) rows for the object based on position. 

meta_df.head()
meta_df.info()
##

# Class to store meta data about papers and does some exploratory operations

#

class PaperMeta:

    def __init__(self, dir_path):

        meta_file = dir_path + "metadata.csv"

        # Read CSV file as pandas dataframe, dtype to specify the data type for the specified keys

        self.df = pd.read_csv(meta_file, dtype={

        'pubmed_id': str,

        'Microsoft Academic Paper ID': str, 

        'doi': str, 'arxiv_id': str

    })



    ##

    # Removes duplicate of the values in the input column

    # \col the column to inspect for duplicates

    # \str_normalize Normalizes string by stemming, removing stop words before checking for duplicates

    # \show_duplicates print url of duplicates for inspection

    # \update_df If true the duplictes are eleminated from the dataframe keeping only the first duplicate

    # \show_duplicates prints repeated values and urls of the repititions for inspection

    # \return Dictionary with keys as col, and values are the corresponding rows

    #

    def remove_duplicates(self, col, show_duplicates=False, update_df=False, str_normalize=False):

        # Cleaning data by replacing meaningless values to be None, this is important when deciding duplicates or outliers

        # In case of abstract some abstracts are string = 'none' or 'Unknown' or empty strings thus replace them by None

        clean_df = self.replace_meaningless_values(col)

        duplicates = clean_df[clean_df.duplicated(subset=col, keep=False) & ~clean_df[col].isnull()]

        duplicate_less_df = self.df[~clean_df.duplicated(subset=col, keep='first') | clean_df[col].isnull()]

        print(f'{col} Length of duplicates: {len(duplicates)} - possible removals keeping first: {len(self.df) - len(duplicate_less_df)}')

        if show_duplicates:

            # Sorting duplicated to show corresponding duplicates in order

            duplicates = duplicates.sort_values(by=col, axis='index')

            # Keeping track of previous index for printing lines between different column values

            prev_index = None

            for index, row in duplicates.iterrows():

                if prev_index is not None and duplicates[col][prev_index] != duplicates[col][index]:

                    print(f'----------------------')

                prev_index = index

                # In case of long column values, print only first n characters for printing purposes

                dup_value = row[col] if col not in ["abstract"] else row[col][:35]

                print (f'{index} - {col} :{dup_value}- Url: {row["url"]}')

        if update_df:

            num_rows_original = len(self.df.index)

            # Keep one instance of column duplicates and all null column values

            self.df = duplicate_less_df

            num_rows_filtered = len(self.df.index)

            print(f'Num rows before/after {col} duplicate removal keeping first:  {num_rows_original}/{num_rows_filtered}, removed rows: {num_rows_original - num_rows_filtered}')

        return duplicates  

    

    ##

    # Replaces meaningless values e.g. empty strings in some cases by None.

    # This is important for duplicate removal or outlier analysis.

    # \param col the column for which its values are to be cleaned

    # \return dataframe after replacing meaningless values.

    #

    def replace_meaningless_values(self, col):

        clean_df = self.df.copy()

        if col == "abstract":

            meaningless_vals = ['[Image: see text]', 'none', '', 'Unknown', '[Figure: see text]']

            clean_df[col] = clean_df["abstract"].replace(to_replace=meaningless_vals, value=None)

        return clean_df



    ##

    # Returns a dict constructed from the data frame with keys as the input chosen column

    # The values are list of rows to accomodate repititions.

    # \param key_col key of the dict

    # \return constructed dictionary 

    #

    def to_dict(self, key_col):

        tmp_dict = {}

        for index, row in self.df.iterrows():

            if row[col] not in tmp_dict:

                tmp_dict[row[col]] = [row]

            else:

                tmp_dict[row[col]].append(row)

        return tmp_dict
meta = PaperMeta(root_path);

dup_cols = ["cord_uid", "doi", "pmcid", "pubmed_id", "arxiv_id", "who_covidence_id", "mag_id", "abstract", 

           "authors", "sha", "title", "pdf_json_files", "pmc_json_files"]



for col_ in dup_cols:

    meta.remove_duplicates(col_, show_duplicates=False, update_df=False);

# meta_dict_sha = meta.df.to_dict("sha");
##

# Class to store and inspect json file objects

#

class Paper:

    def __init__(self, file_path, metadata_dict=None):

        with open(file_path) as file:

            json_obj = json.load(file)

            self.json_keys = list(json_obj.keys())

            self.metadata_keys = list(json_obj['metadata'].keys())

            # The ID changes with kaggle data updates but as of 12/5 it seems they mean sha hash

            self.paper_id = json_obj['paper_id']

            self.title = json_obj['metadata']['title']

            # List of dicts each representing an author

            self.authors = json_obj['metadata']['authors']

            # Dictionary with section titles as keys and correspondig text as values

            self.sections = {}

            # Back matter with similar structure to body text

            self.back_matter_sections = {}

            # Dict of dictionaries to references

            self.references = json_obj['bib_entries']

            # list of dictionary to references

            self.figures = json_obj['ref_entries']

            # Url to paper to be extracted from meta data file

            self.url = None

            # Article's source to be extracted from meta data file

            self.source_x = None

            # cord_uid unique paper identifier for CORD-19 to be extracted from meta data file

            self.cord_uid = None

            self.doi = None

            self.pmcid = None

            self.pubmed_id = None

            self.publish_time = None

            self.journal = None

            self.MAP_id = None # Microsoft Academic Paper

            self.arxiv_id = None

            # self.found_in_metadata = self.compliment_with_dict(metadata_dict)

            

            self.organize_in_sections(json_obj)

            

    ##

    # Organize paper contents into sections including abstract, body_text and back_matter

    #

    def organize_in_sections(self, json_obj):

        # if abstract key does not exist pass (meaning the paper does not have an abstract)

        # Note that this different from json_schema explanation that says that abstract is found in metada dictionary

        if 'abstract' in json_obj:

            for paragraph in json_obj['abstract']:

                if 'abstract' not in self.sections:

                    self.sections['abstract'] = [paragraph['text']]

                else:

                    self.sections['abstract'].append(paragraph['text'])



        # Body text

        for paragraph in json_obj['body_text']:

            # Sections and their contents

            if paragraph['section'] in self.sections:

                self.sections[paragraph['section']].append(paragraph['text'])

            else:

                self.sections[paragraph['section']] = [paragraph['text']]

        # Concatenate paragraphs from list to continuous string separated by new line.

        for sec, paragraphs_list in self.sections.items():

            self.sections[sec] = '\n'.join(self.sections[sec])



        # Back matter text

        for paragraph in json_obj['back_matter']:

            if paragraph['section'] in self.back_matter_sections:

                self.back_matter_sections[paragraph['section']].append(paragraph['text'])

            else:

                self.back_matter_sections[paragraph['section']] = [paragraph['text']]

        # Concatenate paragraphs from list to continuous string separated by new line.

        for sec, paragraphs_list in self.back_matter_sections.items():

            self.back_matter_sections[sec] = '\n'.join(self.back_matter_sections[sec])

    

    ##

    # Complement Paper info from sha based metadata dict, the dictionary has key to list of rows

    #

    def compliment_with_dict(self, metadata_dict):

        self_sha = self.paper_id

        if self_sha in metadata_dict:

            for df_row in metada_dict[self_sha]:

                if df_row['title'] == self.title and df_row['authors'] == self.authors:

                    # Url to paper to be extracted from meta data file

                    self.url = df_row['url']

                    # Article's source to be extracted from meta data file

                    self.source_x = df_row['source_x']

                    # cord_uid unique paper identifier for CORD-19 to be extracted from meta data file

                    self.cord_uid = df_row['cord_uid']

                    self.doi = df_row['doi']

                    self.pmcid = df_row['pcmid']

                    self.pubmed_id = df_row['pubmed_id']

                    self.publish_time = df_row['publish_time']

                    self.journal = df_row['journal']

                    self.MAP_id = df_row['Microsoft Academic Paper ID'] # Microsoft Academic PAper

                    self.arxiv_id = df_row['arxiv_id']

                    return True

        return False

    

    ##

    # Returns the number of words in the title

    #

    def title_length(self):

        if len(self.title) > 0:

            title_words = self.title.split()

            return len(title_words)

        return 0



    ##

    # Returns Word count in paper sections

    #

    def word_count(self):

        counter = 0

        for sec, text in self.sections.items():

            words_per_sec = text.split()

            counter += len(words_per_sec)

        return counter

    

    ##

    # Returns info related to the input criteria

    #

    def get_info(self, criteria):

        if criteria == "words_count":

            return self.word_count()

        elif criteria == "sections_count":

            return len(self.sections)

        elif criteria == "title_len":

            return self.title_length()

        elif criteria == "figs_count":

            return len(self.figures)

        elif criteria == "refs_count":

            return len(self.references)

        elif criteria == "has_abstract":

            return int('abstract' in self.sections)

        elif criteria == "authors_count":

            return len(self.authors)

        elif criteria == "has_back_matter":

            return int(len(self.back_matter_sections) > 0)

        else:

            raise Exception("Unsupported paper critera: " + criteria)

    

    # Get authors names

    def get_authors(self):

        authors = []

        for auth_dict in self.authors:

            tmp_dict = {}

            tmp_dict['order'] = len(authors)

            tmp_dict['first'] = auth_dict['first']

            tmp_dict['last'] = auth_dict['last']

            tmp_dict['email'] = auth_dict['email']

            tmp_dict['affiliation'] = auth_dict['affiliation']

            authors.append(tmp_dict)

        return authors

    # Text representation of the class

    def __repr__(self):

        summary = f'ID: {self.paper_id}\nTitle: {self.title}\n'

        letters_per_sec = 100

        for sec, text in self.sections.items():

            summary += f'{sec} -> {text[:letters_per_sec]}\n'

        summary += " --- Fig./ Tables ---\n"

        for fig_key, fig_dict in self.figures.items():

            summary += f'{fig_dict["type"]} -> {fig_dict["text"]}\n'

        summary += " --- Back Matter ---\n"

        for sec, text in self.back_matter_sections.items():

            summary += f'{sec} -> {text[:letters_per_sec]}\n'

        summary += " --- References ---\n"

        for ref_key, ref_dict in self.references.items():

            # Some references do not have reference ID's maybe this mean the paper is not in the dataset

            if "ref_id" in ref_dict:

                summary += f'{ref_dict["ref_id"]} -> '

            else:

                summary += f'{ref_key} -> '

            summary += f'{ref_dict["title"]}\n'

        return summary

        
paper_ex = Paper(all_json[15236])

paper_ex = Paper(all_json[5005])





print(f'======= Json object keys: {paper_ex.json_keys}')

print(f'======= Metadata keys: {paper_ex.metadata_keys}')



print(f"======= Paper Sections =====\n {paper_ex.sections.keys()}")

print(f"======= Paper Summary =====\n {paper_ex} ")



print(f"======= Authors =====\n {paper_ex.get_authors()}")
stats_dict = {'title_len': [], 'words_count': [], 'figs_count': [], 'refs_count': [], 'has_abstract':[]

               , 'authors_count':[], 'sections_count': [], 'has_back_matter': []}

for json_paper in all_json:

    paper_ = Paper(json_paper)

    for key, _ in stats_dict.items():

        stats_dict[key].append(paper_.get_info(criteria=key))

        

stats_df = pd.DataFrame(stats_dict)

# stats_df.head()

stats_df.info()

stats_df.describe()
def plot_1D_hist(df, col, bins, min_threshold=None, max_threshold=None):

    max_threshold = max(df[col]) if max_threshold is None else max_threshold

    min_threshold = min(df[col]) if min_threshold is None else min_threshold

    df_reduced = stats_df.loc[stats_df[col] < max_threshold]

    retained_percent = float(df_reduced.shape[0])/ df.shape[0]

    plt.hist(df_reduced[col], label=col, bins=bins, range=(min_threshold, max_threshold), histtype='stepfilled')

    plt.legend()

    plt.ylabel("Frequency")

    plt.title(f'Percentage of data {round(retained_percent, 3)}')

    plt.show()

    return df_reduced
col_to_thresholds = {'title_len': (0, 80), 'words_count': (0, 20000), 'figs_count': (0, 50), 'refs_count': (0, 300)

               , 'authors_count':(0, 50), 'sections_count': (0, 75)}

col_to_bins = {'title_len': range(col_to_thresholds['title_len'][1]), 'words_count': 80, 'figs_count': range(col_to_thresholds['figs_count'][1]),

               'refs_count': range(col_to_thresholds['refs_count'][1]), 'authors_count':range(col_to_thresholds['authors_count'][1]),

               'sections_count': range(col_to_thresholds['sections_count'][1])}

stats_df = stats_df.drop_duplicates()

reduced_df_all_cols = stats_df.copy()

for col, min_max_thresholds in col_to_thresholds.items():

    reduced_df_by_col = plot_1D_hist(stats_df, col, bins=col_to_bins[col], min_threshold=min_max_thresholds[0], max_threshold=min_max_thresholds[1])

    # reduced_df_all_cols = reduced_df_all_cols.merge(reduced_df_by_col, how='inner')

    reduced_df_all_cols = reduced_df_all_cols.merge(reduced_df_by_col)



    print(f'{col} - Complete stats_df shape: {stats_df.shape} , reduced stats_df shape: {reduced_df_all_cols.shape}')
pd.plotting.scatter_matrix(reduced_df_all_cols, figsize=(15, 15), hist_kwds={'bins':80});