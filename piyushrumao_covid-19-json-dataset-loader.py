# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import errno

import json

import pickle

import glob

import multiprocessing

from time import time  # To time our operations

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



print('import done')
def clean(txt):

    """

    Basic string loading code.



    :param txt:

    :return:

    """

    txt = re.sub(r'.\n+', '. ', txt)  # replace multiple newlines with period

    txt = re.sub(r'\n+', '', txt)  # replace multiple newlines with period

    txt = re.sub(r'\[\d+\]', ' ', txt)  # remove reference numbers

    txt = re.sub(' +', ' ', txt)

    

    txt = re.sub(',', ' ', txt)

    txt = re.sub(r'\([^()]*\)', '', txt)

    txt = re.sub(r'https?:\S+\sdoi', '', txt)

    txt = re.sub(r'biorxiv', '', txt)

    txt = re.sub(r'preprint', '', txt)

    txt = re.sub(r':', ' ', txt)

    return txt.lower()



class document():

    def __init__(self, file_path):

        if file_path:

            with open(file_path) as file:

                data = json.load(file)

                self.paper_id = data['paper_id']

                self.title = data['metadata']['title']

                self.abstract_tripples = {}

                self.text_tripples = {}

                self.key_phrases = ""

                self.abstract = ""

                self.text = ""

                self.entities = {}

                if 'abstract' in data:

                    for section in data['abstract']:

                        self.abstract = self.abstract + "\n" + section["text"]



                for section in data['body_text']:

                    self.text = self.text + "\n" + section['text']



    def clean_text(self):

        self.abstract = clean(self.abstract)

        self.text = clean(self.text)

        self.title =clean(self.title)

        final_data_dict = self.combine_data()

        return final_data_dict



    def combine_data(self):

        self.data = {'paper_id': self.paper_id,

                     'title': self.title,

                     'abstract': self.abstract,

                     'text': self.text,

                     'abstract_tripples': self.abstract_tripples,

                     'text_tripples': self.text_tripples,

                     'key_phrases': self.key_phrases,

                     'entities': self.entities}

        return self.data



    def extract_data(self):



        self.paper_id = self.data['paper_id']

        self.title = self.data['title']

        self.abstract = self.data['abstract']

        self.text = self.data['text']

        self.abstract_tripples = self.data['abstract_tripples']

        self.text_tripples = self.data['text_tripples']

        self.key_phrases = self.data['key_phrases']

        self.entities = self.data['entities']



    def save(self, dir):

        self.combine_data()



        if not os.path.exists(os.path.dirname(dir)):

            try:

                os.makedirs(os.path.dirname(dir))

            except OSError as exc:  # Guard against race condition

                if exc.errno != errno.EEXIST:

                    raise



        with open(dir, 'w') as json_file:

            json_file.write(json.dumps(self.data))



    def load_saved_data(self, dir):

        with open(dir) as json_file:

            self.data = json.load(json_file)

        self.extract_data()

desired_dirs=['/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset',

             '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',

             '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',

             '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license']





noncomm_use_subset=[]

biorxiv_medrxiv = []

comm_use_subset = []

custom_license = []



for individual_dirs in desired_dirs:

    files_list = []

    data = []

    print('############## Directory Working on #################')

    print(individual_dirs)

    print(individual_dirs.split('/')[-1])

    for dirname,_, filenames in os.walk(individual_dirs):

        #print(dirname)

        for filename in filenames:

            #print(os.path.join(dirname, filename))

            files_list.append(os.path.join(dirname, filename))

    print(len(files_list))

    #print(files_list)

    i=0

    for individual_file in files_list:

        try:

            pub = document(individual_file)

            data_dict = pub.clean_text()

            #print(data_dict)

            data.append(data_dict)

            i+=1

        except:

            pass

    print('Now writing back data')

    print('files processed===>'+str(i))

    with open(individual_dirs.split('/')[-1]+'.pickle', "wb") as f:

                pickle.dump(data,f)

    
total_dataframe = pd.DataFrame()

models = glob.glob('/kaggle/working/' + "*.pickle")

#models = glob.glob('/kaggle/working/biorxiv_medrxiv.pickle')

# print('models via glob===>'+str(models))

for individual_model in models:

    print(individual_model)

    # open a file, where you stored the pickled data

    file = open(individual_model, 'rb')



    # dump information to that file

    data = pickle.load(file)



    # close the file

    file.close()



    print('Showing the pickled data:')

    my_df = pd.DataFrame(data)

    print(my_df.shape)

    my_df = my_df.replace(r'^\s*$', np.nan, regex=True)

    total_dataframe = total_dataframe.append(my_df, ignore_index=True)

    print('Null per column')

    print(my_df.isnull().sum())



total_dataframe = total_dataframe.drop(['abstract_tripples', 'text_tripples','entities','key_phrases'], axis=1)
print('Total data in dataframe')

print(total_dataframe.shape)

total_dataframe.head()

total_dataframe.isnull().sum()
tf_df = pd.DataFrame()

tf_df['merged_text'] = total_dataframe['title'].astype(str) +  total_dataframe['abstract'].astype(str) +  total_dataframe['text'].astype(str)

tf_df['paper_id'] = total_dataframe['paper_id']

tf_df.head()