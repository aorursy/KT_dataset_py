! pip install googletrans
!pip install translate
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

file = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from googletrans import Translator

from bs4 import BeautifulSoup

import string

import re

import json



class PreprocessData():

    def __init__(self,file_name):

        self.f_name = pd.read_csv(file_name)

        self.ps = PorterStemmer()

        self.translator = Translator()

        self.stop_words = set(stopwords.words('english')) 

        self.cnt=0

    

    def getFile(self):

        return self.f_name

    

    def decontract_word(self,data):

        decont = re.sub(r"won't", "will not", data)

        decont = re.sub(r"can\'t", "can not", decont)

        decont = re.sub(r"n\'t", " not", decont)

        decont = re.sub(r"\'re", " are", decont)

        decont = re.sub(r"\'s", " is", decont)

        decont = re.sub(r"\'d", " would", decont)

        decont = re.sub(r"\'ll", " will", decont)

        decont = re.sub(r"\'t", " not", decont)

        decont = re.sub(r"\'ve", " have", decont)

        decont = re.sub(r"\'m", " am", decont)

        return decont

    

    def clean_punct(self,data):

        try:

            new_str = [char for char in data if char not in string.punctuation]

            sent = ''.join(new_str)

            sent = re.sub(r"http\S+", "", sent)

            sent = BeautifulSoup(sent, 'lxml').get_text()

            sent = self.decontract_word(sent)

            sent = re.sub("\S*\d\S*", "", sent).strip()

            sent = re.sub('[^A-Za-z]+', ' ', sent)

            sent = [word.lower() for word in sent.split() if word.lower() not in self.stop_words]

            return ' '.join(sent)

        except Exception as e:

            print(e)

            return "unknown"

    

    def translate_and_clean_data(self,data):

        try:

            

            data = self.translator.translate(data,dest='en').text

            data = self.clean_punct(data)

            print(str(self.cnt)+" "+data)

#             print(str(self.cnt)+" "+self.translator.detect(data).lang)

            self.cnt+=1

            return data

        except Exception as e:

            print(e)

            return "unknown"

    

    def get_translated_clean_text(self,col_name):

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub('\\n',' ',str(x)))

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub("\[\[User.*",'',str(x)))

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))

#         self.f_name[col_name] = self.f_name[col_name].map(self.translate_and_clean_data)

        

    def clean_text(self,col_name):

        self.f_name[col_name] = self.f_name[col_name].map(self.clean_punct)

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub('\\n',' ',str(x)))

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub("\[\[User.*",'',str(x)))

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))

        self.f_name[col_name] = self.f_name[col_name].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))

    

    def save_csv_file(self,name):

        self.f_name.to_csv("pre_proc_file/"+name,index=False)
if __name__ == "__main__":

    pp1 = PreprocessData("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

    pp2 = PreprocessData("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

    pp3 = PreprocessData("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
dt = pp1.getFile()

print(dt['comment_text'].head(10))
pp1.clean_text("comment_text")

dt = pp1.getFile()

print(dt['comment_text'].head(15))

# pp1.save_csv_file("train_set.csv")
pp2.get_translated_clean_text('comment_text')

dt = pp2.getFile()

print(dt['comment_text'].head(10))

# pp2.save_csv_file("valid_set.csv")
pp3.get_translated_clean_text('content')

dt = pp3.getFile()

print(dt['content'].head(10))

# pp2.save_csv_file("test_set.csv")
!mkdir pre_proc_file