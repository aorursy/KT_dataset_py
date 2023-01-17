# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing all the required packages



import nltk

from scipy import sparse

import re 

import string

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords

from textblob import TextBlob

from datetime import datetime

import matplotlib.pyplot as plt
#loading dataset



# Reading the text data present in the directories. Each review is present as text file.

#if the data is not found in the given path then the following code to be run

if not (os.path.isfile('../input/end-to-end-text-processing-for-beginners/train.csv' and 

                       '../input/end-to-end-text-processing-for-beginners/test.csv')):

    path = '../input/imdb-movie-reviews-dataset/aclimdb/aclImdb/'

    train_text  =[] #creating empty list to create a dataset from the txt files in the given data

    train_label =[]

    test_text =[]

    test_label=[]

    train_data_path_pos = os.path.join(path,'train/pos/') #taking the path of the positive dataset

    train_data_path_neg = os.path.join(path,'train/neg/')#taking the path of the negative dataset

    

    #Sorting the dataset into the positive dataset and the negative dataset

    

    for data in ['train','test']: #for each data in the train and test folders in the dataset

        for label in ['pos','neg']: #for each pos and neg folders in the dataset

            for file in sorted(os.listdir(os.path.join(path,data,label))): ##for each file in the given path with the #train or test dataset and the pos/neg labels

                if file.endswith('txt'):#if the file has extension 'txt' or say if it is a text file

                    with open(os.path.join(path,data,label,file)) as file_data:#opening the text file from the given datset 

                        if data =='train':  #if the file is in the train folder

                            train_text.append(file_data.read()) #append the data from the file into the train_text list

                            train_label.append(1 if label=='pos' else 0) #append the label into the train_label list if the label = 'pos'/positive append 1 or else 0

                        else:  #if the file is in the train folder

                            test_text.append(file_data.read())

                            test_label.append(1 if label=='pos' else 0)

                            

                            

    #Creating the dataframe

    

    train_df = pd.DataFrame({'Review':train_text,'Label':train_label})

    test_df = pd.DataFrame({'Review':test_text,'Label':test_label})

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    test_df = test_df.sample(frac=1).reset_index(drop=True)

    

    #Exporting the dataframe into the csv file

    

    train_df.to_csv('train.csv')

    test_df.to_csv('test.csv')

    

else: #if the files found in the given dataset path

    train_df = pd.read_csv('../input/end-to-end-text-processing-for-beginners/train.csv',index_col=0)

    test_df = pd.read_csv('../input/end-to-end-text-processing-for-beginners/test.csv',index_col=0)

                        

#printing the shape/dimensions of the dataset

print('The shape if the training dataset is',train_df.shape)

print('The shape if the test dataset is',test_df.shape)

    
path