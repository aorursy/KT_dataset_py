#GRAD-E1326: Python Programming for Data Scientists
#Ph.D. Hannah Béchara
#Ji Yoon Han & Mariana G. Carrillo 
#Initial Project Report: Tweet sentiment analysis
#Importing libraries for sentiment analysis

#Importing libraries for sentiment analysis 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk #Natural Language Processing Package 
import os #functions for interacting with the operating system
import spacy #Models for NLP
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import transformers
from transformers import BertForSequenceClassification
from wordcloud import WordCloud #For nice wordclouds
import tensorflow as tf #Package to develop train models 
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, classification_report

#Loading and cleaning data
train_data = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding='latin-1')
test_data = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv', encoding='latin-1')
#Train data 
#preview
train_data.head(5)

#info on nulls
train_data.info

#descriptive statistics
train_data.describe
#Test data
#preview
test_data.head(5)

#info on nulls
test_data.info

#descriptive statistics
test_data.describe
stop_words = stopwords.words('english')

def process_tweet(tweet):
    
    # remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)
    
    # remove html tags
    tweet = re.sub(r'<.*?>', ' ', tweet)
    
    # remove digits
    tweet = re.sub(r'\d+', ' ', tweet)
    
    # remove hashtags
    tweet = re.sub(r'#\w+', ' ', tweet)
    
    # remove mentions
    tweet = re.sub(r'@\w+', ' ', tweet)
    
    #removing stop words
    tweet = tweet.split()
    tweet = " ".join([word for word in tweet if not word in stop_words])
    
    return tweet
# Function taken from @Shahraiz's notebook


# Histogram 
sns.distplot(a=test_data['Sentiment'], kde=False) # Need to create dummies for histogram



