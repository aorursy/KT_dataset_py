import numpy as np 

import pandas as pd 

import nltk

import re

import string

from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.tokenize import word_tokenize

from subprocess import check_output

from nltk.stem import WordNetLemmatizer

print(check_output(["ls", "../input"]).decode("utf8"))
bio = pd.read_csv("../input/biology.csv")

cook = pd.read_csv("../input/cooking.csv")

crypto = pd.read_csv("../input/crypto.csv")

diy = pd.read_csv("../input/diy.csv")

robot = pd.read_csv("../input/robotics.csv")

travel = pd.read_csv("../input/travel.csv")

sample_sub = pd.read_csv("../input/sample_submission.csv")

test = pd.read_csv("../input/test.csv")

swords1 = stopwords.words('english')

punctuations = string.punctuation

all_dat = pd.concat([bio,cook,crypto,diy,robot,travel], ignore_index=True)
all_dat.to_csv('all_dat.csv')

test.to_csv('test.csv')

