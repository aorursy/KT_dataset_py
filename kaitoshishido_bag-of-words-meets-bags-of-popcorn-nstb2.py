import os

import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

import re

import nltk

from nltk.corpus import stopwords
train = pd.read_csv('../input/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

test = pd.read_csv('../input/testData.tsv', header=0, delimiter="\t", quoting=3)
# 元のデータ

print (train["review"][0])
# htmlタグの消去

example1 = BeautifulSoup(train["review"][0])

print (example1.get_text())
# 英字と空白のみの文字列に変換

letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text())

print (letters_only)
# 全てを小文字化

lower_case = letters_only.lower()



# 単語に分割

words = lower_case.split() 

print (words)
print (stopwords.words("english"))
words = [w for w in words if not w in stopwords.words("english")]

print (words)
# python3ではxrangeは使えないためrangeを使用する