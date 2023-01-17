# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import glob
import re

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import math
from scipy.sparse import csr_matrix



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/facebookposts/bbc_228735667216.csv')
tokens = []
#df1['description']
#sentence = """At eight o'clock on Thursday morning... Arthur didn't feel very good."""



for i in range(df1['description'].__len__()):
    j = str(df1['description'][i])
    #print(j)
    tokens = tokens + nltk.word_tokenize(j)

#print(tokens)






tagged = nltk.pos_tag(tokens)

allWordsInNumber = tagged.__len__()
allNounInNumber = 0
dfOutNouns = []
dfOutWords = []
for i in range(allWordsInNumber):
    dfOutWords.append(tagged[i][0])
    if tagged[i][1] == 'NNP' or tagged[i][1] == 'NN':
        print(tagged[i][0])
        dfOutNouns.append(tagged[i][0])
        allNounInNumber = allNounInNumber+1

#print(allWordsInNumber)
print(allNounInNumber)

#df = pd.DataFrame(dfOutNouns, columns=["colummn"])
#df.to_csv("C:/pythonProject/t1/Noun9.csv", sep=',')
#df2 = pd.DataFrame(dfOutWords, columns=["colummn"])
#df2.to_csv("C:/pythonProject/t1/Word9.csv", sep=',')

#path =r'C:/pythonProject/t1/Nouns' # use your path
#path =r'C:/pythonProject/t1/' # use your path

#allFiles = glob.glob(path + "/*.csv")

#list_ = []

#for file_ in allFiles:
#    df = pd.read_csv(file_,index_col=None, header=0,skiprows=0,usecols=range(1,2))
#    list_.append(df)

#frame = pd.concat(list_, axis = 0, ignore_index = True)
#frame.to_csv("C:/pythonProject/t1/AllWords.csv", sep=',')


df1 = pd.read_csv('../input/facebookposts/abc_news_86680728811.csv')
out1 = df1[['name','description']]

df2 = pd.read_csv('../input/facebookposts/bbc_228735667216.csv')
out2 = df2[['name','description']]

df3 = pd.read_csv('../input/facebookposts/cbs_news_131459315949.csv')
out3 = df3[['name','description']]

df4 = pd.read_csv('../input/facebookposts/cnn_5550296508.csv')
out4 = df4[['name','description']]

df5 = pd.read_csv('../input/facebookposts/fox_and_friends_111938618893743.csv')
out5 = df5[['name','description']]

df6 = pd.read_csv('../input/facebookposts/fox_news_15704546335.csv')
out6 = df6[['name','description']]

df7 = pd.read_csv('../input/facebookposts/nbc_news_155869377766434.csv')
out7 = df7[['name','description']]

df8 = pd.read_csv('../input/facebookposts/npr_10643211755.csv')
out8 = df8[['name','description']]

df9 = pd.read_csv('../input/facebookposts/the_los_angeles_times_5863113009.csv')
out9 = df9[['name','description']]


frames = [out1,out2,out3,out4,out5,out6,out7,out8,out9]

result = pd.concat(frames)

#result.to_csv("C:/pythonProject/t1/raw.csv", sep=',')
#print(result)

#path =r'C:/pythonProject/t1/Nouns' # use your path
path =r'../input/facebookposts' # use your path

allFiles = glob.glob(path + "/*.csv")

list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0,skiprows=0,usecols=range(1,2))
    list_.append(df)

frame = pd.concat(list_, axis = 0, ignore_index = True)
#frame.to_csv("../input/combinedfile/AllWords.csv", sep=',')


df1 = pd.read_csv('../input/combinedfile/AllNouns.csv')
a = df1['colummn'].__len__()

df2 = pd.read_csv('../input/combinedfile/AllWords.csv')
b = df2['colummn'].__len__()

print(str(int(a)/int(b)*100)+"%")
df1 = pd.read_csv('../input/combinedfile/raw.csv')

mask = df1['description'].str.contains('Thailand', case = True, regex = False)
counter = 0
for i in range(mask.__len__()):
    if mask[i] is True:
        counter = counter+1
mask = mask.dropna()
#print(mask)
print(counter)
#df3 = df1[mask]

index = []
for i in range(len(mask)):
    index.append(i)
mask = mask.reindex(index)

for i in range(len(mask)):
    if mask[i] is True:
        print(df1['description'][i])

#result = pd.concat([df3['description']])
#result.to_csv("C:/pythonProject/t1/thaiDescription.csv", sep=',')

#print(df3['description'])
#counter2 = 0
#mask2 = df1['colummn'] == 'Thailand'
#for i in range(mask.__len__()):
#    if mask2[i] is True:
#        counter2 = counter2+1

#print(mask2)
#print(counter2)


#contents = ""
#with open('../input/thai-related-nouns/thaiNoun.txt', 'r') as file:
  #contents = file.read()
#tokens = []
#df1['description']
#sentence = """At eight o'clock on Thursday morning... Arthur didn't feel very good."""

tokens = nltk.word_tokenize(contents)
#print(contents)
#print(tokens)

tagged = nltk.pos_tag(tokens)

allWordsInNumber = tagged.__len__()
allNounInNumber = 0
dfOutNouns = []
dfOutWords = []
for i in range(allWordsInNumber):
    dfOutWords.append(tagged[i][0])
    if tagged[i][1] == 'NNP' or tagged[i][1] == 'NN':
        print(tagged[i][0])
        dfOutNouns.append(tagged[i][0])
        allNounInNumber = allNounInNumber+1

print(allWordsInNumber)
print(allNounInNumber)

df = pd.DataFrame(dfOutNouns, columns=["colummn"])
df.to_csv("thaiNoun.csv", sep=',')
df1 = pd.read_csv('../input/thai-related-nouns/thaiNoun.csv')

#print(df1['colummn'])

nlp_words = nltk.FreqDist(df1['colummn'])
nlp_words.plot(20)
