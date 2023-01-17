from gensim.models.word2vec import Word2Vec

from gensim.models.fasttext import FastText

from multiprocessing import cpu_count

import gensim.downloader as api

from bs4 import BeautifulSoup

from nltk.tokenize import word_tokenize
xmlFile = open("../input/usc26.xml")

xml = BeautifulSoup(xmlFile, features="xml")

xmlFile.close()
# extract and filter <section> elements

sections = xml.findAll("section")

sections = [section for section in sections if section.num["value"]]

section_list = [section.getText(separator=" ") for section in sections]

len(section_list)

# 
section_list[0][0:250]
def tokenizeSections(section_list):

  words = []

  for section in section_list:

    words.append([word.lower() for word in word_tokenize(section) if word.isalnum()])

  return words



words = tokenizeSections(section_list)
w2v_model = Word2Vec(words, min_count = 4, workers=cpu_count())
ft_model = FastText(words, min_count = 4, workers=cpu_count(), iter=30)
w2v_model.wv.most_similar("deduction")
ft_model.wv.most_similar("deduction")