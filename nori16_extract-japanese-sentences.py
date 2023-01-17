# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from nltk.corpus.reader import XMLCorpusReader



print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
os.system('tar -xvf ../input/wiki_corpus_2.01.tar')
print(check_output(["ls", "./"]).decode("utf8"))
def readXml(file):

    reader = XMLCorpusReader('./', file)

    return reader.words()
import glob

file_list = glob.glob('BDS/*')



words = [readXml(f) for f in file_list]
import re

def filter_words(w):

    pattern1 = r"[a-zA-Z]+"

    pattern2 = r"[!-~]"

    repatter1 = re.compile(pattern1)

    repatter2 = re.compile(pattern2)

    if not repatter1.match(w) and not repatter2.match(w):

        return True

def flatten(nested_list):

    """2重のリストをフラットにする関数"""

    return [e for inner_list in nested_list for e in inner_list]

flatte_words = flatten(words)

ja_words = list(filter(lambda x: filter_words(x), flatte_words))
ja_words