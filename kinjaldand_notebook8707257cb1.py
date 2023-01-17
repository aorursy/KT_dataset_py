# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import nltk

text = nltk.word_tokenize("And now for something completely different")

nltk.pos_tag(text)



text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")

nltk.pos_tag(text)

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

#print(text.similar('woman'))



comment="Easy access off of I-10, and newly renovated room was nice, bringing room quality up to par with newer Drury Inns, including the one in north Phoenix."

text = nltk.word_tokenize(comment)

nltk.pos_tag(text)
