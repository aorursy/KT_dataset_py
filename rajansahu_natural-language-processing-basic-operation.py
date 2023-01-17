# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Words Tokenization
from nltk.tokenize import word_tokenize
sentence= "Sun rises in the east"
print(word_tokenize(sentence))
#sentences Tokenization
from nltk.tokenize import sent_tokenize
example_text= "Sun rises in the east. Sun sets in the west."
print(sent_tokenize(example_text))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_text="This is an example showing off stop word filtration."

stop_words=set(stopwords.words('english'))
words = word_tokenize(example_text)
filtered_sentence =[w for w in words if not w in stop_words]
print(filtered_sentence)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
example_words= ["python","pythoner","pythoning","pythoned"]
for w in example_words:
  print(ps.stem(w))


text="It is very important to be pythonly while you are pythoning with python"
words=word_tokenize(text)
for t in words:
  print(ps.stem(t))
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer =WordNetLemmatizer()
print(lemmatizer.lemmatize("lowest",pos='a'))
from nltk.tokenize import word_tokenize
example_text="Sun rises in the east"
words = word_tokenize(example_text)
def POS_Tagging():
  try:
    for i in words:
      #print(words)
      tagged =nltk.pos_tag(words) 
    print(tagged)
  except Exception as e :
    print(str(e))

POS_Tagging()