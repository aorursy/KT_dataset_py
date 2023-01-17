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
data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
data.head()
data.shape
data1 = data.copy()
# For Lowercase

data1.Message.apply(lambda i: " ".join(i.lower() for i in i.split()))
#For Uppercase

data1.Message.apply(lambda i: " ".join(i.upper() for i in i.split()))
data2 = data1.Message.apply(lambda i: " ".join(i.lower() for i in i.split()))
data2.str.replace("[^\w\s]","")
data3 = data2.str.replace("[^\w\s]","")
data3.str.replace("\d","")
data4 = data3.str.replace("\d","")
import nltk

from nltk.corpus import stopwords
### I will use stopwords in the English language.

sw = stopwords.words("english")
#Showing some sample from the sw

print(sw[:28])
data4.apply(lambda x: " ".join(x for x in x.split() if x not in sw))
data5 = data4.apply(lambda x: " ".join(x for x in x.split() if x not in sw))
for i in range(0,10):

    print(data4[i])
for i in range(0,10):

    print(data5[i])
data5
data5 = pd.DataFrame(data5, columns=["Message"])

data5.head()
#Looking at how many quantities each word there are in the dataset.

words_counts = pd.Series(" ".join(data5.Message).split()).value_counts()

words_counts
low = words_counts[words_counts<5]

len(low)
high = words_counts[words_counts>100]

len(high)
data5 = data5.Message.apply(lambda x: " ".join(x for x in x.split() if x not in low))

data5 = data5.apply(lambda x: " ".join(x for x in x.split() if x not in high))
data5
nltk.download("punkt")
import textblob

from textblob import TextBlob
data5[1]
TextBlob(data5[1]).words
data5.apply(lambda x: TextBlob(x).words)
data6 = data5.apply(lambda x: TextBlob(x).words)
from nltk.stem import PorterStemmer

st = PorterStemmer()
data5.apply(lambda x: " ".join(st.stem(i) for i in x.split()))
from textblob import Word

nltk.download("wordnet")
data5.apply(lambda x: " ".join(Word(i).lemmatize() for i in x.split()))