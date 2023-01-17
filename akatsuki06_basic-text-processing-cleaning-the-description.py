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

import re



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import nltk



from wordcloud import WordCloud, STOPWORDS



df = pd.read_csv('../input/winemag-data_first150k.csv')

df = df[:100]

df.head()
df['description'][0]
import re

description =  re.sub('[^a-zA-Z]',' ',df['description'][0])

description
description = description.lower()



description
#convert string to a list of words

description_words = description.split() 

#iterate over each word and include it if it is not stopword 

description_words = [word for word in description_words if not word in stopwords.words('english')]



description_words
ps = PorterStemmer()

description_words=[ps.stem(word) for word in description_words]

description_words
df['description'][0]=' '.join(description_words)

df['description'][0]
stopword_list = stopwords.words('english')

ps = PorterStemmer()

for i in range(1,len(df['description'])):

    description = re.sub('[^a-zA-Z]',' ',df['description'][i])

    description = description.lower()

    description_words = description.split()

    description_words = [word for word in description_words if not word in stopword_list]

    description_words = [ps.stem(word) for word in description_words]

    df['description'][i] = ' '.join(description_words)
df['description']