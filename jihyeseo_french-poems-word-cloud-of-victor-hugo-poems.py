# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
# first we check the encoding type

# look at the first ten thousand bytes to guess the character encoding

for size in range(1,7):
    length = 10 ** size
    with open("../input/" + filenames, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)
with open("../input/" + filenames, 'rb') as rawdata:
    print(rawdata.readlines(200))
 
df = pd.read_csv("../input/"+filenames, sep = '\t') 
print(df.dtypes)
df.head()
df.sample(10)
df.poet.unique()
df.poet.value_counts()
df.poem.sample(10)
hugopoems = df[['title','poem']][df.poet == 'victor-hugo']
hugopoems.head(5)
hugopoems.title.unique()
len(hugopoems)
# want to make word cloud, with help from https://github.com/cmchurch/nltk_french/blob/master/french-nltk.py

# https://stackoverflow.com/questions/46202600/creating-word-cloud-in-python-from-column-in-csv-file

from nltk.corpus import stopwords
stopset = stopwords.words('french')




bigtextlist = hugopoems.poem.tolist()


# https://stackoverflow.com/questions/42428390/nltk-french-tokenizer-in-python-not-working


import nltk
from nltk.tokenize import word_tokenize

content_french = bigtextlist #["Les astronomes amateurs jouent également un rôle important en recherche; les plus sérieux participant couramment au suivi d'étoiles variables, à la découverte de nouveaux astéroïdes et de nouvelles comètes, etc.", 'Séquence vidéo.', "John Richard Bond explique le rôle de l'astronomie."]
cleanedbigtextlist = []
for i in content_french:
       # print(i)
    cleanedbigtextlist.append(' '.join(word_tokenize(i, language='french')))
        
        
        


bigtext = "\n".join(cleanedbigtextlist)
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopset).generate(bigtext)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40 , stopwords=stopset).generate(bigtext)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# https://stackoverflow.com/questions/42428390/nltk-french-tokenizer-in-python-not-working

import nltk
from nltk.tokenize import word_tokenize

content_french = bigtextlist #["Les astronomes amateurs jouent également un rôle important en recherche; les plus sérieux participant couramment au suivi d'étoiles variables, à la découverte de nouveaux astéroïdes et de nouvelles comètes, etc.", 'Séquence vidéo.', "John Richard Bond explique le rôle de l'astronomie."]
for i in content_french:
       # print(i)
        print(word_tokenize(i, language='french'))
        
sorted(stopset)
