# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import re

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.




from wand.image import Image as Img

Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2017/february 2017/RATES MANAGER 5601 REVISED.pdf', resolution=300)







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import os

import numpy as np

from datetime import datetime

from collections  import Counter

from nltk import word_tokenize

import seaborn as sns

import matplotlib.pyplot as plt

import calendar

from wordcloud import WordCloud ,STOPWORDS

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

print(os.listdir("../input"))

from gensim.models import word2vec

from sklearn.manifold import TSNE

from nltk import pos_tag

from nltk.help import upenn_tagset

import gensim

import matplotlib.colors as mcolors

from nltk import jaccard_distance

from nltk import ngrams

#import textstat

plt.style.use('ggplot')



# job descriptions that are in the format of the text files

# additional data which is in the format of pdf and images

bulletins=os.listdir("../input/cityofla/CityofLA/Job Bulletins/")

additional=os.listdir("../input/cityofla/CityofLA/Additional data/")
# checking all the subsdaries

files=[dir for dir in os.walk('../input/cityofla')]

for file in files:

    print(os.listdir(file[0]))

    print("\n")
csvfiles=[]

for file in additional:

    if file.endswith('.csv'):

        print(file)

        csvfiles.append("../input/cityofla/CityofLA/Additional data/"+file)
print(csvfiles)
job_titles = csvfiles[0]

job_titles = pd.read_csv(job_titles)

print("The number of rows are %d and columns are %d"%(job_titles.shape))
display(job_titles)
job_sample_class = csvfiles[1]

job_sample_class = pd.read_csv(job_sample_class)

print("The number of rows are %d and columns are %d"%(job_sample_class.shape))
job_sample_class.head()
kaggle_data_dictionary = csvfiles[2]

kaggle_data_dictionary = pd.read_csv(kaggle_data_dictionary)

print("The number of rows are %d and columns are %d"%(kaggle_data_dictionary.shape))
kaggle_data_dictionary.head()
#let's checkout how many files are there in our bulletins

print("There are about %d files in our bulletins"%len(bulletins))
# code taken from https://www.kaggle.com/shahules/discovering-opportunities-at-la

def get_headings(bulletin):       

    

    """"function to get the headings from text file

        takes a single argument

        1.takes single argument list of bulletin files"""

    

    with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:    ##reading text files 

        data=f.read().replace('\t','').split('\n')

        data=[head for head in data if head.isupper()]

        return data

        

def clean_text(bulletin):      

    

    

    """function to do basic data cleaning

        takes a single argument

        1.takes single argument list of bulletin files"""

                                            

    

    with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:

        data=f.read().replace('\t','').replace('\n','')

        return data

    

     

#lets read file 

with open('../input/cityofla/CityofLA/Job Bulletins/SENIOR HOUSING INSPECTOR 4244 042718.txt','r') as f:

    data = f.read()

    print(data)

    f.close()

    
get_headings(1)
get_headings(2)