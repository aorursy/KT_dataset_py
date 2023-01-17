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
import pandas as pd



data=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

data.head()
from textblob import TextBlob



pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



data['polarity'] = data['text'].astype(str).apply(pol)

data['subjectivity'] = data['text'].astype(str).apply(sub)

data
def splitText(string,m,n):

    words = string.split()

    grouped_words = [' '.join(words[i: i + n]) for i in range(m, len(words), n)]

    return grouped_words
from textblob import TextBlob

from operator import itemgetter

# Split each routine into 10 parts

import numpy as np

import math



def split_text(text):

    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''



    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text

    length = len(text.split())

    split_list = []

    resultados = ()



    for i in range(length):        

        for j in range(length):

            for t in splitText(text,i,j+1):        

                split_list.append((t,TextBlob(t).sentiment.polarity))    

    

    resultado=(text,

            min(split_list,key=itemgetter(1))[0],

            min(split_list,key=itemgetter(1))[1],

            max(split_list,key=itemgetter(1))[0],

            max(split_list,key=itemgetter(1))[1])

    

    return resultado
# Let's create a list to hold all of the pieces of text

resultados_all = []

for index, row in data.iterrows():

    split = split_text(str(row['text']))

    resultados_all.append((row['textID'],) + split + (row['sentiment'],))



resultados_all
df = pd.DataFrame(resultados_all, columns=['textID', 'texto', 'texto_min', 'value_min','texto_max', 'value_max','sentiment'])
df['selected_text'] = np.where(df['sentiment']=='negative', df['texto_min'], df['texto_max'])
df_final=df[['textID','selected_text']]
df_final.to_csv('submission.csv', index=False)