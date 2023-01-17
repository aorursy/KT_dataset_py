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
df = pd.read_fwf('/kaggle/input/game-of-thrones-books/002ssb.txt',sep='\s+', index_col=False,columns=['Index'])
df.columns
import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
sent_array=df.loc[0:500].to_numpy()
len(sent_array)
df1=pd.DataFrame(data=sent_array.flatten(),columns=['Sentences'])
df1['Sentences'][3]
from nltk.tokenize import word_tokenize

import re



def clean_text(

    string: str, 

    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',

    stop_words=['the', 'a', 'and', 'is', 'be', 'will']) -> str:

    """

    A method to clean text 

    """

    # Cleaning the urls

    string = re.sub(r'https?://\S+|www\.\S+', '', string)



    # Cleaning the html elements

    string = re.sub(r'<.*?>', '', string)



    # Removing the punctuations

    for x in string.lower(): 

        if x in punctuations: 

            string = string.replace(x, "") 



    # Converting the text to lower

    string = string.lower()



    # Removing stop words

    string = ' '.join([word for word in string.split() if word not in stop_words])



    # Cleaning the whitespaces

    string = word_tokenize(string)



    return string 
text_window=2

all_text=[]

word_lists=[]

for text in df1['Sentences']:

    text = clean_text(text)

    



    # Appending to the all text list

    all_text += text 



    # Creating a context dictionary

    for i, word in enumerate(text):

        for w in range(text_window):

            # Getting the context that is ahead by *window* words

            if i + 1 + w < len(text): 

                word_lists.append([word] + [text[(i + 1 + w)]])

                

            # Getting the context that is behind by *window* words    

            if i - w - 1 >= 0:

                word_lists.append([word] + [text[(i - w - 1)]])
def create_dict(text):

    word_list=list(set(text))

    word_list.sort()

    word_freq={};

    for i,word in enumerate(word_list):

        word_freq.update({

            word:i

        })

        

    return word_freq;
word_freq=create_dict(all_text)

word_lists
from scipy import sparse

from tqdm import tqdm

X=[]

Y=[]

num_words=len(word_freq)

words=word_freq.keys()



for i,context_word in tqdm(enumerate(word_lists)):

    main_word_index=word_freq.get(context_word[0])

    context_word_index=word_freq.get(context_word[1])

    

    X_row=np.zeros(num_words)

    Y_row=np.zeros(num_words)

    

    X_row[main_word_index]=1

    Y_row[context_word_index]=1

    

    X.append(X_row)

    Y.append(Y_row)

    

X = np.asarray(X)

Y = np.asarray(Y)
from keras.models import Input, Model

from keras.layers import Dense

import matplotlib.pyplot as plt
embed_size = 2

import itertools

# Defining the neural network

inp = Input(shape=(X.shape[1],))

x = Dense(units=embed_size, activation='linear')(inp)

x = Dense(units=Y.shape[1], activation='softmax')(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.fit(

    x=X, 

    y=Y, 

    batch_size=32,

    epochs=500,

    )
weights = model.get_weights()[0]
embedding_dict = {}

for word in words: 

    embedding_dict.update({

        word: weights[word_freq.get(word)]

        })
plt.figure(figsize=(10, 10))

for word in list(word_freq.keys()):

    coord = embedding_dict.get(word)

    plt.scatter(coord[0], coord[1])

    plt.annotate(word, (coord[0], coord[1]))
embedding_dict
try:

    os.mkdir(f'{os.getcwd()}\\output')        

except Exception as e:

    print(f'Cannot create output folder: {e}')
with open(f'{os.getcwd()}\\output\\embedding.txt', 'w') as f:

    for key, value in embedding_dict.items():

        try:

            f.write(f'{key}: {value}\n')   

        except Exception as e:

            print(f'Cannot write word {key} to dict: {e}')
with open('file_emb.txt','w') as data:

    data.write(str(embedding_dict))
import pandas as pd

df = pd.DataFrame.from_dict(embedding_dict)
df=df.T
df.to_csv('word_emb.csv')