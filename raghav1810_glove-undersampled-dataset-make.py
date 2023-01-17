# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
GLOVE_DIM = 50  # Number of dimensions of the GloVe word embeddings

input_path = '../input'  # Path where all input files are stored
import sys

import regex as re



FLAGS = re.MULTILINE | re.DOTALL



def hashtag(text):

    text = text.group()

    hashtag_body = text[1:]

    if hashtag_body.isupper():

        result = "<hashtag> {} <allcaps>".format(hashtag_body)

    else:

        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))

    return result



def allcaps(text):

    text = text.group()

    return text.lower() + " <allcaps>"





def tokenize(text):

    # Different regex parts for smiley faces

    eyes = r"[8:=;]"

    nose = r"['`\-]?"



    # function so code less repetitive

    def re_sub(pattern, repl):

        return re.sub(pattern, repl, text, flags=FLAGS)



    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")

    text = re_sub(r"/"," / ")

    text = re_sub(r"@\w+", "<user>")

    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")

    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")

    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")

    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")

    text = re_sub(r"<3","<heart>")

    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")

    text = re_sub(r"#\S+", hashtag)

    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")

    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")



    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.

    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)

    text = re_sub(r"([A-Z]){2,}", allcaps)



    return text.lower().split()
glove_file = 'glove.twitter.27B.' + str(GLOVE_DIM) + 'd.txt'

glove_dir = 'glove-global-vectors-for-word-representation/'

emb_dict = {}

glove = open(input_path +"/"+ glove_dir + glove_file)

for line in glove:

    values = line.split()

    word = values[0]

    vector = np.asarray(values[1:], dtype='float32')

    emb_dict[word] = vector

glove.close()
def inGlove(lizt):

    return np.mean(np.array([emb_dict[i] for i in lizt if i in emb_dict]), axis=0)
data = pd.read_csv("../input/personality-dataset-undersampled1/Personality_dataset_undersampled1.csv")

data["word_vector"] = data.recent_tweets.apply(tokenize)
data.word_vector = data.recent_tweets.apply(inGlove)
data = data.dropna(subset=["word_vector"])
data.word_vector = data.word_vector.apply(lambda x: x.tolist())
data2 = pd.DataFrame(np.array(data.word_vector.tolist()))
data2["personality"] = data.hashtag
data2
data2.to_csv("personality_vector_dataset_undersampled_processed.csv")