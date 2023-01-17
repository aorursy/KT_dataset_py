import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd



from nltk import word_tokenize, FreqDist

from nltk.corpus import stopwords



from string import punctuation



stopwords_en = set(stopwords.words("english"))



def preprocess_text(text):

    # words = text.split()

    words = word_tokenize(text)

    

    # remove everything but alphabeticals

    words = [word for word in words if word.isalpha()]

    

    # remove punctuation

    table = str.maketrans('', '', punctuation)

    words = [w.translate(table) for w in words]

    

    # make lowercase

    words = [word.lower() for word in words]

    

    # remove stopwords

    words = [word for word in words if not word in stopwords_en]

    

    return words

    

filenames = os.listdir("../input")

n = len(filenames)

df = pd.DataFrame()

countries = ['Romania',

             'Sweden',

             'Slovakia',

             'Malta',

             'Italy',

             'Cyprus',

             'The Netherlands',

             'Slovenia',

             'Czech Republic',

             'Bulgaria',

             'Portugal',

             'Estonia',

             'Denmark',

             'Germany',

             'Belgium',

             'Croatia',

             'UK',

             'Ireland',

             'Austria',

             'Finland',

             'Latvia',

             'Lithuania',

             'Hungary',

             'Greece']



for idx, filename in enumerate(filenames):

    print("Reading {0} ({1}/{2}).".format(filename, idx + 1, n))

    with open(os.path.join("..", "input", filename), "r", encoding="ISO-8859-1") as fp:

        text = fp.read()

        words = preprocess_text(text)

        m = len(words)

        fd = FreqDist(words)

        freq_list = []

        for item in fd.items():

            freq_list.append(item + (countries[idx],))

        df = df.append(freq_list)

        

df.columns = ["Word", "Count", "Country"]

df.set_index(["Word", "Country"], inplace=True)



# create column frequency with word count normalized to number of words in document

df["Frequency"] = df.groupby("Country").transform(lambda x: x / x.count())



# create column document frequency (df) with number of occurences in all documents

df["DF"] = df.groupby(["Word"])["Count"].transform(lambda x: x.count())



# create column inverse document frequency (idf) with frequency is weighted against the occurence in other documents

df["IDF"] = df["Frequency"] * np.log(len(filenames) / df["DF"])



print(df.shape)
import matplotlib.pyplot as plt

%matplotlib inline



# plot occurences of "nuclear"

df.xs("nuclear", level="Word")["Frequency"].sort_values(ascending=True).plot.barh()

plt.xlabel("Frequency of 'nuclear' in NECP")
# plot occurences of "renewable"

df.xs("renewable", level="Word")["Frequency"].sort_values(ascending=True).plot.barh()

plt.xlabel("Frequency of 'renewable' in NECP")
df.xs("Sweden", level="Country").sort_values(ascending=False, by="IDF")