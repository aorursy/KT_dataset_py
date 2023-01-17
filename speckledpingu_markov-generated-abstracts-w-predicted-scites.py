# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import RidgeClassifierCV

from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_absolute_error



from collections import defaultdict

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_articles = pd.read_csv("../input/scirate_quant-ph.csv")

df_articles = df_articles.drop("Unnamed: 0",axis=1)

df_articles.head()
master_abstract_corpus = list(df_articles.abstract.values)

for index,row in df_articles.iterrows():

    master_abstract_corpus.append(row.abstract)



# Create the n-gram dictionary

abstract_ngrams = defaultdict(list)

for abstract in master_abstract_corpus:

    split_list = abstract.split()

    # Index to -2 rather than -1 so that the index does not fall out of bounds at the end

    # Then create a 2-gram tuple as the key for the 3rd word

    for i in range(len(split_list)-2):

        abstract_ngrams[(split_list[i],split_list[i+1])].append(split_list[i+2])



def generate_abstract():

    # Intitialize the random seed n-gram

    initial_char = "a"

    while not initial_char.isupper():

        seed = random.choice(list(abstract_ngrams.keys()))

        initial_char = seed[0][0]



    # Set up initial values with length equal to the number of sentences desired

    length = 10

    generated_abstract = []

    ngram_0 = seed[0]

    ngram_1 = seed[1]

    i = 0

    while i <= length:

        try:

            # Check if it's the end of a sentence, then resample for a new seed

            # Also, resample the seed for a new sentence so it does not simply duplicate an

            # entire abstract

            if ngram_1[-1] == ".":

                generated_abstract.append(ngram_1)

                initial_char = "a"

                while initial_char.islower():

                    seed = random.choice(list(abstract_ngrams.keys()))

                    ngram_0 = seed[0]

                    ngram_1 = seed[1]

                    initial_char = seed[0]

                i += 1

            # Check to see if ngram_0 has information in it

            elif ngram_0 is not []:

                generated_abstract.append(ngram_0)

                generating_ngram = (ngram_0,ngram_1)

                following_word = random.choice(list(abstract_ngrams[generating_ngram]))

                ngram_0 = ngram_1

                ngram_1 = following_word

            # Else, resample in case of failure

            elif ngram_0 is []:

                initial_char = "a"

                while initial_char.islower():

                    seed = random.choice(list(abstract_ngrams.keys()))

                    ngram_0 = seed[0]

                    ngram_1 = seed[1]

                    initial_char = seed[0]

        except:

            pass

    return " ".join(generated_abstract)
to_be_graded_abstracts = []

for num_abstracts in range(4):

    to_be_graded_abstracts.append(generate_abstract())
df_articles.scites.describe()
X = df_articles.abstract

y = df_articles.scites

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y)



rfr = RandomForestRegressor()

rfr.fit(X_train,y_train)

y_pred = rfr.predict(X_test)

mean_absolute_error(y_test,y_pred)
alphas = 10**np.linspace(-4,2,150)

ridge = RidgeClassifierCV(alphas=alphas)

ridge.fit(X_train,y_train)

y_pred = ridge.predict(X_test)

mean_absolute_error(y_test,y_pred)
vectorized_to_be_graded_abstracts = tfidf.transform(np.asarray(to_be_graded_abstracts))

y_pred = rfr.predict(vectorized_to_be_graded_abstracts)

for i in range(len(y_pred)):

    print(y_pred[i])

    print(to_be_graded_abstracts[i])

    print("\n")