!pip install git+git://github.com/llefebure/dynamic_bernoulli_embeddings.git
import pickle

import re



import numpy as np

import pandas as pd

from dynamic_bernoulli_embeddings.analysis import DynamicEmbeddingAnalysis

from dynamic_bernoulli_embeddings.training import train_model

from nltk import word_tokenize as nltk_word_tokenize

from gensim.corpora import Dictionary

from tqdm.notebook import tqdm

tqdm.pandas()
def _bad_word(word):

    if len(word) < 2:

        return True

    if any(c.isdigit() for c in word):

        return True

    if "/" in word:

        return True

    return False



def word_tokenize(text):

    text = re.sub(r"co-operation", "cooperation", text)

    text = re.sub(r"-", " ", text)

    words = [w.lower().strip("'.") for w in nltk_word_tokenize(text)]

    words = [w for w in words if not _bad_word(w)]

    return words
dataset = pd.read_csv("../input/un-general-debates/un-general-debates.csv")

dataset["bow"] = dataset.text.progress_apply(word_tokenize)

dataset["time"] = dataset.year - dataset.year.min()
dictionary = Dictionary(dataset.bow)

dictionary.filter_extremes(no_below=10, no_above=1.)

dictionary.compactify()

print(len(dictionary))
model, loss_history = train_model(

    dataset, dictionary.token2id, validation=.1, num_epochs=6, k=100)
loss_history.loss.plot(title="Training Loss")
loss_history.l_pos.plot(title="Positive")
loss_history.l_neg.plot(title="Negative")
loss_history.l_prior.plot(title="Prior")
np.save("embeddings", model.get_embeddings())

loss_history.to_csv("loss_history.csv", index=False)

pickle.dump(dictionary.token2id, open("dictionary.pkl", "wb"))
emb = DynamicEmbeddingAnalysis(model.get_embeddings(), dictionary.token2id)
emb.absolute_drift()
over_time = {}

for i in range(0, dataset.time.max() + 1, 5):

    col = str(dataset.year.min() + i)

    over_time[col] = emb.neighborhood("climate", i, 10)

pd.DataFrame(over_time)
over_time = {}

for i in range(0, dataset.time.max() + 1, 5):

    col = str(dataset.year.min() + i)

    over_time[col] = emb.neighborhood("afghanistan", i, 10)

pd.DataFrame(over_time)
pd.DataFrame([(dataset.year.min() + i, term) for i, term in emb.change_points(20)], columns=["Year", "Term"])