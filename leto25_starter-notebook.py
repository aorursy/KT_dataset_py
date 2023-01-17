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
import csv

import re

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
samples = []

labels = []

with open("/kaggle/input/hotel-reviews-classification/train.csv", newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=",", quotechar='"')

    next(reader) #skip csv header

    for row in reader:

        samples += [(row[1], row[2])]
print(f"{samples[23]}")
def clean_sample(text):

    text = text.lower()

    text = re.sub("[^а-яА-Яa-zA-Z0-9]", " ", text)

    text = re.sub("\s+", " ", text)

    text = text.strip()

    return text
samples = [(clean_sample(s[0]), s[1]) for s in samples]
print(f"{samples[23]}")
embeddings_dict = {}

with open("/kaggle/input/glove6b100d/glove.6B.100d.txt") as emb_file:

    for line in emb_file.readlines():

        items = line.split()

        word = items[0]

        vector = np.asarray(items[1:], "float32")

        embeddings_dict[word] = vector
embeddings_dict["unk"]
def get_sentence_embedding(sentence, embeddings_dict):

    words = sentence.split()

    

    embeddings = []

    unk_words = []

    for word in words:

        if word in embeddings_dict.keys():

            embeddings += [embeddings_dict[word]]

        else:

            embeddings += [embeddings_dict["unk"]]

    

    if len(embeddings) < 1:

        print(f"Wrong sentence: {sentence}")

        embeddings += [embeddings_dict["empty"]]

        embeddings += [embeddings_dict["sentence"]]

    

    embeddings = np.asarray(embeddings)

    sentence_embedding = np.mean(embeddings, axis=0)

    return sentence_embedding
get_sentence_embedding(samples[23][0], embeddings_dict)
samples = [(get_sentence_embedding(s[0], embeddings_dict), s[1]) for s in samples]

samples = [s for s in samples if len(s[0]) > 0]
data = [s[0] for s in samples]

labels = [s[1] for s in samples]

X_train, X_test, y_train, y_test = train_test_split(data, labels)
svm = LinearSVC(loss="squared_hinge", C=1.0, dual=False)

svm.fit(X_train, y_train)
preds = svm.predict(X_test)

accuracy_score(y_test, preds)
test_samples = []

with open("/kaggle/input/hotel-reviews-classification/test.csv", newline='') as csvfile:

    reader = csv.reader(csvfile, delimiter=",", quotechar='"')

    next(reader) #skip csv header

    for row in reader:

        test_samples += [(row[0], row[1])]
len(test_samples)
test_samples = [(s[0], clean_sample(s[1])) for s in test_samples]
test_samples[23][1]
result = [(s[0], svm.predict([get_sentence_embedding(s[1], embeddings_dict)])[0]) for s in test_samples]
result
with open("/kaggle/working/prediction.csv", "w") as pred_file:

    pred_file.write("Id,Prediction\n")

    result = [f"{r[0]},{r[1]}" for r in result]

    result = "\n".join(result)

    pred_file.write(result)