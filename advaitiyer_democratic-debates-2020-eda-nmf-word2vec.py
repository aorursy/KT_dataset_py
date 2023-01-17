# Importing the essential libraries for the exercise.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Linking the directory to access the dataset.

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the data, and exploring its characteristics.

path = "../input/democratic-debate-transcripts-2020/debate_transcripts_v2_2020-02-23.csv"

data = pd.read_csv(path, encoding = "ISO-8859-1")

data[0:5]
# Dictionary of the debates.

debate_names = data["debate_name"]

number_of_debates = len(set(debate_names))

print("Total number of democratic debates:", number_of_debates, "debates")



# Dictionary of number of sections.

debate_section = data["debate_section"]

number_of_sections = len(set(debate_section))

print("Maximum number of sections in the debates:", number_of_sections, "sections")



# Dictionary of name of speakers.

dem_speakers = data["speaker"]

number_of_speakers = len(set(dem_speakers))

print("Total number of democratic speakers:",number_of_speakers, "speakers")



# Mean duration of speech.

print("The average speaking time is:",np.mean(data["speaking_time_seconds"]), "seconds")
# Sorted dictionary of debates.

debs = dict()

for i in debate_names:

    debs[i] = debs.get(i, 0) + 1



debates = {k: v for k, v in sorted(debs.items(), key=lambda item: item[1])}

 

# Sorted dictionary of sections.

secs = dict()

for i in debate_section:

    secs[i] = secs.get(i, 0) + 1



sections = {k: v for k, v in sorted(secs.items(), key=lambda item: item[1])}



# Sorted dictionary of speakers.

spkrs = dict()

for i in dem_speakers:

    spkrs[i] = spkrs.get(i, 0) + 1



speakers = {k: v for k, v in sorted(spkrs.items(), key=lambda item: item[1])}
# Plot of debates

import matplotlib.pyplot as plt



plt.bar(list(debates.keys()), debates.values(), color='red')

plt.title("Histogram of Debates")

plt.xticks(rotation = 90)

plt.rcParams["figure.figsize"] = (10,10)
# Plot of speakers' activity

plt.bar(list(speakers.keys()), speakers.values(), color='green')

plt.title("Histogram of Speakers' Activity")

plt.xticks(rotation=90)

plt.rcParams["figure.figsize"] = (20,20)
import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

random_state = 0



# Taking into consideration only nouns so as to identify the topics.

def only_nouns(texts):

    output = []

    for doc in nlp.pipe(texts):

        noun_text = " ".join(token.lemma_ for token in doc if token.pos_ == 'NOUN')

        output.append(noun_text)

    return output
# Merging the nouns-only list to the dataset.

data_new = only_nouns(data["speech"])

speech_nouns = pd.DataFrame(data_new)

data["Index"] = data.index

speech_nouns["Index"] = speech_nouns.index

democrat_data = pd.merge(data, speech_nouns, on="Index")

democrat_data.columns = ["debate_name", "debate_section", "speaker", "speech", "speaking_time_seconds", "index", "speech_nouns"]

democrat_data = democrat_data.drop(["index"], axis=1)

democrat_data.head()
# Number of topics to extract

n_topics = 10



# Vectorization of the nouns.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vec = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)

features = vec.fit_transform(democrat_data.speech_nouns)



# Non-negative matrix factorization.

from sklearn.decomposition import NMF

cls = NMF(n_components=n_topics, random_state=random_state)

cls.fit(features)
# List of unique words

feature_names = vec.get_feature_names()



# Number of top words per topic

n_top_words = 20



for i, topic_vec in enumerate(cls.components_):

    print(i, end=' ')

    for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:

        print(feature_names[fid], end=' ')

    print()
import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



democrat_speeches = list()

lines = data["speech"].tolist()



for line in lines:

    tokens = word_tokenize(line)

    # lowercase

    tokens = [word.lower() for word in tokens]

    # remove punctuation

    table = str.maketrans("","", string.punctuation)

    strip = [w.translate(table) for w in tokens]

    # remove remaining non-alphabets

    words = [word for word in strip if word.isalpha()]

    # filter stop-words

    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]

    democrat_speeches.append(words)
import gensim



# Train word2vec model

model = gensim.models.Word2Vec(sentences = democrat_speeches, size = 100, window = 5, workers = 4, min_count = 1)

# Vocab size

words = list(model.wv.vocab)

print("Vocabulary size: ", len(words))
print("Talked issue #1: Tax")

model.wv.most_similar("tax")
print("Talked issue #2: Healthcare")

model.wv.most_similar("healthcare")
print("Talked issue #3: Rebuttals")

model.wv.most_similar("rebuttal")
print("Talked issue #4: Policy")

model.wv.most_similar("policy")
print("Talked issue #5: Law")

model.wv.most_similar("law")
print("Talked issue #6: Immigration")

model.wv.most_similar("immigration")
print("Talked issue #7: Impeachment")

model.wv.most_similar("impeachment")
print("Talked issue #8: Climate")

model.wv.most_similar("climate")
print("Talked issue #9: Class")

model.wv.most_similar("class")
print("Talked issue #10: Racism")

model.wv.most_similar("racism")