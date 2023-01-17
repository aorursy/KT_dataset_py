from pandas import read_csv

data = read_csv("../input/7282_1.csv")
selected_columns = ["name", "reviews.rating", "reviews.text", "reviews.title"]

data = data[selected_columns]
data.head()
data.shape
from pandas import DataFrame, cut

DataFrame(cut(data["reviews.rating"],[i for i in range(0,11)], right=True).value_counts())
hotels = data["name"].value_counts()

hotels = hotels[hotels > 100]

DataFrame(hotels)
sample = data[data["name"] == "The Alexandrian, Autograph Collection"]
all_text = " ".join(sample["reviews.text"].apply(str).values)
from nltk import word_tokenize, FreqDist, bigrams, trigrams

from nltk.corpus import stopwords
tokens = [word for word in word_tokenize(all_text) if word.isalnum() and word not in stopwords.words("english")]
FreqDist(tokens).most_common(50)
FreqDist(bigrams(tokens)).most_common(50)
FreqDist(trigrams(tokens)).most_common(50)