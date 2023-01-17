import pandas as pd



df = pd.read_csv("../input/nlp-getting-started/train.csv")

print(df.shape)

print(df.head())

df.describe(include='all')
locations = df.location

keywords = df.keyword

text = df.text
import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



def describe_column(column, column_name):

    data = Counter(column)

    print("Most common {}: {}".format(column_name, data.most_common(3)))

    data = pd.DataFrame.from_dict(data, orient='index')

    sns.distplot(data)

    plt.show()

    

describe_column(keywords, "keywords")

describe_column(locations, "locations")

describe_column(text, "text")
text_lengths = Counter(map(len, text))

print("Most common text lengths: {}".format(text_lengths.most_common(5)))

data = pd.DataFrame.from_dict(text_lengths, orient='index')

sns.distplot(data)

plt.show()
