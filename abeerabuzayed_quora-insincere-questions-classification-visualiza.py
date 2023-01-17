import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd 

train_data = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

test_data = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_data.head(10)
#test_data.head(10)
print("Train shape : ", train_data.shape)

print("Test shape : ", test_data.shape)
train_data.columns
train_data= train_data.drop(['qid'], axis=1)

test_data= test_data.drop(['qid'], axis=1)
test_data.head(10)
train_data.isnull().sum()
test_data.isnull().sum()
sns.countplot(train_data['target'])
train_data['target'].value_counts()
sincere_percent= (len(train_data.question_text[train_data['target'] == 0]) /  len(train_data['question_text']) * 100)

insincere_percent= (len(train_data.question_text[train_data['target'] == 1]) / len(train_data['question_text']) * 100)
print(sincere_percent, insincere_percent)
import matplotlib.pyplot as plt

# Data to plot

labels = 'Sincere', 'Insincere'

sizes = [sincere_percent, insincere_percent]

colors = ['lightskyblue', 'lightcoral']

explode = (0.1, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
import nltk

from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict

train1_data = train_data[train_data["target"]==1]

train0_data = train_data[train_data["target"]==0]
def cloud(text, title, size = (10,7)):

    # Processing Text

    wordcloud = WordCloud(width=800, height=400, background_color ='white',

                          collocations=False

                         ).generate(" ".join(text))

    

    # Output Visualization

    fig = plt.figure(figsize=size, dpi=80)

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.axis('off')

    plt.title(title, fontsize=25,color='k')

    plt.tight_layout(pad=0)

    plt.show()

cloud(train_data['question_text'], title="Word Cloud of Questions")
cloud(train0_data["question_text"], title="Word Cloud of sincere Questions")
cloud(train1_data["question_text"], title="Word Cloud of insincere Questions")
## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]

## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train0_data["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_data["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]

import seaborn as sns

plt.figure(figsize=(11,10))

plt.title("Frequent words of sincere question")

fd_sorted0_head= fd_sorted0.head(40)

sns.barplot(x=fd_sorted0_head['wordcount'], y=fd_sorted0_head['word'])
plt.figure(figsize=(11,10))

plt.title("Frequent words of insincere question")

fd_sorted1_head= fd_sorted1.head(40)

sns.barplot(x=fd_sorted1_head['wordcount'], y=fd_sorted1_head['word'])
freq_dict = defaultdict(int)

for sent in train0_data["question_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_data["question_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]
import seaborn as sns

plt.figure(figsize=(11,10))

plt.title("Frequent words of sincere question")

fd_sorted0_head= fd_sorted0.head(40)

sns.barplot(x=fd_sorted0_head['wordcount'], y=fd_sorted0_head['word'])
import seaborn as sns

plt.figure(figsize=(11,10))

plt.title("Frequent words of insincere question")

fd_sorted1_head= fd_sorted1.head(40)

sns.barplot(x=fd_sorted1_head['wordcount'], y=fd_sorted1_head['word'])