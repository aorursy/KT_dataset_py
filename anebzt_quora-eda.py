import os



import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
!ls ../input/



print("\nEmbeddings:")

!ls ../input/embeddings/
print('File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
train = pd.read_csv('../input/train.csv').fillna(' ')

test = pd.read_csv('../input/test.csv').fillna(' ')
print("Shape of training set: ", train.shape)

print("Shape of test set: ", test.shape)



train_target = train['target'].values

np.unique(train_target)

print("\nPercentage of insincere questions irt sincere questions: ", train_target.mean(), "%")
train.sample(10)
test.sample(10)
insincere_q = train[train["target"] == 1]["question_text"].tolist()



with open('insinceres.txt', 'w') as f:

    for item in insincere_q:

        f.write("%s\n" % item)
from collections import defaultdict

from nltk.corpus import stopwords

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



stop_words = set(stopwords.words('english')) 

insinc_df = train[train.target==1]

sinc_df = train[train.target==0]



def plot_ngrams(n_grams):



    ## custom function for ngram generation ##

    def generate_ngrams(text, n_gram=1):

        token = [token for token in text.lower().split(" ") if token != "" if token not in stop_words]

        ngrams = zip(*[token[i:] for i in range(n_gram)])

        return [" ".join(ngram) for ngram in ngrams]



    ## custom function for horizontal bar chart ##

    def horizontal_bar_chart(df, color):

        trace = go.Bar(

            y=df["word"].values[::-1],

            x=df["wordcount"].values[::-1],

            showlegend=False,

            orientation = 'h',

            marker=dict(

                color=color,

            ),

        )

        return trace



    def get_bar(df, bar_color):

        freq_dict = defaultdict(int)

        for sent in df["question_text"]:

            for word in generate_ngrams(sent, n_grams):

                freq_dict[word] += 1

        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

        fd_sorted.columns = ["word", "wordcount"]

        trace = horizontal_bar_chart(fd_sorted.head(10), bar_color)

        return trace    



    trace0 = get_bar(sinc_df, 'blue')

    trace1 = get_bar(insinc_df, 'blue')



    # Creating two subplots

    if n_grams == 1:

        wrd = "words"

    elif n_grams == 2:

        wrd = "bigrams"

    elif n_grams == 3:

        wrd = "trigrams"

    

    fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                              subplot_titles=["Frequent " + wrd + " of sincere questions", 

                                              "Frequent " + wrd + " of insincere questions"])

    fig.append_trace(trace0, 1, 1)

    fig.append_trace(trace1, 1, 2)

    fig['layout'].update(height=500, width=1150, paper_bgcolor='rgb(233,233,233)', title=wrd + " Count Plots")

    py.iplot(fig, filename='word-plots')

plot_ngrams(1)
plot_ngrams(2)
plot_ngrams(3)
## Number of words in the text

train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text

train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text

train["num_chars"] = train["question_text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["question_text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text

train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))



## Average length of the words in the text

train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["mean_word_len"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
## Truncate some extreme values for better visuals ##

train['num_words'].loc[train['num_words']>50] = 50

train['num_unique_words'].loc[train['num_unique_words']>50] = 50

train['num_chars'].loc[train['num_chars']>300] = 300

train['mean_word_len'].loc[train['mean_word_len']>10] = 10



f, axes = plt.subplots(5, 1, figsize=(15,40))



sns.boxplot(x='target', y='num_words', data=train, ax=axes[0])

axes[0].set_xlabel('Target', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='target', y='num_unique_words', data=train, ax=axes[1])

axes[1].set_xlabel('Target', fontsize=12)

axes[1].set_title("Number of unique words in each class", fontsize=15)



sns.boxplot(x='target', y='num_chars', data=train, ax=axes[2])

axes[2].set_xlabel('Target', fontsize=12)

axes[2].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='target', y='num_stopwords', data=train, ax=axes[3])

axes[3].set_xlabel('Target', fontsize=12)

axes[3].set_title("Number of stopwords in each class", fontsize=15)



sns.boxplot(x='target', y='mean_word_len', data=train, ax=axes[4])

axes[4].set_xlabel('Target', fontsize=12)

axes[4].set_title("Mean word length in each class", fontsize=15)



plt.show()
print(train.columns)

train.head()