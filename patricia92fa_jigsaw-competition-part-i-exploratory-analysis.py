import numpy as np

import pandas as pd

import json

from tqdm import tqdm

tqdm.pandas()

pd.set_option("display.precision", 2)



import warnings

warnings.filterwarnings('ignore')



# import os for system interaction and garbage collector

import os

import gc



# import textblog for text processing

from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# import gensim for statistical semantical analysis and structure

import gensim



from sklearn.model_selection import KFold



from keras.layers import *

from keras.initializers import *

from keras.constraints import *

from keras.regularizers import *

from keras.activations import *

from keras.optimizers import *

import keras.backend as K

from keras.models import Model

from keras.utils import plot_model

from keras.utils.vis_utils import model_to_dot

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping, ModelCheckpoint



from IPython.display import SVG

import matplotlib.pyplot as plt

import seaborn as sns
import zipfile



# unzip file to specified path

def import_zipped_data(file, output_path):

    with zipfile.ZipFile("../input/jigsaw-toxic-comment-classification-challenge/"+file+".zip","r") as z:

        z.extractall("/kaggle/input")

        

datasets = ['train.csv', 'test.csv', 'test_labels.csv']



kaggle_home = '/kaggle/input'

for dataset in datasets:

    import_zipped_data(dataset, output_path = kaggle_home)
test_df = pd.read_csv('/kaggle/input/test.csv')

train_df = pd.read_csv('/kaggle/input/train.csv')
train_df.head()
# col with input text

TEXT = 'comment_text'
non_toxic = len(train_df[train_df['toxic'] == 0])

toxic = len(train_df[train_df['toxic'] == 1])

print(f'There are {non_toxic} non-toxic comments, representing a {round((non_toxic/len(train_df)*100))}% of the total {len(train_df)} samples collected.')

print(f'Only {toxic} - about a {round((toxic/len(train_df))*100)}% - are toxic comments.')
plt.barh(['non_toxic', 'toxic'], [non_toxic, toxic], color = 'r', alpha = 0.5)

plt.title('Toxicity distribution')

plt.show()
labels = ['obscene', 'threat', 'insult', 'identity_hate']

class_cnt = {}

for label in labels:

    # count number of samples per toxicity type

    class_cnt[label] = len(train_df[train_df[label] == 1])

    

# sort dict from bigger to lower key value

class_cnt = {k: v for k, v in sorted(class_cnt.items(), key = lambda item: item[1], reverse = True)}
plt.bar(*zip(*class_cnt.items()), color = 'r', alpha = 0.5)

plt.title('Toxicity type distribution')

plt.show()
print(f'The percentage respect to toxic comments of each toxicity subtype are:')

for label in labels:

    print(f'>> {label} comments: {round((class_cnt[label]/toxic)*100)}%')
print(f'The percentage respect to toxic comments of each toxicity subtype are:')

for label in labels:

    print(f'>> {label} comments: {round((class_cnt[label]/len(train_df))*100)}%')
labels = ['toxic', 'severe_toxic']

class_cnt = {}

for label in labels:

    # count number of samples per toxicity type

    class_cnt[label] = len(train_df[train_df[label] == 1])

    

# sort dict from bigger to lower key value

class_cnt = {k: v for k, v in sorted(class_cnt.items(), key = lambda item: item[1], reverse = True)}
plt.bar(*zip(*class_cnt.items()), color = 'r', alpha = 0.5)

plt.title('Toxicity level distribution')

plt.show()
# compute character length of comments

lengths = train_df[TEXT].apply(len)

lengths_df = lengths.to_frame()

# print basic metrics

lengths.mean(), lengths.std(), lengths.min(), lengths.max()
lengths = train_df[TEXT].apply(len)

lengths_df = lengths.to_frame()

sns.boxplot(x=lengths_df, color='r')

plt.title('Boxplot of characters per sentence')

plt.show()
Q1, Q3 = lengths_df.quantile(0.25), lengths_df.quantile(0.75)

IQR = Q3 - Q1

IQR
legths_df_iqr = lengths_df[lengths_df[TEXT] < int(round(IQR))]

sns.boxplot(x=legths_df_iqr, color='r')

plt.title('Boxplot of characters per sentence without IQR Outliers')

plt.show()
lengths = train_df[TEXT].apply(len)

train_df['lengths'] = lengths

lengths = train_df.loc[train_df['lengths']<1125]['lengths']

sns.distplot(lengths, color='r')

plt.title('Number of characters per sentence')

plt.show()
words = train_df[TEXT].apply(lambda x: len(x) - len(''.join(x.split())) + 1)

train_df['words'] = words

words = train_df.loc[train_df['words']<200]['words']

sns.distplot(words, color='r')

plt.title('Number of words per sentence')

plt.show()
avg_word_len = train_df[TEXT].apply(lambda x: 1.0*len(''.join(x.split()))/(len(x) - len(''.join(x.split())) + 1))

train_df['avg_word_len'] = avg_word_len

avg_word_len = train_df.loc[train_df['avg_word_len']<10]['avg_word_len']

sns.distplot(avg_word_len, color='b')

plt.title('Average word length')

plt.show()
# take a small sample of training dataset to speed up sentiment analysis

tiny_train_df = train_df.sample(n=10000)
import matplotlib.patches as mpatches

non_toxic_0 = tiny_train_df.loc[(tiny_train_df.toxic<0.5) & (tiny_train_df.words<200)]['words']

toxic_1 = tiny_train_df.loc[(tiny_train_df.toxic>0.5) & (tiny_train_df.words<200)]['words']

sns.distplot(non_toxic_0, color='green')

sns.distplot(toxic_1, color='red')

red_patch = mpatches.Patch(color='red', label='Toxic')

green_patch = mpatches.Patch(color='green', label='Non-Toxic')

plt.legend(handles=[red_patch, green_patch])

plt.title('Toxicity per word number in text')

plt.show()
# take a small sample of training dataset to speed up sentiment analysis

tiny_train_df = train_df.sample(n=10000)
sia = SentimentIntensityAnalyzer()

non_toxic_0 = tiny_train_df.loc[tiny_train_df.toxic<0.5][TEXT].apply(lambda x: sia.polarity_scores(x))

toxic_1 = tiny_train_df.loc[tiny_train_df.toxic>0.5][TEXT].apply(lambda x: sia.polarity_scores(x))
sns.distplot([polarity['neg'] for polarity in non_toxic_0], color='green')

sns.distplot([polarity['neg'] for polarity in toxic_1], color='red')

red_patch = mpatches.Patch(color='red', label='Toxic')

green_patch = mpatches.Patch(color='green', label='Non-Toxic')

plt.legend(handles=[red_patch, green_patch])

plt.title('Distribution of negativity in comments')

plt.show()
sns.distplot([polarity['pos'] for polarity in non_toxic_0], color='green')

sns.distplot([polarity['pos'] for polarity in toxic_1], color='red')

red_patch = mpatches.Patch(color='red', label='Toxic')

green_patch = mpatches.Patch(color='green', label='Non-Toxic')

plt.legend(handles=[red_patch, green_patch])

plt.title('Distribution of positivity in comments')

plt.show()
sns.distplot([polarity['neu'] for polarity in non_toxic_0], color='green')

sns.distplot([polarity['neu'] for polarity in toxic_1], color='red')

red_patch = mpatches.Patch(color='red', label='Toxic')

green_patch = mpatches.Patch(color='green', label='Non-Toxic')

plt.legend(handles=[red_patch, green_patch])

plt.title('Distribution of neutrality in comments')

plt.show()
sns.distplot([polarity['compound'] for polarity in non_toxic_0], color='green')

sns.distplot([polarity['compound'] for polarity in toxic_1], color='red')

red_patch = mpatches.Patch(color='red', label='Toxic')

green_patch = mpatches.Patch(color='green', label='Non-Toxic')

plt.legend(handles=[red_patch, green_patch])

plt.title('Distribution of complexity in comments')

plt.show()
from wordcloud import WordCloud



def class_wordcloud(dataframe, label, max_words):

    # data preprocessing: concatenate all reviews per class

    text = " ".join(x for x in dataframe[dataframe[label]==1].comment_text)



    # create and generate a word cloud image

    wordcloud = WordCloud(max_words=max_words, background_color="white", collocations=False).generate(text)



    # display the generated image

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(f"Most popular {max_words} words in class {label}")

    plt.show()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for label in labels:

    class_wordcloud(train_df, label, 50)
# data preprocessing: concatenate all reviews per class

text = " ".join(x for x in train_df[train_df['toxic']==0].comment_text)



# create and generate a word cloud image

wordcloud = WordCloud(max_words=50, background_color="white", collocations=False).generate(text)



# display the generated image

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title(f"Most popular 50 words for non-toxic comments")

plt.show()