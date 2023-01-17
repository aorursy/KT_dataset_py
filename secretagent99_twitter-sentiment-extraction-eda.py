import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
os.listdir('../input/')
train_data = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test_data = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train_data.head()

print("The Training data shape: ", train_data.shape, "\n")
train_data.head()
print("The Testing data shape: ", test_data.shape, "\n")
test_data.head()
#check unique labels in sentiment
print(train_data['sentiment'].unique())
#count total sentiment of each label of sentiment
train_data['sentiment'].value_counts()
train_data.isna().sum() # any null value in train data
test_data.isna().sum() #any null value in test data
train_data.describe()
sns.countplot(train_data['sentiment'], label="Count")
plt.show()
def clean(text):
    tweet_blob = TextBlob(text)
    return ' '.join(tweet_blob.words)

print(clean(train_data['text'].iloc[10]))
    

