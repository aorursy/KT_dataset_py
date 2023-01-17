import pickle

import pandas as pd

import numpy as np

import string

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

#load the pickle file, (that is containing only top-15 conditions)

with open("../input/text-classification-1-eda/top_10_train.pickle", 'rb') as data:

    train_df = pickle.load(data)

    

with open("../input/text-classification-1-eda/top_10_test.pickle", 'rb') as data:

    test_df = pickle.load(data)
df_all = pd.concat([train_df, test_df]).reset_index(drop=True)

print(df_all.shape)

df_all.head()
#our new dataframe

train_features=df_all.loc[: ,["condition", "review", "review_length"]].reset_index(drop=True)

train_features.head()
train_features.loc[0]['review']
#remove special characters



train_features['review_parsed_1'] = train_features['review'].str.replace("\r", " ")

train_features['review_parsed_1'] = train_features['review_parsed_1'].str.replace("\n", " ")

train_features['review_parsed_1'] = train_features['review_parsed_1'].str.replace("    ", " ")

train_features['review_parsed_1'] = train_features['review_parsed_1'].str.replace('"', '')
#remove punctuation signs and numbers



punk_num = string.punctuation + "1234567890"

train_features['review_parsed_2'] = train_features['review_parsed_1']



for punct_sign in punk_num:

    train_features['review_parsed_2'] = train_features['review_parsed_2'].str.replace(punct_sign, '')
#make all letters lowercase



train_features['review_parsed_3'] = train_features['review_parsed_2'].str.lower()
#remove possessive pronouns



train_features['review_parsed_4'] = train_features['review_parsed_3'].str.replace("'s", "")
#Lemmatization downloading punkt and wordnet from NLTK



nltk.download('punkt')

print("--------------------------")

nltk.download('wordnet')





#create a lemmatizer object

wordnet_lemmatizer = WordNetLemmatizer()
# Iterate thru. every word with lemmatizer

# https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/

# https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html



len_rows = len(train_features)

lemmatized_text_list = []



for row in range(0, len_rows):

    

    # Create an empty list containing lemmatized words

    lemmatized_list = []

    

    # Save the text and its words into an object

    text = train_features.loc[row]['review_parsed_4']

    text_words = text.split(" ")



    # Iterate through every word to lemmatize

    for word in text_words:

        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v")) #v=verb

        

    # Join the list

    lemmatized_text = " ".join(lemmatized_list)

    

    # Append to the list containing the texts

    lemmatized_text_list.append(lemmatized_text)

    
train_features['review_parsed_5'] = lemmatized_text_list
#stop words

nltk.download('stopwords')
#remove the stop words, (helping of a regular expression only detecting whole words)

stop_words_list = list(stopwords.words('english'))

#stop_words[:10]



train_features['review_parsed_6'] = train_features['review_parsed_5']



for stop_word in stop_words_list:



    regex_stopword = r"\b" + stop_word + r"\b"

    train_features['review_parsed_6'] = train_features['review_parsed_6'].str.replace(regex_stopword, '')
train_features.loc[5]['review']
train_features.loc[5]['review_parsed_1']
#train_features.loc[5]['review_parsed_2']
#train_features.loc[5]['review_parsed_3']
#train_features.loc[5]['review_parsed_4']
#train_features.loc[5]['review_parsed_5']
train_features.loc[5]['review_parsed_6']
train_features.head()
# remove unnecessary parsed columns from the dataframe 

list_columns = ["condition", "review", "review_parsed_6", "review_length"]

train_features = train_features[list_columns]



train_features = train_features.rename(columns={'review_parsed_6': 'review_parsed'})
train_features
#Word count in each review

train_features['count_word']=train_features["review_parsed"].apply(lambda x: len(str(x).split()))



#Unique word count 

train_features['count_unique_word']=train_features["review_parsed"].apply(lambda x: len(set(str(x).split())))



#Letter count

train_features['count_letters']=train_features["review_parsed"].apply(lambda x: len(str(x)))



#punctuation count

train_features["count_punctuations"] = train_features["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



#upper case words count

train_features["count_words_upper"] = train_features["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



#title case words count

train_features["count_words_title"] = train_features["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



#numbers count

train_features["count_numbers"] = train_features["review"].apply(lambda x: len([w for w in str(x).split() if w.isnumeric()]))



#Number of stopwords

train_features["count_stopwords"] = train_features["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words_list]))



#Average length of the words

train_features["mean_word_len"] = train_features["review_parsed"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_features.head()
# train_features

with open('train_features.pkl', 'wb') as output:

    pickle.dump(train_features, output)

    
# Correlation Heatmap of the features



train_corr = train_features.corr()



# correlation matrix

sns.set(font_scale=1)

plt.figure(figsize=(14,10))

sns.heatmap(train_corr, annot=True, fmt=".4f",vmin=-1, vmax=1, linewidths=.5, cmap = sns.color_palette("BrBG", 100))

#plt.yticks(rotation=0)

plt.show()
#average words count&condition relationship



sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.barplot(x="condition", y="count_word", data=train_features, palette=sns.light_palette((210, 90, 60), input="husl"))

plt.xticks(rotation=90)

plt.title("condition vs words count", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("words count", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("condition", {"fontname": "serif", "fontweight":"bold"})
#average titles(possible drug names) count&condition relationship



sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))

ax = sns.barplot(x="condition", y="count_words_title", data=train_features, palette=sns.light_palette("navy", reverse=True))

plt.xticks(rotation=90)

plt.title("condition vs titles count", {"fontname":"fantasy", "fontweight":"bold", "fontsize":"medium"})

plt.ylabel("titles count", {"fontname": "serif", "fontweight":"bold"})

plt.xlabel("condition", {"fontname": "serif", "fontweight":"bold"})
#encode "condition" col



le = LabelEncoder()

arr=le.fit_transform(train_features['condition'])



train_features['condition']=arr



#get pickle for le object to use "text classification" notebook

with open('le.pkl', 'wb') as output:

    pickle.dump(le, output) 
df_train, df_test = train_test_split(train_features, test_size=0.25, random_state=8)
# get pickles for train&test will be used for text representation in the next step   

with open('df_train.pkl', 'wb') as output:

    pickle.dump(df_train, output)

    

with open('df_test.pkl', 'wb') as output:

    pickle.dump(df_test, output) 