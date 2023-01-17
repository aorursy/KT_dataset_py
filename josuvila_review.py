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
import pandas as pd



# read data

reviews_df = pd.read_csv("../input/amazon/STEM Group.csv")

# append the positive and negative text reviews

reviews_df["review"] = reviews_df["Product_review"]

# create the label

reviews_df["is_bad_review"] = reviews_df["Product_rating"].apply(lambda x: 1 if x < 5 else 0)

# select only relevant columns

reviews_df = reviews_df[["review", "is_bad_review"]]

reviews_df.head(10)
# return the wordnet object value corresponding to the POS tag

from nltk.corpus import wordnet



def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN

    

import string

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer



def clean_text(text):

    # lower text

    text = text.lower()

    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    stop = stopwords.words('english')

    text = [x for x in text if x not in stop]

    # remove empty tokens

    text = [t for t in text if len(t) > 0]

    # pos tag text

    pos_tags = pos_tag(text)

    # lemmatize text

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # remove words with only one letter

    text = [t for t in text if len(t) > 1]

    # join all

    text = " ".join(text)

    return(text)



# clean text data

reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: clean_text(x))

# add sentiment anaylsis columns

from nltk.sentiment.vader import SentimentIntensityAnalyzer



sid = SentimentIntensityAnalyzer()

reviews_df["sentiments"] = reviews_df["review"].apply(lambda x: sid.polarity_scores(x))

reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)
# add number of characters column

reviews_df["nb_chars"] = reviews_df["review"].apply(lambda x: len(x))



# add number of words column

reviews_df["nb_words"] = reviews_df["review"].apply(lambda x: len(x.split(" ")))
# add tf-idfs columns

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df = 10)

tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()

tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())

tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]

tfidf_df.index = reviews_df.index

reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
# show is_bad_review distribution

reviews_df["is_bad_review"].value_counts(normalize = True)
# wordcloud function



from wordcloud import WordCloud

import matplotlib.pyplot as plt



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color = 'white',

        max_words = 200,

        max_font_size = 40, 

        scale = 3,

        random_state = 42

    ).generate(str(data))



    fig = plt.figure(1, figsize = (20, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(wordcloud)

    plt.show()

    

# print wordcloud

show_wordcloud(reviews_df["review"])
# highest positive sentiment reviews (with more than 5 words)

reviews_df[reviews_df["nb_words"] >= 3].sort_values("pos", ascending = False)[["review", "pos"]].head(60)




# lowest negative sentiment reviews (with more than 5 words)

reviews_df[reviews_df["nb_words"] >= 3].sort_values("neg", ascending = False)[["review", "neg"]].head(60)

reviews_df[reviews_df["nb_words"] >= 3].sort_values("neg", ascending = False)[["review", "neg"]].to_csv('results.csv', header = True)
# plot sentiment distribution for positive and negative reviews



import seaborn as sns



for x in [0, 1]:

    subset = reviews_df[reviews_df['is_bad_review'] == x]

    

    # Draw the density plot

    if x == 0:

        label = "Good reviews"

    else:

        label = "Bad reviews"

    sns.distplot(subset['compound'], hist = False, label = label)
# feature selection

label = "is_bad_review"

ignore_cols = [label, "review", "review_clean"]

features = [c for c in reviews_df.columns if c not in ignore_cols]



# split the data into train and test

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df[label], test_size = 0.20, random_state = 42)
# train a random forest classifier

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

rf.fit(X_train, y_train)



# show feature importance

feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)

feature_importances_df.head(20)
# ROC curve



from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt



y_pred = [x[1] for x in rf.predict_proba(X_test)]

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)



roc_auc = auc(fpr, tpr)



plt.figure(1, figsize = (15, 10))

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()