import pandas as pd
import numpy as np

# read data
reviews_df = pd.read_csv("../input/Hotel_Reviews.csv")
# append the positive and negative text reviews
reviews_df["review"] = reviews_df["Negative_Review"] + reviews_df["Positive_Review"]
# create the label
reviews_df["is_bad_review"] = reviews_df["Reviewer_Score"].apply(lambda x: 1 if x < 5 else 0)
# select only relevant columns
reviews_df = reviews_df[["review", "is_bad_review"]]
reviews_df.head()
reviews_bad_df = reviews_df[reviews_df["is_bad_review"] == 1]
reviews_good_df = reviews_df[reviews_df["is_bad_review"] == 0]
reviews_df = reviews_good_df.sample(frac = 0.05, replace = False, random_state=42)
reviews_df = pd.concat([reviews_df, reviews_bad_df])
print(reviews_df)
# remove 'No Negative' or 'No Positive' from text
reviews_df["review"] = reviews_df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
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
# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)
# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
reviews_df.head()
reviews_df.shape
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
reviews_df[reviews_df["nb_words"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)
# lowest negative sentiment reviews (with more than 5 words)
reviews_df[reviews_df["nb_words"] >= 5].sort_values("neg", ascending = False)[["review", "neg"]].head(10)
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
print(y_train)
# train a neural network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256, 32, 8),max_iter=1000000,early_stopping=True,verbose=True,tol=0.000005,n_iter_no_change=15)
mlp.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
mlp_predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,mlp_predictions))
print(classification_report(y_test,mlp_predictions))
print(X_test.reset_index(drop=True).loc[103][X_test.reset_index(drop=True).loc[103]>0])
newDataTF = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words="english", lowercase=True, max_features=500000, vocabulary=tfidf.vocabulary_)
def PrepareInput(review):
    review = clean_text(review)
    modelInput = pd.DataFrame(sid.polarity_scores(review), index=[0])
    modelInput['nb_chars'] = len(review)
    modelInput['nb_words'] = len(review.split(" "))
    d2vVectors = [model.infer_vector(review.split(" "))]
    d2vDF = pd.DataFrame(d2vVectors, index=[0], columns=["doc2vec_vector_" + str(x) for x in range(len(d2vVectors[0]))])
    modelInput = pd.concat([modelInput, d2vDF], axis=1)
    modelInputTfidfResult = newDataTF.fit_transform([review]).toarray()
    modelInputTfidfDf = pd.DataFrame(modelInputTfidfResult, columns=newDataTF.get_feature_names())
    modelInputTfidfDf.columns = ["word_" + str(x) for x in modelInputTfidfDf.columns]
    modelInputTfidfDf.index = [0]
    modelInput = pd.concat([modelInput, modelInputTfidfDf], axis=1)
    
    return modelInput

def PredictSingle(self, review):
    modelIn = PrepareInput(review)
    modelPrediction = self.predict(modelIn)
    print("Good review" if modelPrediction[0] == 0 else "Bad review")
    return modelPrediction

mlp.__class__.PredictReview = PredictSingle

mlp.PredictReview("my expectations were extremely low, yet i was still let down. unwashed bedsheets, bad location, not to mention the fact that the breakfast bar is always missing foods and takes hours to get restocked.")

mlp.PredictReview("i loved the service i received at this hotel. on top of the kind and thoughtful staff, i saw no signs of cut corners when it came to the cleanliness. five stars!")
while True:
    inputSentence = input()
    mlp.PredictReview(inputSentence)