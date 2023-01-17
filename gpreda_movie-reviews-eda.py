import numpy as np

import pandas as pd

import os

import time

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
os.listdir("../input")
train_df = pd.read_csv("../input/movie-reviews-classification/train.csv")

test_df = pd.read_csv("../input/movie-reviews-classification/test.csv")
train_df.head()
test_df.head()
print(f"Datase shape: train: {train_df.shape}; test: {test_df.shape}")
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(train_df)
missing_data(test_df)
def plot_features_distribution(features, title, df, isLog=False):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        if(isLog):

            sns.distplot(np.log1p(df[feature]),kde=True,hist=False, bins=120, label=feature)

        else:

            sns.distplot(df[feature],kde=True,hist=False, bins=120, label=feature)

    plt.xlabel('')

    plt.legend()

    plt.show()



def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show() 



stopwords = set(STOPWORDS)



def show_wordcloud(feature,df):

    data = df.loc[~df[feature].isnull(), feature].values

    count = (~df[feature].isnull()).sum()

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    fig.suptitle("Prevalent words in {} ({} rows)".format(feature,count), fontsize=20)

    fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



def show_confusion_matrix(valid_y, predicted, size=1, trim_labels=False):

    mat = confusion_matrix(valid_y, predicted)

    plt.figure(figsize=(4*size, 4*size))

    sns.set()

    target_labels = np.unique(valid_y)

    if(trim_labels):

        target_labels = [x[0:70] for x in target_labels]

    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,

                xticklabels=target_labels,

                yticklabels=target_labels

               )

    plt.xlabel('true label')

    plt.ylabel('predicted label')

    plt.show()
show_wordcloud('primary_title', train_df)
show_wordcloud('primary_title', test_df)
show_wordcloud('original_title', train_df)
show_wordcloud('original_title', test_df)
show_wordcloud('text', train_df)
show_wordcloud('text', test_df)
plot_features_distribution(['start_year', 'end_year'], 'Running years distribution (train)', train_df)
plot_features_distribution(['start_year', 'end_year'], 'Running years distribution (test)', test_df)
plot_features_distribution(['runtime_minutes'], 'Runtime minutes distribution (train)', test_df)
plot_features_distribution(['runtime_minutes'], 'Runtime minutes distribution (test)', test_df)
print(f"Movies in train: {train_df.title_id.nunique()} and test: {test_df.title_id.nunique()}")

l1 = set(train_df.title_id.unique())

l2 = set(test_df.title_id.unique())

card_int = len(l1.intersection(l2))

print(f"Common movies in train & test: {card_int}")
plot_count('is_adult', 'Adult movie (train)', train_df, size=1)
plot_count('is_adult', 'Adult movie (test)', test_df, size=1)
def count_vect_feature(feature, df, max_features=5000):

    start_time = time.time()

    cv = CountVectorizer(max_features=max_features,

                             ngram_range=(1, 3),

                             stop_words='english')

    X_feature = cv.fit_transform(df[feature])

    print('Count Vectorizer `{}` completed in {} sec.'.format(feature, round(time.time() - start_time,2)))

    return X_feature

data = pd.concat([train_df, test_df])
data.shape
X_feature = count_vect_feature('text', data, max_features=5000)
X = X_feature[0:train_df.shape[0]]

X_test = X_feature[train_df.shape[0]:]

y = train_df.polarity.values

print(f"X: {X.shape} test X: {X_test.shape} y: {y.shape}")

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.3, random_state = 42) 

print(f"train X: {train_X.shape}, valid X: {valid_X.shape}, train y: {train_y.shape}, valid y: {valid_y.shape}")
clf = MultinomialNB(fit_prior=True)
clf.fit(train_X, train_y)
predicted_valid = clf.predict(valid_X)

show_confusion_matrix(valid_y, predicted_valid, size=1)

print(classification_report(valid_y, predicted_valid))
print(f"ROC-AUC: {roc_auc_score(predicted_valid, valid_y)}")
predict_test = clf.predict(X_test)
submission = pd.read_csv('../input/movie-reviews-classification/sampleSubmission.csv')

submission['polarity'] = predict_test

submission.to_csv('submission.csv', index=False)