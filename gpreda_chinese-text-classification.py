import numpy as np

import pandas as pd

import os

import json

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS

from joblib import Parallel, delayed

import tqdm

import jieba

import time

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, recall_score
data_df = pd.read_csv("../input/chinese-official-daily-news-since-2016/chinese_news.csv")
print(f"Rows: {data_df.shape[0]}, Cols: {data_df.shape[1]}")
data_df.head()
print(f"Samples with content null: {data_df.loc[data_df['content'].isnull()].shape[0]}")
print(f"Samples with headline null: {data_df.loc[data_df['headline'].isnull()].shape[0]}")
data_df = data_df.loc[~data_df['content'].isnull()]
print(f"New data shape: {data_df.shape}")
!wget https://github.com/adobe-fonts/source-han-sans/raw/release/SubsetOTF/SourceHanSansCN.zip

!unzip -j "SourceHanSansCN.zip" "SourceHanSansCN/SourceHanSansCN-Regular.otf" -d "."

!rm SourceHanSansCN.zip

!ls
import matplotlib.font_manager as fm

font_path = './SourceHanSansCN-Regular.otf'

prop = fm.FontProperties(fname=font_path)
def plot_count(feature, title, df, font_prop=prop, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop);

    plt.show()    
plot_count('tag', 'tag (all data)', font_prop=prop, df=data_df,size=1.2)
data_df['datetime'] = data_df['date'].apply(lambda x: pd.to_datetime(x))

data_df['year'] = data_df['datetime'].dt.year

data_df['month'] = data_df['datetime'].dt.month

data_df['dayofweek'] = data_df['datetime'].dt.dayofweek
def jieba_cut(x, sep=' '):

    return sep.join(jieba.cut(x, cut_all=False))



print('raw', data_df['headline'][0])

print('cut', jieba_cut(data_df['headline'][0], ', '))
%%time

data_df['headline_cut'] = Parallel(n_jobs=4)(

    delayed(jieba_cut)(x) for x in tqdm.tqdm_notebook(data_df['headline'].values)

)
%%time

data_df['content_cut'] = Parallel(n_jobs=4)(

    delayed(jieba_cut)(x) for x in tqdm.tqdm_notebook(data_df['content'].values)

)
prop = fm.FontProperties(fname=font_path, size=20)


stopwords = set(STOPWORDS)



def show_wordcloud(data, font_path=font_path, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        font_path=font_path,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        prop = fm.FontProperties(fname=font_path)

        fig.suptitle(title, fontsize=40, fontproperties=prop)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(data_df['headline_cut'], font_path, title = 'Prevalent words in headline, all data')
data_df.tag.unique()
data_tag_df = data_df.loc[data_df.tag=='详细全文']

show_wordcloud(data_tag_df['headline_cut'], font_path, title = 'Prevalent words in headline, tag=详细全文')
data_tag_df = data_df.loc[data_df.tag=='国内']

show_wordcloud(data_tag_df['headline_cut'], font_path, title = 'Prevalent words in headline, tag=国内')
data_tag_df = data_df.loc[data_df.tag=='国际']

show_wordcloud(data_tag_df['headline_cut'], font_path, title = 'Prevalent words in headline, tag=国际')
train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state = 42) 
print(f"train: {train_df.shape}, test: {test_df.shape}")
plot_count('tag', 'tag (train)', font_prop=prop, df=train_df,size=1.2)
plot_count('tag', 'tag (test)', font_prop=prop, df=test_df,size=1.2)
train_df.head()
def count_vect_feature(feature, df, max_features=5000):

    start_time = time.time()

    cv = CountVectorizer(max_features=max_features,

                             ngram_range=(1, 1),

                             stop_words='english')

    X_feature = cv.fit_transform(df[feature])

    print('Count Vectorizer `{}` completed in {} sec.'.format(feature, round(time.time() - start_time,2)))

    return X_feature, cv
X_headline, cv = count_vect_feature('headline_cut', train_df, 20000)
X_content, cv = count_vect_feature('content_cut', train_df, 30000)
target =  'tag'

X = X_content

y = train_df[target].values

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.2, random_state = 42) 
train_X.shape, valid_X.shape, train_y.shape, valid_y.shape
%%time

clf_svc = SVC(kernel='linear')

clf_svc = clf_svc.fit(train_X, train_y)
def show_confusion_matrix(valid_y, predicted, size=1, font_prop=prop, trim_labels=False):

    mat = confusion_matrix(valid_y, predicted)

    plt.figure(figsize=(4*size, 4*size))

    f, ax = plt.subplots(1,1, figsize=(4*size,4*size))

    sns.set()

    target_labels = np.unique(valid_y)

    if(trim_labels):

        target_labels = [x[0:70] for x in target_labels]

    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,

                xticklabels=target_labels,

                yticklabels=target_labels

               )

    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop);

    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop);

    plt.xlabel('true label')

    plt.ylabel('predicted label')

    plt.show()
predicted_valid = clf_svc.predict(valid_X)

prediction_acc = np.mean(predicted_valid == valid_y)

prediction_f1_score = f1_score(valid_y, predicted_valid, average='weighted')

prediction_recall = recall_score(valid_y, predicted_valid, average='weighted')

print("Valid:\n========================================================")

print(f"Feature: {target} \t| Prediction accuracy: {prediction_acc}")

print(f"Feature: {target} \t| Prediction F1-score: {prediction_f1_score}")

print(f"Feature: {target} \t| Prediction recall: {prediction_recall}")

show_confusion_matrix(valid_y, predicted_valid, font_prop=prop,size=1.5)

print(classification_report(valid_y, predicted_valid))
%%time

clf_nb = MultinomialNB(fit_prior='true')

clf_nb = clf_nb.fit(train_X, train_y)
predicted_valid = clf_nb.predict(valid_X)

prediction_acc = np.mean(predicted_valid == valid_y)

prediction_f1_score = f1_score(valid_y, predicted_valid, average='weighted')

prediction_recall = recall_score(valid_y, predicted_valid, average='weighted')

print("Valid:\n========================================================")

print(f"Feature: {target} \t| Prediction accuracy: {prediction_acc}")

print(f"Feature: {target} \t| Prediction F1-score: {prediction_f1_score}")

print(f"Feature: {target} \t| Prediction recall: {prediction_recall}")

show_confusion_matrix(valid_y, predicted_valid, font_prop=prop,size=1.5)

print(classification_report(valid_y, predicted_valid))