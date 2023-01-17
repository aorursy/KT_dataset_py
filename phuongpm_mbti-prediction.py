# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re

import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold

from xgboost import XGBClassifier,plot_importance

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.feature_selection import SelectFromModel

from itertools import compress





from pylab import rcParams

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.misc import imread

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/mbti_1.csv')
df.head()
dist = df['type'].value_counts()

dist
dist.index
plt.hlines(y=list(range(16)), xmin=0, xmax=dist, color='skyblue')

plt.plot(dist, list(range(16)), "D")

# plt.stem(dist)

plt.yticks(list(range(16)), dist.index)

plt.show()
df['seperated_post'] = df['posts'].apply(lambda x: x.strip().split("|||"))

df['num_post'] = df['seperated_post'].apply(lambda x: len(x))
df.head()
num_post_df = df.groupby('type')['num_post'].apply(list).reset_index()
rcParams['figure.figsize'] = 10,5

sns.violinplot(x='type',y='num_post',data=df)

plt.xlabel('')

plt.ylabel('Number of posts')
def count_youtube(posts):

    count = 0

    for p in posts:

        if 'youtube' in p:

            count += 1

    return count

        

df['youtube'] = df['seperated_post'].apply(count_youtube)
sns.violinplot(x='type',y='youtube',data=df)

plt.xlabel('')

plt.ylabel('Number of posts which mention Youtube')

plt.show()
plt.hist(df['youtube'])

plt.title('Distribution of number of posts containing Youtube across individuals')

plt.show()
df['seperated_post'][1]
# Before expanding the dataframe, give everyone an unique ID?

df['id'] = df.index
len(df)
expanded_df = pd.DataFrame(df['seperated_post'].tolist(), index=df['id']).stack().reset_index(level=1, drop=True).reset_index(name='idposts')
expanded_df.head()
expanded_df=expanded_df.join(df.set_index('id'), on='id', how = 'left')
expanded_df=expanded_df.drop(columns=['posts','seperated_post','num_post','youtube'])
def clean_text(text):

    result = re.sub(r'http[^\s]*', '',text)

    result = re.sub('[0-9]+','', result).lower()

    result = re.sub('@[a-z0-9]+', 'user', result)

    return re.sub('[%s]*' % string.punctuation, '',result)

    
final_df = expanded_df.copy()
final_df['idposts'] = final_df['idposts'].apply(clean_text)
final_df.head()
cleaned_df = final_df.groupby('id')['idposts'].apply(list).reset_index()
cleaned_df.head()
df['clean_post'] = cleaned_df['idposts'].apply(lambda x: ' '.join(x))
df.head()
vectorizer = CountVectorizer(stop_words = ['and','the','to','of',

                                           'infj','entp','intp','intj',

                                           'entj','enfj','infp','enfp',

                                           'isfp','istp','isfj','istj',

                                           'estp','esfp','estj','esfj',

                                           'infjs','entps','intps','intjs',

                                           'entjs','enfjs','infps','enfps',

                                           'isfps','istps','isfjs','istjs',

                                           'estps','esfps','estjs','esfjs'],

                            max_features=1500,

                            analyzer="word",

                            max_df=0.8,

                            min_df=0.1)
corpus = df['clean_post'].values.reshape(1,-1).tolist()[0]

vectorizer.fit(corpus)

X_cnt = vectorizer.fit_transform(corpus)
X_cnt
# Transform the count matrix to a tf-idf representation

tfizer = TfidfTransformer()

tfizer.fit(X_cnt)

X = tfizer.fit_transform(X_cnt).toarray()
X.shape
all_words = vectorizer.get_feature_names()

n_words = len(all_words)
df['fav_world'] = df['type'].apply(lambda x: 1 if x[0] == 'E' else 0)

df['info'] = df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)

df['decision'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)

df['structure'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
df.head()
X_df = pd.DataFrame.from_dict({w: X[:, i] for i, w in enumerate(all_words)})
def sub_classifier(keyword):

    y_f = df[keyword].values

    X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_df, y_f, stratify=y_f)

    f_classifier = XGBClassifier()

    print(">>> Train classifier ... ")

    f_classifier.fit(X_f_train, y_f_train, 

                     early_stopping_rounds = 10, 

                     eval_metric="logloss", 

                     eval_set=[(X_f_test, y_f_test)], verbose=False)

    print(">>> Finish training")

    print("%s:" % keyword, sum(y_f)/len(y_f))

    print("Accuracy %s" % keyword, accuracy_score(y_f_test, f_classifier.predict(X_f_test)))

    print("AUC %s" % keyword, roc_auc_score(y_f_test, f_classifier.predict_proba(X_f_test)[:,1]))

    return f_classifier
fav_classifier = sub_classifier('fav_world')
info_classifier = sub_classifier('info')
decision_classifier = sub_classifier('decision')
str_classifier = sub_classifier('structure')
rcParams['figure.figsize'] = 20, 10

plt.subplots_adjust(wspace = 0.5)

ax1 = plt.subplot(1, 4, 1)

plt.pie([sum(df['fav_world']), 

         len(df['fav_world']) - sum(df['fav_world'])], 

        labels = ['Extrovert', 'Introvert'],

        explode = (0, 0.1),

       autopct='%1.1f%%')



ax2 = plt.subplot(1, 4, 2)

plt.pie([sum(df['info']), 

         len(df['info']) - sum(df['info'])], 

        labels = ['Sensing', 'Intuition'],

        explode = (0, 0.1),

       autopct='%1.1f%%')



ax3 = plt.subplot(1, 4, 3)

plt.pie([sum(df['decision']), 

         len(df['decision']) - sum(df['decision'])], 

        labels = ['Thinking', 'Feeling'],

        explode = (0, 0.1),

       autopct='%1.1f%%')



ax4 = plt.subplot(1, 4, 4)

plt.pie([sum(df['structure']), 

         len(df['structure']) - sum(df['structure'])], 

        labels = ['Judging', 'Perceiving'],

        explode = (0, 0.1),

       autopct='%1.1f%%')



plt.show()
# Get the default params for the current E/I classifier

fav_classifier.get_xgb_params()
# set up parameters grids

params = {

        'min_child_weight': [1, 5],

        'gamma': [0.5, 1, 1.5, 2],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 5, 7]

        }
# xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

#                     silent=True, nthread=1)
# keyword = 'fav_world'

# y = df[keyword].values

# folds = 3

# param_comb = 5



# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=1, cv=skf.split(X,y), verbose=3, random_state=1001 )
# random_search.fit(X, y)
# subsample=0.6, min_child_weight=1, max_depth=5, gamma=0.5, colsample_bytree=1.0 is the best hyperparameters in the search

plot_importance(fav_classifier, max_num_features = 20)

plt.title("Features associated with Extrovert")

plt.show()
plot_importance(info_classifier, max_num_features = 20)

plt.title("Features associated with Sensing")

plt.show()
plot_importance(decision_classifier, max_num_features=20)

plt.title("Features associated with Thinking")

plt.show()
plot_importance(str_classifier, max_num_features=20)

plt.title("Features associated with Judging")

plt.show()
# Start with one review:

def generate_wordcloud(text, title):

    # Create and generate a word cloud image:

    wordcloud = WordCloud(background_color="white").generate(text)



    # Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(title, fontsize = 40)

    plt.show()
df_by_personality = df.groupby("type")['clean_post'].apply(' '.join).reset_index()
df_by_personality.head()
for i, t in enumerate(df_by_personality['type']):

    text = df_by_personality.iloc[i,1]

    generate_wordcloud(text, t)
test_string = 'I like to observe, think, and analyze to find cons and pros. Based on my analysis, I like to create a solution based on cost effective analysis to maximize the resource to improve the performance. I like talking to my friends. I like to read and learn. I simulate a lot of different situations to see how I would react. I read or watch a lot to improve myself. I love talking to them and seeing what they have been up to. I have a variety of friends, and I appreciate they all experience different things. Listening to their emotion, experience, and life is always great.'.lower()

final_test = tfizer.transform(vectorizer.transform([test_string])).toarray()
test_point = pd.DataFrame.from_dict({w: final_test[:, i] for i, w in enumerate(all_words)})
fav_classifier.predict_proba(test_point) #[I, E]
info_classifier.predict_proba(test_point) #[N,S]
decision_classifier.predict_proba(test_point) #[F,T]
str_classifier.predict_proba(test_point) #[P,J]