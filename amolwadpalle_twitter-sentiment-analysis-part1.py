# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import numpy as np

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from imblearn.under_sampling import NearMiss

from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve

from matplotlib import style

from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import recall_score, confusion_matrix
train = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

# remove the unwanted columns from the train and test

test.drop(['id'], axis = 1, inplace = True)

train.drop(['id'], axis = 1, inplace = True)

train.replace('[^a-zA-Z#]', ' ', inplace = True, regex = True)

test.replace('[^a-zA-Z#]', ' ', inplace = True, regex = True)

#  to lemmatize, we first have to tokenize it, meaning.. splitting the sentence to words

lemmatizer = WordNetLemmatizer()

# create a object for stopwords 

stop_words = stopwords.words('english')

# print(len(stop_words))

# add the word 'user' to the stopwords list

stop_words.append('user')

# print(len(stop_words))
train['clean_tweet'] = np.nan

test['clean_tweet'] = np.nan

train.head()
# tokenize, remove word with less than 3 chars, remove words from stopwords and lemmatize them

for i in range(len(train.index)):

    words = nltk.word_tokenize(train.iloc[i, 1])

    words = [word for word in words if len(word) > 3]

    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stop_words)]

    train.iloc[i, 2]  = ' '.join(words)

    words = nltk.word_tokenize(train.iloc[i, 2])

print(train.head())

# similar way the test data set needs to be handeled.

for i in range(len(test.index)):

    words = nltk.word_tokenize(test.iloc[i,0])

    words = [word for word in words if len(word) > 3]

    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stop_words)]

    test.iloc[i, 1]  = ' '.join(words)

    words = nltk.word_tokenize(test.iloc[i, 1])

print(test.head())
# wordclod for all the words in train



txt = " ".join(text for text in train.clean_tweet)



wordcloud = WordCloud(max_font_size = 100, max_words = 50, background_color = 'orange').generate(txt)



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
# wordclod for racist messages

bad = " ".join([text for text in train['clean_tweet'][train['label']== 1]])



wordcloud = WordCloud(max_font_size = 100, max_words = 50, background_color = 'red').generate(bad)



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
# wordclod for non racist messages

non_racist = " ".join([text for text in train['clean_tweet'][train['label']== 0]])



wordcloud = WordCloud(max_font_size = 100, max_words = 50, background_color = 'green').generate(non_racist)



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
# function for extract hashtag words and see if they contribute in any way

def hashtag_extract(x):

    hashtags = []

    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags
ht_regular = hashtag_extract(train['tweet'][train['label']== 0])

ht_racist = hashtag_extract(train['tweet'][train['label']== 1])



ht_racist



# unlisting the lists - 

Ht_regular = sum(ht_regular,[])

ht_racist = sum(ht_racist,[])



# print(ht_racist)

a = nltk.FreqDist(Ht_regular)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
neg = nltk.FreqDist(ht_racist)

neg_dataframe = pd.DataFrame({'hash': list(neg.keys()),

                             'count' : list(neg.values())

                             })

top10 = neg_dataframe.nlargest(10, 'count')

plt.figure(figsize = (12, 5))

sns.barplot(data = top10, x = 'hash', y = 'count')
# TFIDF transform



vector = TfidfVectorizer(max_features=1000, stop_words='english', lowercase = False)

tfidf = vector.fit_transform(train['clean_tweet'])

tfidf_test = vector.transform(test['clean_tweet'])



print(tfidf_test.shape)
clf = LogisticRegression()

y = train['label']



X_train, X_valid, y_train, y_valid = train_test_split(tfidf, y, train_size = 0.8, random_state = 42)

print('X Train Shape', X_train.shape)

print('y Train Shape', y_train.shape)

print('X Valid Shape', X_valid.shape)

print('y Train Shape', y_valid.shape)



clf.fit(X_train, y_train)

prediction = clf.predict(X_valid)



score = f1_score(y_valid, prediction)

print('F1 - Score using imbalanced data',score)



acc_score = accuracy_score(y_valid, prediction)

print('Accuracy Score using imbalanced data = ',acc_score)



# predict_test = clf.predict(tfidf_test)

# test['label'] = predict_test

# test.to_csv('twitter_test.csv', index = False)
train.head()

train['label'].value_counts()

# sns.distplot(train['label'])
nm = NearMiss()

X_nearmiss,y_nearmiss = NearMiss().fit_sample(tfidf, y)





X_train_nm, X_valid_nm, y_train_nm, y_valid_nm = train_test_split(X_nearmiss, y_nearmiss)

print('X Train Shape', X_train_nm.shape)

print('y Train Shape', y_train_nm.shape)

print('X Valid Shape', X_valid_nm.shape)

print('y Valid Shape', y_valid_nm.shape)
clf.fit(X_train_nm, y_train_nm)

prediction_nm = clf.predict(X_valid_nm)

f1_score_nm = f1_score(y_valid_nm, prediction_nm)

print('F1 - Score using NearMiss = ',f1_score_nm)



acc_score = accuracy_score(y_valid_nm, prediction_nm)

print('Accuracy Score using NearMiss = ',acc_score)


style.use('fivethirtyeight')

def plot_learning_curve(model, X, y):

    train_size, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.01, 1, 50), cv=10,

                                                       scoring='accuracy', n_jobs=3, verbose=1, random_state=42,

                                                      shuffle=True)

    train_scores_mean = np.mean(train_scores, axis = 1)

    train_scores_std = np.std(train_scores, axis = 1)

    test_scores_mean = np.mean(test_scores, axis = 1)

    test_scores_std = np.std(test_scores, axis = 1)

    plt.figure(figsize = (8, 4))

    plt.plot(train_size, train_scores_mean, color = 'red', label = 'Training Score')

    plt.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean+train_scores_std, color = '#DDDDDD')

    plt.fill_between(train_size, test_scores_mean - test_scores_std, test_scores_mean+test_scores_std, color = '#DDDDDD')

    plt.plot(train_size, test_scores_mean, color = 'green', label = 'CV Score')

    plt.title('Learning Curve ')

    plt.xlabel('CV Train Size')

    plt.ylabel('Accuracy')

    plt.legend(loc = 'best')

    plt.show()
plot_learning_curve(clf, X_nearmiss,y_nearmiss)
# from imblearn.over_sampling import SMOTE





sm = SMOTETomek(random_state=42)

X_sm, y_sm = sm.fit_resample(tfidf, y)



X_train_sm, X_valid_sm, y_train_sm, y_valid_sm = train_test_split(X_sm, y_sm)

print('X Train Shape', X_train_sm.shape)

print('y Train Shape', y_train_sm.shape)

print('X Valid Shape', X_valid_sm.shape)

print('y Valid Shape', y_valid_sm.shape)



clf.fit(X_train_sm,y_train_sm)

prediction_sm = clf.predict(X_valid_sm)



f1_score_sm = f1_score(y_valid_sm, prediction_sm)

print('F1 - Score using NearMiss = ',f1_score_sm)



acc_score_sm = accuracy_score(y_valid_sm, prediction_sm)

print('Accuracy Score using NearMiss = ',acc_score_sm)
plot_learning_curve(clf, X_sm, y_sm)
models = []

models.append(('LRC', LogisticRegression()))

models.append(('RFC', RandomForestClassifier()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('SVC', SVC()))

models.append(('XGB', XGBClassifier()))



f1_score_all = []

accuracy_all = []

recall_all = []


for name,model in models:



    model.fit(X_train_sm,y_train_sm)

    prediction_sm = model.predict(X_valid_sm)

    acc_score_sm = accuracy_score(y_valid_sm, prediction_sm)

    recall_sc = recall_score(y_valid_sm, prediction_sm)

#     print('Accuracy Score using NearMiss = ',acc_score_sm)    

#     f1_score_all.append(f1_sc)

    accuracy_all.append(acc_score_sm) 

    recall_all.append(recall_sc)

    cm = confusion_matrix(y_valid_sm, prediction_sm)

    print(cm)



# print(f1_score)

print(accuracy_all)

print(recall_all)
clf_names = []

for name, model in models:

    clf_names.append(name)

clf_names
df = pd.DataFrame(list(zip(clf_names, accuracy_all, recall_all)), columns = ['model', 'accuracy', 'recall'])

df
values = list(zip(accuracy_all, recall_all))

data = pd.DataFrame(values, columns=['accuracy', 'recall'])





g = sns.lineplot(data=data, palette="tab10", linewidth=2.5)

g.set(xticklabels=['kk','LRC', 'RFC', 'DTC', 'SVC', 'XGB'])
# from sklearn.model_selection import RandomizedSearchCV



# # Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 800, num = 4)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]

# max_depth.append(None)

# # Minimum number of samples required to split a node

# min_samples_split = [2, 5]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2]

# # Method of selecting samples for training each tree

# # bootstrap = [True, False]



# # Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

# #                'bootstrap': bootstrap

#               }

# print(random_grid)
# # Use the random grid to search for best hyperparameters

# # First create the base model to tune

# clf = RandomForestClassifier()

# # Random search of parameters, using 3 fold cross validation, 

# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)

# # Fit the random search model

# rf_random.fit(X_train_sm,y_train_sm)
# rf_random.best_params_
# # Create the parameter grid based on the results of random search 

# from sklearn.model_selection import GridSearchCV





# parm_grid = {'n_estimators' : [550,600,650],

#             'min_samples_split' : [4,5,6],

#             'min_samples_leaf' : [1,2],

#             'max_features' : ['sqrt'],

#             'max_depth' : [None],

#             'bootstrap' : [True]}

# clf = RandomForestClassifier()



# grid_search = GridSearchCV(estimator = clf, param_grid = parm_grid, cv = 3, n_jobs = 1, verbose = 2)



# print(grid_search)
# grid_search.fit(X_train_sm,y_train_sm)

# grid_search.best_params_