# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

from wordcloud import WordCloud

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import precision_score, recall_score

from sklearn.linear_model import LogisticRegressionCV

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import train and test set

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
# visualise the occurances/distribution of all the class types.

num_c = train['type'].value_counts()

f, ax = plt.subplots(figsize=(12,6))

sns.barplot( num_c.values, num_c.index, palette="Blues_d")

ax.xaxis.grid(False)

ax.set(xlabel="Number of Occurrences")

ax.set(ylabel="Personality Types")

ax.set(title="Personality Occurances")

sns.despine(trim=True, left=True, bottom=True)
# drop the  'Id' colum since it's unnecessary for the prediction process.

train_type = train[['type']]

test_ID = test['id']

train.drop(['type'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)
# put all the features together to enable quick transformation process

post_features = pd.concat([train, test],sort=False).reset_index(drop=True)

post_features.head()
# using regular expressions for dealing with special patterns of noise.

pattern = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

post_features = post_features.replace(to_replace = pattern, value = ' ', regex = True)
post_features.head()
# creating a cleaned corpus

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords



lem = WordNetLemmatizer() # returns the actual root word of a text token

# ps = PorterStemmer() # cuts off the surfix but doesnt necessarily return the root word

corpus =[]

# cleaning text

for i in range(0,len(post_features['posts'])):



    post = re.sub('[^a-zA-Z]', ' ',str(post_features.iloc[i].values)) 

    post = post.lower()

    post = post.split()

    # ps.stem(word) #MethodAttempt

    stop_wrds = set(stopwords.words('english'))

    post = [lem.lemmatize(word, "v") for word in post if not word in stop_wrds]

    post = ' '.join(post)

    corpus.append(post)



post_features['posts_clean'] = corpus

post_features.drop(['posts'], axis=1, inplace=True)              
post_features.head()
# Create Ranked Statistical Features with TfidfVectorizer, best choice because insignificant words recieve a lower rank

obj = TfidfVectorizer()

post_final_features = obj.fit_transform(post_features['posts_clean'])

print(post_final_features[0])



#AnotherMethodAttempt

# Create Ranked Statistical Features with CountVectorizer has no ranking machanism

# obj = CountVectorizer() 

# post_final_features = obj.fit_transform(post_features['posts_clean']).toarray()

# print(post_final_features[0:5])
# label extraction and creating our target variables from personality types

pd.options.mode.chained_assignment = None

yc = pd.DataFrame(train_type['type'])

train_type['I-E'] = train_type['type'].astype(str).str[0]

train_type['I-E'] = train_type['I-E'].map({"I": 0, "E": 1})

train_type['S-N'] = train_type['type'].astype(str).str[1]

train_type['S-N'] = train_type['S-N'].map({"S": 0, "N": 1})

train_type['F-T'] = train_type['type'].astype(str).str[2]

train_type['F-T'] = train_type['F-T'].map({"F": 0, "T": 1})

train_type['P-J'] = train_type['type'].astype(str).str[3]

train_type['P-J'] = train_type['P-J'].map({"P": 0, "J": 1})

train_type.drop('type', axis=1, inplace=True) 

y = train_type

print(y[0:5])
y = np.array(y)

print(y[0:5])
# Spliting the data back to train(X,y) and test(X_sub)

X = post_final_features[:len(y), :]

X_final_test = post_final_features[len(y):, :]

print('Features size for train(X,y) and test(X_final_test):')

print('X', X.shape, 'y', y.shape, 'X_final_test', X_final_test.shape)
# wordcloud of the most frequently used words by each personality

yc['posts_clean'] = post_features['posts_clean']

labels = yc['type'].unique()

row, col = 5, 3

wc = WordCloud(stopwords = ['infj','entp','intp','intj', 'isfps','istps','isfjs','istjs',

                             'entjs','enfjs','infps','enfps','entj','enfj','infp','enfp',

                             'estps','esfps','estjs','esfjs','isfp','istp','isfj','istj',

                             'estp','esfp','estj','esfj','infjs','entps','intps','intjs'])

fig, ax = plt.subplots(5, 3, figsize=(20,15))

for i in range(5):

    for j in range(3):

        c_type = labels[i*col+j]

        c_ax = ax[i][j]

        df = yc[yc['type'] == c_type]

        wordc = wc.generate(df['posts_clean'].to_string())

        c_ax.imshow(wordc)

        c_ax.axis('off')

        c_ax.set_title(label=c_type,fontdict = {'fontsize': 20})
# Predicting model

model = OneVsRestClassifier(LogisticRegressionCV(Cs=30, solver = 'saga',

                                                       multi_class = 'multinomial', cv=5), n_jobs =-1)
# split the train set to create a validation set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# train model with 80% of train set

model.fit(X_train, y_train)
# predict 20% of train set

y_pred = model.predict(X_test)
# performance matrics and model eveluation using sklearn.metrics inbuilt classification metric

print(classification_report(y_test, y_pred, target_names=['Mind', 'Energy', 'Nature', 'Tactics']))


#AnotherMethodAttempt

# from sklearn.naive_bayes import MultinomialNB

# model = OneVsRestClassifier(MultinomialNB())
#AnotherMethodAttempt

# from sklearn.neighbors import KNeighborsClassifier

# model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
# Reinitialise final model

model = OneVsRestClassifier(LogisticRegressionCV(Cs=30, solver = 'saga',

                                                 multi_class = 'multinomial', cv=10), n_jobs =-1)
# train final model with full train set 

model.fit(X, y)
# pridicting the actual test set (X_final_test)

y_predicted = model.predict(X_final_test)
# Final model results

y_predicted[0:5]
# Plot the accuracy for each classifier

f, ax = plt.subplots(figsize=(20, 10))

sns.set_color_codes(palette='deep')

plt.subplots_adjust(wspace = 0.5)

ax = plt.subplot(2, 4, 1)

plt.pie([sum(y_predicted[:,0]), len(y_predicted[:,0]) - sum(y_predicted[:,0])],

        labels = ['Extraverted', 'Introverted'],explode = (0, 0.1),autopct='%1.1f%%', colors=['y','orange'])

ax.set(title="Mind")



ax = plt.subplot(2, 4, 2)

plt.pie([sum(y_predicted[:,1]), len(y_predicted[:,1]) - sum(y_predicted[:,1])], 

        labels = ['Sensing', 'Intuitive'],explode = (0, 0.1),autopct='%1.1f%%', colors=['y','orange'])

ax.set(title="Energy")



f, ax2 = plt.subplots(figsize=(20, 10))

plt.subplots_adjust(wspace = 0.5)



ax2 = plt.subplot(2, 4, 1)

plt.pie([sum(y_predicted[:,2]), len(y_predicted[:,2]) - sum(y_predicted[:,2])], 

        labels = ['Thinking', 'Feeling'],explode = (0, 0.1),autopct='%1.1f%%', colors=['salmon','y'])

ax2.set(title="Nature")



ax2 = plt.subplot(2, 4, 2)

plt.pie([sum(y_predicted[:,3]),  len(y_predicted[:,3]) - sum(y_predicted[:,3])], 

        labels = ['Judging', 'Perceiving'], explode = (0, 0.1), autopct='%1.1f%%', colors=['y','salmon'])

ax2.set(title="Tactics")

plt.show()

# format submission of the predicted classes

submission = pd.DataFrame({'id' : np.array(test_ID),'mind' : y_predicted[:,0], 

                           'energy' : y_predicted[:,1], 'nature' : y_predicted[:,2], 

                           'tactics' : y_predicted[:,3]})

print('Save submission')
# save DataFrame to csv file for submission

# submission.to_csv("new_submission.csv", index=False)