# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from wordcloud import WordCloud, STOPWORDS

from sklearn.metrics import log_loss

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer

import re

import string

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#Save the 'id' column

test_ID = test['id']



#Now drop the  'id' colum since it's unnecessary for  the prediction process.

test.drop("id", axis = 1, inplace = True)
train['type'].value_counts().plot(kind = 'bar')

plt.title('Personality Types')

plt.ylabel('Count')

plt.show()
# create subplot axes for all personality types

fig, ax = plt.subplots(len(train['type'].unique()), sharex=True, figsize=(15,10*len(train['type'].unique()))) 



# create a word cloud for each personality type

k = 0

for i in train['type'].unique():

    df_4 = train[train['type'] == i]

    wordcloud = WordCloud().generate(df_4['posts'].to_string())

    ax[k].imshow(wordcloud)

    ax[k].set_title(i)

    ax[k].axis("off")

    k+=1
ntrain = train.shape[0] # Defining ntrain

ntest = test.shape[0]#

y_train = train[['type']]

features = pd.concat((train, test), sort=False).reset_index(drop=True)

features.drop(['type'], axis=1, inplace=True)

features.head()
stop = stopwords.words('english')

def cleaner(text):

    """""

    Function that takes in messy text data and returns a list clean text data

    

    params:

        text: string, text to be cleaned 

    """""

    stemmer = PorterStemmer()                                        # groups words having the same stems

    text = text.replace('|||', ' ')                                  # replaces post separators with empty space

    text = re.sub(r'\bhttps?:\/\/.*?[\r\n]*? ', 'URL ', text, flags=re.MULTILINE)  # replace hyperlink with 'URL'

    text = text.translate(str.maketrans('', '', string.punctuation)) # removes punctuation

    text = text.translate(str.maketrans('', '', string.digits))      # removes digits

    text = text.lower().strip()                                      # convert to lower case

    final_text = []

    for w in text.split():

        if w not in stop:

            final_text.append(stemmer.stem(w.strip()))

    return ' '.join(final_text)
ctv = CountVectorizer(preprocessor=cleaner,ngram_range=(1, 2),max_df=0.8, max_features=2000)

features_ctv =  ctv.fit_transform(features['posts']) 

features_ctv.shape
count_vect_df = pd.DataFrame(features_ctv.todense(), columns=ctv.get_feature_names())

count_vect_df = features_ctv 
# mapping dicts

mind = {"I": 0, "E": 1}

energy = {"S": 0, "N": 1}

nature = {"F": 0, "T": 1}

tactics = {"P": 0, "J": 1}



# create new columns using our maps

y_train['mind'] = y_train['type'].astype(str).str[0]

y_train['mind'] = y_train['mind'].map(mind)

y_train['energy'] = y_train['type'].astype(str).str[1]

y_train['energy'] = y_train['energy'].map(energy)

y_train['nature'] = y_train['type'].astype(str).str[2]

y_train['nature'] = y_train['nature'].map(nature)

y_train['tactics'] = y_train['type'].astype(str).str[3]

y_train['tactics'] = y_train['tactics'].map(tactics)
X_train = count_vect_df[:ntrain]

X_test = count_vect_df[ntrain:]
y_mind = y_train['mind']

y_energy = y_train['energy']

y_nature = y_train['nature']

y_tactics = y_train['tactics']
models = [LogisticRegression(C=0.001, solver='warn', random_state=5), GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   random_state =5)]



categories = [y_mind, y_energy, y_nature, y_tactics]
def run_model(X, y_categories, models):

    """""

    This function runs a given list of machine learning models and returns the Mean Column-Wise Log Loss Score.

    

    params:

        X: list, array or dataframe of features

        y_categories: list of multiple binary target classifiers

        models: list of machine learning models to run

    """""

    for model in models:

        score  = 0

        for category in y_categories: 

            X_train, X_test, y_train, y_test = train_test_split(X , category, test_size=0.2, random_state=42)

            model.fit(X_train, y_train)

            score += log_loss(y_test, model.predict(X_test))

        print(model)

        print("\nMean Column-Wise Log Loss Score: {:.4f}\n".format(score/4))
run_model(X_train, categories, models)
gboost1 = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   random_state =5)

gboost2 = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   random_state =5)

gboost3 = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   random_state =5)

gboost4 = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   random_state =5)



gboost1.fit(X_train, y_mind)

gboost2.fit(X_train, y_energy)

gboost3.fit(X_train, y_nature)

gboost4.fit(X_train, y_tactics)
y_pred_mind = gboost1.predict(X_test)

y_pred_energy = gboost2.predict(X_test)

y_pred_nature = gboost3.predict(X_test)

y_pred_tactics = gboost4.predict(X_test)
submission = pd.DataFrame(data = test_ID, columns= ['id'])

submission['mind'] = y_pred_mind

submission['energy'] = y_pred_energy

submission['nature'] = y_pred_nature

submission['tactics'] = y_pred_tactics
submission.to_csv('submit.csv', index=False)