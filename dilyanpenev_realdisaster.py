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
import spacy

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import MinMaxScaler
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print(train_df.shape)

train_df.sample(5)
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print(test_df.shape)

test_df.head(5)
all_data = pd.concat([train_df, test_df], sort=False)

all_data.isnull().sum()
all_data["keyword"].nunique()
all_data["location"].nunique()
nlp = spacy.load("en_core_web_lg", disable=['parser', 'tagger', 'ner'])
def clean_keyword_list(keywords):

    new_list = [re.sub('%20', ' ', k) for k in keywords]

    return new_list

    

    

def clean_keyword(keyword):

    return re.sub('%20', ' ', keyword)

    

    

def keyword_in_tweet(keywords, tweet):

    for word in tweet.lower():

        if word in keywords:

            return word

    for k in keywords:

        if k in tweet.lower():

            return k

    return np.nan
keyword_list = all_data[all_data['keyword'].notnull()]["keyword"].unique().tolist()

keyword_list = clean_keyword_list(keyword_list)

# print(keyword_list)

all_data.loc[all_data['keyword'].notnull(), 'keyword'] = all_data[all_data['keyword'].notnull()]['keyword'].apply(lambda x: clean_keyword(x))

all_data.loc[all_data['keyword'].isnull(), 'keyword'] = all_data[all_data['keyword'].isnull()]['text'].apply(lambda x: keyword_in_tweet(keyword_list, x))

all_data['keyword'].fillna('none', inplace=True)

all_data['keyword'].isnull().sum()
all_data.loc[all_data['location'].notnull() , "location"] = True

all_data['location'].fillna(False, inplace=True)

all_data.sample(5)
all_data = pd.get_dummies(all_data, columns=['location'], drop_first=True)

all_data.sample(5)
df_mislabeled = all_data.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

df_mislabeled.index.tolist()
all_data.loc[all_data['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target'] = 0

all_data.loc[all_data['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target'] = 0

all_data.loc[all_data['text'] == 'To fight bioterrorism sir.', 'target'] = 0

all_data.loc[all_data['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target'] = 1

all_data.loc[all_data['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target'] = 1

all_data.loc[all_data['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target'] = 0

all_data.loc[all_data['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target'] = 0

all_data.loc[all_data['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target'] = 1

all_data.loc[all_data['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target'] = 1

all_data.loc[all_data['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target'] = 0

all_data.loc[all_data['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target'] = 0

all_data.loc[all_data['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target'] = 0

all_data.loc[all_data['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target'] = 0

all_data.loc[all_data['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target'] = 0

all_data.loc[all_data['text'] == "Caution: breathing may be hazardous to your health.", 'target'] = 1

all_data.loc[all_data['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target'] = 0

all_data.loc[all_data['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target'] = 0

all_data.loc[all_data['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target'] = 0
all_data[all_data["id"]==3022]['text'].values
# tweet length

all_data['tweet_length'] = all_data['text'].apply(lambda x: len(str(x).split()))

all_data.head(5)
# number of hashtags

all_data['hashtag_count'] = all_data['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

all_data.head(5)
# number of mentions

all_data['mention_count'] = all_data['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

all_data.head(5)
# number of url

all_data['url_count'] = all_data['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w]))

all_data.sample(5)
all_data[all_data['id']==6710]
def normalize(text):

    list = []

    for word in nlp(text):

        if word.is_alpha:

            if not word.is_stop:

                if word.text in nlp.vocab:

                    list.append(word.lemma_)

    result = ' '.join(list)

    return result
all_data['text'] = all_data['text']+ " " + all_data['keyword']

all_data.loc[:, 'norm_text'] = all_data['text'].apply(lambda x: normalize(x.lower()))

all_data.sample(10)
X = all_data.iloc[:7613].drop("target", axis=1)

y = all_data.iloc[:7613]["target"]

test_X = all_data.iloc[7613:].drop("target", axis=1)
count = CountVectorizer()

bow = count.fit_transform(X['norm_text'])

feature_names = count.get_feature_names()

df_bow = pd.DataFrame(bow.toarray(), columns=feature_names)

X = pd.concat([X, df_bow], axis=1)

X.shape
X = X.drop(['text', 'norm_text', 'keyword', 'id'], axis=1)

X.shape
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=8)

nb = MultinomialNB()
nb.fit(X_train, y_train)

cross_val_score(nb, X_train, y_train, cv=5)
nb.score(X_val, y_val)
bow_test = count.transform(test_X['norm_text'])

df_bow_test = pd.DataFrame(bow_test.toarray(), columns=feature_names)

test_X = pd.concat([test_X, df_bow_test], axis=1)

test_X.shape
test_X = test_X.drop(['text', 'norm_text', 'keyword', 'id'], axis=1)

test_X.shape
predictions = nb.predict(test_X)

output = pd.DataFrame({'id': test_df['id'], 'target': predictions})

output = output.astype({'target': 'int64'})

print(output.head(10))

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")