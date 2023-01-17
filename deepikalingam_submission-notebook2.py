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
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

import seaborn as sns



import re, string
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.groupby("target")["id"].nunique()
sns.countplot(train_df.target).set_title('Target variable distribution in train set')
def get_num_words(data):

    return len(data.split())

train_df['num_chars'] = train_df.text.apply(len)

train_df['num_words'] = train_df.text.apply(get_num_words)

test_df['num_chars'] = test_df.text.apply(len)

test_df['num_words'] = test_df.text.apply(get_num_words)

sns.boxplot(x='target', y='num_chars', data=train_df[['num_chars', 'target']]).set_title('Number of characters')

sns.boxplot(x='target', y='num_words', data=train_df[['num_words', 'target']]).set_title('Number of words')
X_train = train_df["text"].copy()

X_test = test_df["text"].copy()

y_train = train_df["target"].copy()



txt_vectorizer = TfidfVectorizer()

X_train_tf = txt_vectorizer.fit_transform(X_train)



X_test_tf = txt_vectorizer.transform(X_test)





multinominalNB_clf = MultinomialNB().fit(X_train_tf, y_train)



test_df.loc[:,"target"] = multinominalNB_clf.predict(X_test_tf)
predProbGivenText_df = pd.DataFrame(multinominalNB_clf.predict_proba(X_test_tf))
uniq_keywords = train_df["keyword"].unique()[1:]

print(len(uniq_keywords))

print(uniq_keywords)
def replace_keywords(df_og):

    df = df_og.copy()

    df["keyword"] = df["keyword"].replace("ablaze","blaze")

    df["keyword"] = df["keyword"].replace("blazing","blaze")

    df["keyword"] = df["keyword"].replace("annihilated","annihilation")

    df["keyword"] = df["keyword"].replace("attacked","attack")

    df["keyword"] = df["keyword"].replace("bioterror","bioterrorism")

    df["keyword"] = df["keyword"].replace("blown%20up","blew%20up")

    df["keyword"] = df["keyword"].replace("bloody","blood")

    df["keyword"] = df["keyword"].replace("bleeding","blood")

    df["keyword"] = df["keyword"].replace("body%20bags","body%20bag")

    df["keyword"] = df["keyword"].replace("body%20bagging","body%20bag")

    df["keyword"] = df["keyword"].replace("bombed","bomb")

    df["keyword"] = df["keyword"].replace("bombing","bomb")

    df["keyword"] = df["keyword"].replace("burning%20buildings","buildings%20burning")

    df["keyword"] = df["keyword"].replace("buildings%20on%20fire","buildings%20burning")

    df["keyword"] = df["keyword"].replace("burned","burning")

    df["keyword"] = df["keyword"].replace("casualties","casualty")

    df["keyword"] = df["keyword"].replace("catastrophe","catastrophic")

    df["keyword"] = df["keyword"].replace("collapse","collapsed")

    df["keyword"] = df["keyword"].replace("collide","collision")

    df["keyword"] = df["keyword"].replace("collided","collision")

    df["keyword"] = df["keyword"].replace("crash","crashed")

    df["keyword"] = df["keyword"].replace("crush","crushed")

    df["keyword"] = df["keyword"].replace("dead","death")

    df["keyword"] = df["keyword"].replace("deaths","death")

    df["keyword"] = df["keyword"].replace("deluge","deluged")

    df["keyword"] = df["keyword"].replace("demolished","demolish")

    df["keyword"] = df["keyword"].replace("demolition","demolish")

    df["keyword"] = df["keyword"].replace("derailment","derail")

    df["keyword"] = df["keyword"].replace("derailed","derail")

    df["keyword"] = df["keyword"].replace("desolation","desolate")

    df["keyword"] = df["keyword"].replace("destroyed","destroy")

    df["keyword"] = df["keyword"].replace("destruction","destroy")

    df["keyword"] = df["keyword"].replace("detonate","detonation")

    df["keyword"] = df["keyword"].replace("devastated","devastation")

    df["keyword"] = df["keyword"].replace("drowned","drown")

    df["keyword"] = df["keyword"].replace("drowning","drown")

    df["keyword"] = df["keyword"].replace("electrocute","electrocuted")

    df["keyword"] = df["keyword"].replace("evacuated","evacuate")

    df["keyword"] = df["keyword"].replace("evacuation","evacuate")

    df["keyword"] = df["keyword"].replace("explode","explosion")

    df["keyword"] = df["keyword"].replace("exploded","explosion")

    df["keyword"] = df["keyword"].replace("fatality","fatalities")

    df["keyword"] = df["keyword"].replace("floods","flood")

    df["keyword"] = df["keyword"].replace("flooding","flood")

    df["keyword"] = df["keyword"].replace("bush%20fires","forest%20fire")

    df["keyword"] = df["keyword"].replace("forest%20fires","forest%20fire")

    df["keyword"] = df["keyword"].replace("hailstorm","hail")

    df["keyword"] = df["keyword"].replace("hazardous","hazard")

    df["keyword"] = df["keyword"].replace("hijacking","hijack")

    df["keyword"] = df["keyword"].replace("hijacker","hijack")

    df["keyword"] = df["keyword"].replace("hostages","hostage")

    df["keyword"] = df["keyword"].replace("injured","injury")

    df["keyword"] = df["keyword"].replace("injures","injury")

    df["keyword"] = df["keyword"].replace("inundated","inundation")

    df["keyword"] = df["keyword"].replace("mass%20murderer","mass%20murder")

    df["keyword"] = df["keyword"].replace("obliterated","obliterate")

    df["keyword"] = df["keyword"].replace("obliteration","obliterate")

    df["keyword"] = df["keyword"].replace("panicking","panic")

    df["keyword"] = df["keyword"].replace("quarantined","quarantine")

    df["keyword"] = df["keyword"].replace("rescuers","rescue")

    df["keyword"] = df["keyword"].replace("rescued","rescue")

    df["keyword"] = df["keyword"].replace("rioting","riot")

    df["keyword"] = df["keyword"].replace("dust%20storm","sandstorm")

    df["keyword"] = df["keyword"].replace("screamed","screams")

    df["keyword"] = df["keyword"].replace("screaming","screams")

    df["keyword"] = df["keyword"].replace("sirens","siren")

    df["keyword"] = df["keyword"].replace("suicide%20bomb","suicide%20bomber")

    df["keyword"] = df["keyword"].replace("suicide%20bombing","suicide%20bomber")

    df["keyword"] = df["keyword"].replace("survived","survive")

    df["keyword"] = df["keyword"].replace("survivors","survive")

    df["keyword"] = df["keyword"].replace("terrorism","terrorist")

    df["keyword"] = df["keyword"].replace("thunderstorm","thunder")

    df["keyword"] = df["keyword"].replace("traumatised","trauma")

    df["keyword"] = df["keyword"].replace("twister","tornado")

    df["keyword"] = df["keyword"].replace("typhoon","hurricane")

    df["keyword"] = df["keyword"].replace("weapons","weapon")

    df["keyword"] = df["keyword"].replace("wild%20fires","wildfire")

    df["keyword"] = df["keyword"].replace("wounded","wounds")

    df["keyword"] = df["keyword"].replace("wrecked","wreckage")

    df["keyword"] = df["keyword"].replace("wreck","wreckage")

    return(df)
train_df = replace_keywords(train_df)

test_df = replace_keywords(test_df)
uniq_keywords = train_df["keyword"].unique()[1:]

kword_resArr = []

print(len(uniq_keywords))

for kword in uniq_keywords:

    kword_df = train_df.loc[train_df["keyword"] == kword,: ]

    total_kword = float(len(kword_df))

    target0_n = float(len(kword_df.loc[kword_df["target"]==0,:]))

    target1_n = float(len(kword_df.loc[kword_df["target"]==1,:]))

    kword_prob_df = pd.DataFrame({'keyword':[kword],

                                 "keywordPred0": [target0_n/total_kword],

                                 "keywordPred1": [target1_n/total_kword]})

    kword_resArr.append(kword_prob_df)

predProbGivenKeyWord_df= pd.concat(kword_resArr)

predProbGivenKeyWord_df.head()
test_df["textprob0"]=predProbGivenText_df.loc[:,0].copy()

test_df["textprob1"]=predProbGivenText_df.loc[:,1].copy()

test_df.head()
test_df = test_df.merge(predProbGivenKeyWord_df, how='left', on="keyword")

test_df["keywordPred0"]=test_df["keywordPred0"].fillna(0.5)

test_df["keywordPred1"]=test_df["keywordPred1"].fillna(0.5)

test_df["pred0"]=test_df["textprob0"]*test_df["keywordPred0"]

test_df["pred1"]=test_df["textprob1"]*test_df["keywordPred1"]

test_df["target"]=test_df["pred1"]>test_df["pred0"]

test_df["target"] = test_df["target"].astype(np.int)
submission_df = test_df.loc[:,["id","target"]].copy()

submission_df.head()
submission_df.to_csv('submission.csv', index = False)