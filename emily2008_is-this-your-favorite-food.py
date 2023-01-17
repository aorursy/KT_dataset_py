import json

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")



def load_data(filename):

    with open(filename, 'r') as f:

        data = json.load(f)

    df = pd.DataFrame(data)    

    return df



train=load_data("../input/train.json")

test=load_data("../input/test.json")



train.sample(5)
train.info()
import re

from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

tqdm.pandas()



lemmatizer = WordNetLemmatizer()

def preprocess(ingredients):

    ingredients_text=" ".join(ingredients)

    ingredients_text=ingredients_text.lower()

    ingredients_text=ingredients_text.replace("-",'')

    ingredients_text=ingredients_text.replace("'",'')

    sentence=[]

    for word in ingredients_text.split():

        if re.findall('[0-9]',word): continue

        if len(word)<=2: continue

        word=lemmatizer.lemmatize(word)

        sentence.append(word)

    return " ".join(sentence)

                

train['ingredients_text']=train['ingredients'].progress_apply(lambda x: preprocess(x))

train=train.drop(['ingredients'],axis=1)

train["ingredients_len"]=train['ingredients_text'].apply(lambda x: len(x.split()))



test['ingredients_text']=test['ingredients'].progress_apply(lambda x: preprocess(x))

test=test.drop(['ingredients'],axis=1)

test["ingredients_len"]=test['ingredients_text'].apply(lambda x: len(x.split()))



train.sample(5)
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,6))

train.groupby('cuisine')['id'].count().plot.bar()

plt.show()
fig=plt.figure(figsize=(8,6))

train.groupby('cuisine')['ingredients_len'].mean().plot.bar()

plt.show()
from wordcloud import WordCloud

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))



def cloud_words(cuisine):

    text = ' '.join(train[train['cuisine']==cuisine]['ingredients_text'].values)

    wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',

                          width=1200, height=1000).generate(text)

    plt.figure(figsize=(12, 8))

    plt.imshow(wordcloud)

    plt.title('Top ingredients in '+ cuisine+' food')

    plt.axis("off")

    plt.show()

    

cloud_words("chinese")

cloud_words("italian")
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import FunctionTransformer, LabelEncoder



vectorizer=make_pipeline(

                        TfidfVectorizer(),

                        FunctionTransformer(lambda x: x.astype('float16'), validate=False)

                        )

X_train=vectorizer.fit_transform(train['ingredients_text'].values)

X_test=vectorizer.transform(test['ingredients_text'].values)



X_train.shape
LabelEncoder=LabelEncoder()

y_train=LabelEncoder.fit_transform(train['cuisine'].values)

label_dict=dict(zip(LabelEncoder.classes_,LabelEncoder.transform(LabelEncoder.classes_)))

print(label_dict)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score,cross_validate

import seaborn as sns



#Initiallly explore the models 

def model_explore(X, y):

    models = [

        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

        SVC(),

        LinearSVC(),

        MultinomialNB(),

        LogisticRegression(random_state=0),

        SGDClassifier(max_iter=200),

    ]

    CV = 5

    cv_df = pd.DataFrame(index=range(CV * len(models)))

    entries = []

    for model in models:

        model_name = model.__class__.__name__

        accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)

        for fold_idx, accuracy in enumerate(accuracies):

            entries.append((model_name, fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    sns.boxplot(x='model_name', y='accuracy', data=cv_df)

    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

                  size=5, jitter=True, edgecolor="gray", linewidth=2)

    plt.show()

    return cv_df
cv_baseline=model_explore(X_train, y_train)

cv_baseline.groupby("model_name").accuracy.mean()
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_val_score,cross_validate





classifier_SVC = SVC(C=200, # penalty parameter

                 kernel='rbf', # kernel type, rbf working fine here

                 degree=3, # default value

                 gamma=1, # kernel coefficient

                 coef0=1, # change to 1 from default value of 0.0

                 shrinking=True, # using shrinking heuristics

                 tol=0.001, # stopping criterion tolerance 

                 probability=False, # no need to enable probability estimates

                 cache_size=1000, # 200 MB cache size

                 class_weight=None, # all classes are treated equally 

                 verbose=False, # print the logs 

                 max_iter=-1, # no limit, let it run

                 decision_function_shape=None, # will use one vs rest explicitly 

                 random_state=None)



model_SVC_OVR = OneVsRestClassifier(classifier_SVC)



scores_SVC = cross_validate(model_SVC_OVR, X_train, y_train, cv=3)



print("The test accuracy of SVC is {}".format(scores_SVC['test_score'].mean()))
SGD=SGDClassifier(max_iter=1000)

model_SGD_OVR = OneVsRestClassifier(SGD)

scores_SGD = cross_validate(model_SGD_OVR, X_train, y_train, cv=3)



print("The test accuracy of SGD is {}".format(scores_SGD['test_score'].mean()))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score,cross_validate



Linear_SVC=LinearSVC()

model_LinearSVC_OVR = OneVsRestClassifier(Linear_SVC)

scores_LinearSVC = cross_validate(model_LinearSVC_OVR, X_train, y_train, cv=3)



print("The test accuracy of LinearSVC is {}".format(scores_LinearSVC['test_score'].mean()))
LogisticRegression=LogisticRegression(random_state=2019)

model_LogisticRegression_OVR = OneVsRestClassifier(LogisticRegression)

scores_LogisticRegression = cross_validate(model_LogisticRegression_OVR, X_train, y_train, cv=3)



print("The test accuracy of LogisticRegression is {}".format(scores_LogisticRegression['test_score'].mean()))
model_SVC_OVR.fit(X_train,y_train)

y_pred=model_SVC_OVR.predict(X_test)



y_pred=LabelEncoder.inverse_transform(y_pred)

test['cuisine']=y_pred

test[['id','cuisine']].to_csv("submission.csv",index=False)
