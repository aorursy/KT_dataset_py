import pandas as pd

import os

import re

import string



from nltk.tokenize import word_tokenize



from sklearn import linear_model

from sklearn import metrics

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer



from sklearn import tree

from sklearn import naive_bayes

from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier
# read the training data

df = pd.read_csv("../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv")
df.head()
# we create new column called kfold and fill it with -1

df["kfold"] = -1



# the next step is to randomize the rown of the data

df = df.sample(frac=1).reset_index(drop=True)



# fetch the value

y = df.Rating.values



# initiate the kfold class from model selection modul

kf = model_selection.StratifiedKFold(n_splits=5)



# Fill the new kfold column

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):

    df.loc[v_, 'kfold'] = f
df.head()
models = {

    "lr": linear_model.LogisticRegression(),

    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),

    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),

    "bayes": naive_bayes.MultinomialNB(),

    "XGBClassifier": XGBClassifier()

}



vectorizers = {

    "count_vectorizer": CountVectorizer(

        tokenizer = word_tokenize,

        token_pattern=None

        # do not see improvement use ngram

        #ngram_range=(1, 3)

        ),

    "tfid_vectorizer": TfidfVectorizer(

        tokenizer=word_tokenize,

        token_pattern=None

        # do not see improvement use ngram

        #ngram_range=(1, 3)

        )

}
def clean_text(s):

    """

    This function cleans the text a bit

    :param s: string

    :return: cleaned string

    """

    # Convert to lower case

    s = s.lower()



    # split by all whitespaces

    s = s.split()



    # join tokens by single space

    # why we do this?

    # this will remove all kinds of weird space # "hi. how are you" becomes

    # "hi. how are you"

    s = " ".join(s)



    # remove all punctuations using regex and string module

    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)



    # you can add more cleaning here if you want

    # and then return the cleaned string

    return s

def run(fold, model, vectorizer):

    # applying clean_text to Revies column

    df.loc[:, 'Review'] = df.Review.apply(clean_text)



    # training data is where kfold is not equal to provided fold

    # also, note that we reset the index

    df_train = df[df.kfold != fold].reset_index(drop=True)



    # validation data is where kfold is equal to provided fold

    df_test = df[df.kfold == fold].reset_index(drop=True)



    # initialize CountVectorizer with NLTK,s word_tokenize

    # function as tokenizer

    vectorizer = vectorizers[vectorizer]



    #fit count_vec on training data reviews

    vectorizer.fit(df_train.Review)



    #transform training and validation data reviews

    xtrain = vectorizer.transform(df_train['Review'])

    xtest = vectorizer.transform(df_test['Review'])



    ytrain = df_train.Rating



    # initialize model

    clf = models[model]



    #initialize hyperparameter if you want use

    # if not just give # sign in

    # clf = logreg(clf,xtrain,ytrain)



    #fit the model on training data reviews and Rating

    clf.fit(xtrain, df_train.Rating)



    # make prediction on test data

    # threshold for predictions is 0.5

    preds = clf.predict(xtest)



    #calculate accuracy

    accuracy = metrics.accuracy_score(df_test.Rating, preds)



    print(f"Model={model}")

    print(f"Vectorizer={vectorizer}")

    print(f"Fold={fold}")

    print(f"Accuracy = {accuracy}")

    print("")
run(1,"lr","tfid_vectorizer"),

run(1,"decision_tree_gini","tfid_vectorizer"),

run(1,"decision_tree_entropy","tfid_vectorizer"),

run(1,"bayes","tfid_vectorizer"),

run(1,"XGBClassifier","tfid_vectorizer")
run(1,"lr","tfid_vectorizer"),

run(1,"lr","count_vectorizer")
run(0,"lr","tfid_vectorizer"),

run(1,"lr","tfid_vectorizer"),

run(2,"lr","tfid_vectorizer"),

run(3,"lr","tfid_vectorizer"),

run(4,"lr","tfid_vectorizer"),
run(3,"lr","tfid_vectorizer")