import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.shape, test_df.shape
train_df.head()
train_df.loc[:, 'keyword'].value_counts()
for i in train_df.loc[~train_df['keyword'].isnull(), 'keyword'].drop_duplicates():

    print(i)
print(train_df[train_df['target'] == 0].shape)

train_df[train_df['target'] == 1].shape
train_df.loc[~train_df['keyword'].isnull(), :]
train_df.sample(10).loc[~train_df['keyword'].isnull(), ['keyword', 'text']]
sum(train_df['keyword'].isnull()), len(train_df['keyword'].isnull())
train_df[train_df["target"] == 0]["text"].values[19], train_df.loc[pd.Series(train_df["target"] == 0), ["keyword", "location"]].values[19]
train_df[train_df["target"] == 1]["text"].values[23], train_df.loc[pd.Series(train_df["target"] == 1), ["keyword", "location"]].values[23]
 
pd.set_option('display.max_colwidth', -1)

train_df["text"].head(8)
s = train_df["text"][0:5].tolist()



count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(s)



# print(f"VECTORES to dense --> {example_train_vectors.todense()}")



print(count_vectorizer.get_feature_names(), len(count_vectorizer.get_feature_names()), example_train_vectors.todense().shape)

print(train_df["text"].iloc[2])

for i in [5, 21, 37, 47]:

    print(count_vectorizer.get_feature_names()[i])
#Palabras que aparecen un par de veces al menos

def n_tweets(n):

    return train_df["text"].tolist()[0:n]

n_tuit = 0

print(f"WORDS -->  {count_vectorizer.get_feature_names()}")

for tweet in n_tweets(5):

    print(f" Dense Vector --> {example_train_vectors[n_tuit].todense()}")

    print(f"TWEET {tweet}")

    for i in np.where(example_train_vectors.todense()[n_tuit] >= 2)[1]:

        print(f"Words that appear more than 1 time --> {count_vectorizer.get_feature_names()[i]}")

    n_tuit += 1

    print(" ")
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

clf = linear_model.RidgeClassifier(tol=1e-2, solver="sag")

clf
train_vectors, train_df["target"]
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
scores.mean()
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
from sklearn.svm import SVC

clf1 = SVC(gamma='auto', kernel = 'poly')



scores = model_selection.cross_val_score(clf1, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf1.fit(train_vectors, train_df["target"])







sample_submission["target2"] = clf1.predict(test_vectors)
sample_submission[sample_submission['target2'] != 0]
train_vectors.todense()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)

clf



scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
def evaluate_model_list(lista_modelos):

    for modelo in lista_modelos:

        print(f"<--- New model ---> {modelo.__class__.__name__}") 

        scores = model_selection.cross_val_score(modelo, train_vectors, train_df["target"], cv=5, scoring="f1")

        print(modelo, scores, scores.mean())

        
from sklearn.linear_model import LogisticRegression

models = list()

for i in [0.1, 1, 100,1000]:

    clf1 = LogisticRegression(C = i, penalty = 'l2', random_state=42, solver='liblinear')

    clf2 = LogisticRegression(C = i, penalty = 'l1', random_state=42, solver='liblinear')

    clf3 = linear_model.RidgeClassifier(alpha = i,tol=1e-2, solver="sag")

    models += [clf1, clf2, clf3]





evaluate_model_list(models)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression( penalty = 'l1', random_state=0)

clf



scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv("submission.csv", index=False)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

def make_submissions(classif_model):

    classif_model.fit(train_vectors, train_df["target"])

    sample_submission["target"] = classif_model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)

    print("SUBMISSION SATISFACTORIA")



make_submissions(linear_model.RidgeClassifier(alpha = 100,tol=1e-2, solver="sag"))
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

def make_submissions(classif_model):

    classif_model.fit(train_vectors, train_df["target"])

    sample_submission["target"] = classif_model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)

    print("SUBMISSION SATISFACTORIA")



make_submissions(linear_model.RidgeClassifier(alpha = 100,tol=1e-2, solver="sag"))
train_df.head()
from sklearn.naive_bayes import GaussianNB, MultinomialNB
gnb = GaussianNB()

lista_NBS = [gnb]

for i in [0.1, 1.0, 10, 100]:

    mnb = MultinomialNB(alpha=i)

    lista_NBS += [mnb]

lista_NBS
def evaluate_model_list(lista_modelos):

    for modelo in lista_modelos:

        scores = model_selection.cross_val_score(modelo, train_vectors.todense(), train_df["target"], cv=5, scoring="f1")

        print(modelo, scores, scores.mean())

        print("<--- New model ---> ")

evaluate_model_list(lista_NBS)
make_submissions(MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))
# from sklearn.tree import DecisionTreeClassifier



# tree1 = DecisionTreeClassifier(criterion = 'gini' ,random_state=0, max_depth=5)

# tree2 = DecisionTreeClassifier(criterion = 'entropy' ,random_state=0, max_depth=5)

# tree_list = [tree1, tree2]

# evaluate_model_list(tree_list)
# from sklearn.ensemble import AdaBoostClassifier



# from timeit import default_timer as timer



# start = timer()

# adaboosts = [AdaBoostClassifier(n_estimators=i, random_state=42) for i in [1, 10, 20]]

# evaluate_model_list(adaboosts)

# end = timer()

# print(end - start)



# from sklearn.ensemble import RandomForestClassifier



# start = timer()

# evaluate_model_list([RandomForestClassifier(n_estimators = i, random_state = 42) for i in [1,10,50,100]])

# end = timer()

# print(end - start)

## PODEMOS VER QUE LOS ARBOLES COMPUESTOS FUNCIONAN MEJOR QUE LOS SIMPLES

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_vectors, train_df['target'], test_size=0.3, random_state=0)

from xgboost import XGBClassifier



xgb_classifir = XGBClassifier(

                              learning_rate=0.1,

                              num_round=1000,

                              max_depth=10,

                              min_child_weight=2,

                              colsample_bytree=0.8,

                              subsample=0.9,

                              gamma=0.4,

                              reg_alpha=1e-5,

                              reg_lambda=1,

                              n_estimators=2000,

                              objective='binary:logistic',

                              eval_metric=["auc", "logloss", "error"],

                              early_stopping_rounds=50)

xgb_classifir.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_valid, y_valid)])

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve





y_pred_xgb = xgb_classifir.predict(X_valid)

print(confusion_matrix(y_valid, y_pred_xgb))

print(accuracy_score(y_valid, y_pred_xgb))

print(f1_score(y_valid, y_pred_xgb))



sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# sample_submission["target"] = xgb_classifir.predict(X_valid)

# sample_submission.to_csv("submission.csv", index=False)
y_pred_xgb = xgb_classifir.predict(test_vectors)

y_pred_xgb

print(sample_submission["target"].head(10))

sample_submission["target"] = y_pred_xgb

print("NUEVOS")

print(sample_submission["target"].head(10))

sample_submission.to_csv("submission.csv", index=False)
import nltk

sentence = "My name is George and I love NLP from New York"

tokens = nltk.word_tokenize(sentence)

print(tokens)
import nltk

from nltk.corpus import stopwords



sentence = "This is a sentence for removing stop words"

print(f"Sentence --> {sentence}")

tokens = nltk.word_tokenize(sentence)



stop_words = stopwords.words('english')

filtered_tokens, unfiltered_tokens = [w for w in tokens if w not in stop_words], [w for w in tokens if w in stop_words]

print(f"Filtered toekns {filtered_tokens}, and unfiltered {unfiltered_tokens}")

print(stop_words,)

print(len(stop_words), len(stopwords.words('spanish')))

print(stopwords.words('spanish'),)
import nltk



snowball_stemmer = nltk.stem.SnowballStemmer('spanish')
def stemer (string):

    return snowball_stemmer.stem(string)
cafes = ['café', 'cafetera', 'descafeinado', 'cafetero', 'cafeteria', 'cafeína']
[stemer(palabra) for palabra in cafes]
def stemer1 (string):

    return nltk.stem.SnowballStemmer('english').stem(string)

[stemer1(palabra) for palabra in ['cook', 'cooks', 'cooked', 'cooking']]