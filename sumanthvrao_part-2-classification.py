!pip install textstat

!pip install language_check
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/overall/overall"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import glob

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,

                              AdaBoostClassifier)

from sklearn.model_selection import train_test_split

import _pickle as cPickle

import textstat

import pandas as pd

import numpy as np

import textstat

import glob

from nltk.corpus import stopwords

import re

from sklearn.neural_network import MLPClassifier

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,

                              AdaBoostClassifier)

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn                        import metrics, svm

from sklearn.svm                    import SVC

from sklearn.neighbors              import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold

import statistics

import string

from nltk.corpus import stopwords

import re

from nltk.stem.snowball import SnowballStemmer
df = pd.DataFrame(columns=['text', 'label'])



path="../input/overall/overall"



i=0

file_list = glob.glob(path+"/fake/*.txt")

for file_name in file_list:

    file=open(file_name,"r", encoding="utf8")

    a=file.read()

    df.loc[i]=[a,1]

    i=i+1



file_list = glob.glob(path+"/real/*.txt")

for file_name in file_list:

    file=open(file_name,"r", encoding="utf8")

    a=file.read()

    df.loc[i]=[a,0]

    i=i+1 
df.tail()
tag_dict={

    'J':wordnet.ADJ,

    'V':wordnet.VERB,

    "N": wordnet.NOUN,

    "R": wordnet.ADV

}

def get_pos_tag(word):

    tag=nltk.pos_tag([word])[0][1][0].upper()

    return tag_dict.get(tag,wordnet.NOUN)



def clean_text(text):

    ## Remove puncuation

    #text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    ## Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) >= 3]

    

    text = " ".join(text)



    ## Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", "", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", "", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", "", text)

    text = re.sub(r"\.", "", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", "", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", "", text)



    ## Stemming

#     text = text.split()

#     stemmer = SnowballStemmer('english')

#     stemmed_words = [stemmer.stem(word) for word in text]

#     text = " ".join(stemmed_words)

    ##Lemmatization

    sentence_clean=[]

    for word in text.split():

        sentence_clean.append(lemmatizer.lemmatize(word,get_pos_tag(word)))

    return " ".join(sentence_clean)





# apply the above function to df['text']

lemmatizer = WordNetLemmatizer() 

df['text'] = df['text'].map(lambda x: clean_text(x))
print(df.loc[1,'text'])

#tfidf

transformer = TfidfTransformer(smooth_idf=False)

count_vectorizer = CountVectorizer(ngram_range=(2, 3))

counts = count_vectorizer.fit_transform(df['text'].values)

tfidf = transformer.fit_transform(counts)



target=df['label'].values.astype('int')

selector = SelectKBest(chi2, k=1000)

selector.fit(tfidf, target)

top_words = selector.get_support().nonzero()



# Pick only the most informative columns in the data.

chi_matrix = tfidf[:,top_words[0]]


# Our list of functions to apply.

transform_functions = [

    

    lambda x: x.count(" ")/len(x.split()),

    lambda x: x.count(".")/len(x.split()),

    lambda x: x.count("!")/len(x.split()),

    lambda x: x.count("?")/len(x.split()),

    lambda x: x.count("-")/len(x.split()),

    lambda x: x.count(",")/len(x.split()),

    lambda x: x.count("$")/len(x.split()),

    lambda x: x.count("(")/len(x.split()),

    lambda x: len(x) / (x.count(" ") + 1),

    lambda x: x.count(" ") / (x.count(".") + 1),

    lambda x: len(re.findall("\d", x)),

    lambda x: len(re.findall("[A-Z]", x)),

    lambda x: textstat.flesch_reading_ease(x),

    lambda x: textstat.smog_index(x),

    lambda x: textstat.flesch_kincaid_grade(x),

    lambda x: textstat.coleman_liau_index(x),

    lambda x: textstat.automated_readability_index(x),

    lambda x: textstat.dale_chall_readability_score(x),

    lambda x: textstat.difficult_words(x),

    lambda x: textstat.linsear_write_formula(x),

    lambda x: textstat.gunning_fog(x),

]



# Apply each function and put the results into a list.

columns = []

for func in transform_functions:

    columns.append(df["text"].apply(func))

    

# Convert the meta features to a numpy array.

meta = np.asarray(columns).T

features = np.hstack([ meta,chi_matrix.todense()])

targets = df['label'].values

print('Features-shape: ',features.shape)

print('Target-shape: ',target.shape)

#split in samples

from sklearn.model_selection import train_test_split

features, features_test, targets, targets_test = train_test_split(features, targets, random_state=0)

features_test = features_test.astype('int')

targets_test = targets_test.astype('int')

print(features_test.shape,targets_test.shape)

print(features.shape)
## MODEL 1: Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

import statistics

#split in samples

skf = StratifiedKFold(n_splits=5, random_state=0,shuffle=True) #5 splits; 80-20 splitting, using Stratified K fold

targets = targets.astype('int')



C_possible = [1e0,1e-1,1e-2,1e-3]

for C in C_possible:

    clf_lr=LogisticRegression(C=C, solver='liblinear')

    score=[]

    for train_index, val_index in skf.split(features,targets):

        X_train, X_test = features[train_index], features[val_index] 

        y_train, y_test = targets[train_index], targets[val_index]

        clf_lr.fit(X_train,y_train)

        confidence = clf_lr.score(X_test, y_test)

        score.append(confidence) 

    print(score)

    print('C: ',C,' and score: ',statistics.mean(score))

    print("Testing accuracy")

    print(clf_lr.score(features_test,targets_test))
## MODEL 2: MLP classifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold

import statistics

#split in samples

skf = StratifiedKFold(n_splits=5, random_state=8,shuffle=True) #5 splits; 80-20 splitting, using Stratified K fold

targets = targets.astype('int')



alpha_possible = [1e0]

for alpha in alpha_possible:

    clf_mlp=MLPClassifier(alpha=alpha, learning_rate='adaptive',random_state=10, solver='lbfgs')

    score=[]

    for train_index, val_index in skf.split(features,targets):

        X_train, X_test = features[train_index], features[val_index] 

        y_train, y_test = targets[train_index], targets[val_index]

        clf_mlp.fit(X_train,y_train)

        confidence = clf_mlp.score(X_test, y_test)

        score.append(confidence) 

    print(score)

    print('Alpha: ',alpha,' and score: ',statistics.mean(score))

    print("Testing accuracy")

    print(clf_mlp.score(features_test,targets_test))
## MODEL 3: Decision trees

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

import statistics

#split in samples

skf = StratifiedKFold(n_splits=5, random_state=2,shuffle=True) #5 splits; 80-20 splitting, using Stratified K fold

targets = targets.astype('int')



min_imp = [1e-5]

for imp in min_imp:

    clf_dt=DecisionTreeClassifier(criterion='gini', splitter='best', min_impurity_decrease=imp)

    score=[]

    for train_index, val_index in skf.split(features,targets):

        X_train, X_test = features[train_index], features[val_index] 

        y_train, y_test = targets[train_index], targets[val_index]

        clf_dt.fit(X_train,y_train)

        confidence = clf_dt.score(X_test, y_test)

        score.append(confidence) 

    print(score)

    print('Impurity ',imp,' and score: ',statistics.mean(score))

    print("Testing accuracy")

    print(clf_dt.score(features_test,targets_test))

## MODEL 4: SVM

from sklearn.model_selection import StratifiedKFold

import statistics

#split in samples

score=[]

skf = StratifiedKFold(n_splits=5, random_state=None) #5 splits; 80-20 splitting, using Stratified K fold

targets = targets.astype('int')

clf_svc=SVC(gamma='auto')

for train_index, val_index in skf.split(features,targets):

    X_train, X_test = features[train_index], features[val_index] 

    y_train, y_test = targets[train_index], targets[val_index]

    clf_svc.fit(X_train,y_train)

    confidence = clf_svc.score(X_test, y_test)

    score.append(confidence)

print(score)

print("Testing accuracy")

print(clf_svc.score(features_test,targets_test))
## MODEL 5: Random Forest

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

import statistics

#split in samples

score=[]

skf = StratifiedKFold(n_splits=5, random_state=None) #5 splits; 80-20 splitting, using Stratified K fold

targets = targets.astype('int')

clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

for train_index, val_index in skf.split(features,targets):

    X_train, X_test = features[train_index], features[val_index] 

    y_train, y_test = targets[train_index], targets[val_index]

    clf_rf.fit(X_train,y_train)

    confidence = clf_rf.score(X_test, y_test)

    score.append(confidence)

print(score)

print("Testing accuracy")

print(clf_rf)

print(clf_rf.score(features_test,targets_test))
## MODEL 6: KNN

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

import statistics

#split in samples

score=[]

skf = StratifiedKFold(n_splits=5, random_state=None) #5 splits; 80-20 splitting, using Stratified K fold

targets = targets.astype('int')

clf_knn = KNeighborsClassifier(n_neighbors=7)

for train_index, val_index in skf.split(features,targets):

    X_train, X_test = features[train_index], features[val_index] 

    y_train, y_test = targets[train_index], targets[val_index]

    clf_knn.fit(X_train,y_train)

    confidence = clf_knn.score(X_test, y_test)

    score.append(confidence)

print(score)

print("Testing accuracy")

print(clf_knn)

print(clf_knn.score(features_test,targets_test))
classifiers=[clf_lr,clf_mlp,clf_dt,clf_svc,clf_rf,clf_knn]

learner_error = dict()

def feature_extraction_meta_classifer(feat,tar):

    global learner_error

    classifier_outputs=[]

    print(type(feat))

    for classifier in classifiers:

        l=classifier.predict(feat)

        diff = l - tar

        print(diff)

        error_index_list = np.argwhere(diff)

        error_index_list=np.array([i[0] for i in error_index_list])

        learner_error[str(classifier)]=error_index_list

        classifier_outputs.append(l)

    classifier_outputs=np.array(classifier_outputs).T

    targets_outputs=tar[:]

    return (classifier_outputs,targets_outputs)
#Creating a testing dataset

classifier_outputs,targets_outputs=feature_extraction_meta_classifer(features,targets)

classifier_outputs_test,targets_outputs_test=feature_extraction_meta_classifer(features_test,targets_test)
#print(learner_error)

# error_lr=set(learner_error[str(clf_knn)])

# error_mlp=set(learner_error[str(clf_mlp)])

# intersect=len(error_lr.intersection(error_mlp))

# total=len(error_lr)

# print(intersect/total)
# from sklearn.ensemble import AdaBoostClassifier

# clf_final=AdaBoostClassifier()

# clf_final.fit(classifier_outputs,targets)

# print(clf_final.score(classifer_outputs,targets))

# clf_final.score(classifer_outputs_test,targets_test)
# print(clf_final.estimator_weights_)



# print(clf_final.n_classes_)
df_classifiers=pd.DataFrame(classifier_outputs)

df_classifiers.corr()
#Based on the correlation matrix, consider all columns except column 4

classifier_output_trimmed=classifier_outputs[:,[0,1,2,4,5]]

classifier_output_test_trimmed=classifier_outputs_test[:,[0,1,2,4,5]]
from sklearn.neural_network import MLPClassifier

clf_final=MLPClassifier(alpha=1e1)

clf_final.fit(classifier_output_trimmed,targets_outputs)

print(clf_final.score(classifier_output_trimmed,targets_outputs))

clf_final.score(classifier_output_test_trimmed,targets_test)