import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from nltk import word_tokenize

from scipy.sparse import coo_matrix

from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

df = pd.read_csv('../input/train.csv')#[:50000]

test=pd.read_csv('../input/test.csv')#[:10000]

df.head()
df['target'].value_counts()






tf_vectorizer =CountVectorizer(binary=True,strip_accents='unicode',max_features=90000).fit(df['question_text'].append(test['question_text']))

listOfWords = tf_vectorizer.get_feature_names()

dictOfWords = { listOfWords[i]:i for i in range(0, len(listOfWords) ) }

tf_vectorizer.transform(df['question_text'])

from nltk import word_tokenize

from scipy.sparse import coo_matrix



def create_cooccurrence_matrix(filename,tokenizer,window_size,vocabulary):

    #vocabulary={}

    data=[]

    row=[]

    col=[]

    for sentence in filename:

        sentence=sentence.strip()

        #print(sentence)

        tokens=[token for token in tokenizer(sentence) if token!=u""]

        for pos,token in enumerate(tokens):

            i=vocabulary.setdefault(token,len(vocabulary))

            start=max(0,pos-window_size)

            end=min(len(tokens),pos+window_size+1)

            for pos2 in range(start,end):

                if pos2==pos: 

                    continue

                j=vocabulary.setdefault(tokens[pos2],len(vocabulary))

                data.append(1.); row.append(i); col.append(j);

    

    cooccurrence_matrix=coo_matrix((data,(row,col)))

    return vocabulary,cooccurrence_matrix

#voca,coo=create_cooccurrence_matrix(df.question_text.str.lower().values,word_tokenize,100,dictOfWords)

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer





X_train, X_test, y_train, y_test = train_test_split(df['question_text'],

                                                    df['target'],

                                                    test_size=0.2)

#tf_vectorizer = TfidfVectorizer().fit(df_under['question_text'])

#tf_vectorizer = CountVectorizer(tokenizer=word_tokenize,stop_words).fit(df['question_text'])

X_train = tf_vectorizer.transform(X_train)

X_test = tf_vectorizer.transform(X_test)

X_train.shape
#from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1)

clf.fit(X_train, y_train)

#clf = MultinomialNB().fit(X_train, y_train)

predicted = clf.predict(X_test)

np.mean(predicted == y_test)
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.neighbors import NearestCentroid

from sklearn.linear_model import RidgeClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.metrics import f1_score,confusion_matrix

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



names = ["LR",#"MLP",

        #"SVC",        #"SVC3",

        "XGB",

         "Passive-Aggressive",    

        "linearSVC","NearestCentroid",

        "multNB","bernouilliNB","Ridge Classifier",

         "Perceptron",#"kNN",



         "SGD modeL2","SGD elast",

         #"Nearest Neighbors",# "Linear SVM", 

         #"RBF SVM", #"Gaussian Process",

         "Decision Tree", #"Random Forest", #"Neural Net",

        "AdaBoost",

         #"Naive Bayes" #, "QDA"

        ]



classifiers = [

    LogisticRegression(),

    #MLPClassifier(),

    #SVC(kernel='linear'),

    #SVC(kernel='sigmoid'),

    XGBClassifier(learning_rate=0.1,n_estimators=100),

    PassiveAggressiveClassifier(max_iter=50, tol=1e-3),    

    LinearSVC(penalty="l2", dual=False,tol=1e-3),

    NearestCentroid(),

    MultinomialNB(alpha=.01),

    BernoulliNB(alpha=.01),

    RidgeClassifier(tol=1e-2, solver="sag"),

    Perceptron(max_iter=50, tol=1e-3),

    #KNeighborsClassifier(n_neighbors=10),



    SGDClassifier(alpha=.0001, max_iter=50,penalty="l2"),

    SGDClassifier(alpha=.0001, max_iter=50,penalty="elasticnet"),

    #KNeighborsClassifier(5),

    

    #SVC(kernel="linear", C=0.025),

    #SVC(gamma=2, C=1),

    #GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    #MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    #GaussianNB(),

    #QuadraticDiscriminantAnalysis()

    ]



#or countmatrix or tfidfmatrix

#X_train, X_test, y_train, y_test = train_test_split(countmatrix, y, test_size=0.2, random_state=42)

    # iterate over classifiers

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    y_pred=clf.predict(X_test)

    print(name,score,f1_score(y_test,y_pred))

    print('Confusion matrix:', confusion_matrix(y_pred, y_test)  )  
from sklearn.metrics import f1_score





f1_score(y_test, predicted,average=None)
df_test = pd.read_csv('../input/test.csv')

X_submission = tf_vectorizer.transform(df_test['question_text'])

predicted_test = clf.predict(X_submission)



df_test['prediction'] = predicted_test

submission = df_test.drop(columns=['question_text'])

submission.head()
submission.to_csv('submission.csv', index=False)