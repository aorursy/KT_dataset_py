# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import string

import nltk

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
df = pd.read_excel("../input/News Title.xls")

df = df.iloc[0:5000,:]

df.head()
import seaborn as sns

%matplotlib inline
df = df.drop(columns=['No'])

df.head()
df.describe()
sns.set_style("whitegrid")

plt.figure(figsize=(5,5))

sns.countplot(df['Category'])
df['length'] = df['News Title'].apply(len)

df.head()
df.shape
import re

from nltk.stem.porter import PorterStemmer

#Every mail starts with 'Subject :' will remove this from each text 

df['News Title'] = df['News Title'].map(lambda text: text[1:])

df['News Title'] = df['News Title'].map(lambda text:re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())

ps = PorterStemmer()

corpus=df['News Title'].apply(lambda text_list:' '.join(list(map(lambda word:ps.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))



# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus.values).toarray()

Y = df.iloc[:, 1].values
# encode the labels, converting them from strings to integers

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

Y = le.fit_transform(Y)

Y
# perform a training testing split, using 75% of the data for

# training and 25% for evaluation

(trainX, testX, trainY, testY) = train_test_split(X, Y, random_state=3, test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, accuracy_score
# define the dictionary of models our script can use

# the key to the dictionary is the name of the model

# (supplied via command line argument) and the value is the model itself

models = {

    "knn": KNeighborsClassifier(n_neighbors=1),

    "naive_bayes": MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),

    "logit": LogisticRegression(solver="lbfgs", multi_class="multinomial"),

    "svm": SVC(kernel="sigmoid", gamma="scale",degree=3, C=1.0),

    "decision_tree": DecisionTreeClassifier(),

    "random_forest": RandomForestClassifier(n_estimators = 200, oob_score = True, n_jobs = -1, random_state = 5, max_features = "auto"),

    'mlp': MLPClassifier(hidden_layer_sizes=(500,500))

}
# train the K-Neighbors model

print("[INFO] using '{}' model".format("knn"))

model = models["knn"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions1 = model.predict(testX)

KNN_accur = accuracy_score(testY, predictions1)

print('\nSkor akurasi kemampuan model K-Neighbors dalam mengklasifikasikan News Title adalah:', KNN_accur)
# train the Naive Bayes model

print("[INFO] using '{}' model".format("naive_bayes"))

model = models["naive_bayes"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions2 = model.predict(testX)

NB_accur = accuracy_score(testY, predictions2)

print('\nSkor akurasi kemampuan model Naive Bayes dalam mengklasifikasikan News Title adalah:', NB_accur)
# train the Logistic Regression model

print("[INFO] using '{}' model".format("logit"))

model = models["logit"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions3 = model.predict(testX)

Logit_accur = accuracy_score(testY, predictions3)

print('\nSkor akurasi kemampuan model Logistic Regression dalam mengklasifikasikan News Title adalah:', Logit_accur)
# train the Support Vector Machine model

print("[INFO] using '{}' model".format("svm"))

model = models["svm"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions4 = model.predict(testX)

SVM_accur = accuracy_score(testY, predictions4)

print('\nSkor akurasi kemampuan model Support Vector Machine dalam mengklasifikasikan News Title adalah:', SVM_accur)
# train the Decision Tree model

print("[INFO] using '{}' model".format("decision_tree"))

model = models["decision_tree"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions5 = model.predict(testX)

DT_accur = accuracy_score(testY, predictions5)

print('\nSkor akurasi kemampuan model Decision Tree dalam mengklasifikasikan News Title adalah:', DT_accur)
# train the Random Forest model

print("[INFO] using '{}' model".format("random_forest"))

model = models["random_forest"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions6 = model.predict(testX)

RF_accur = accuracy_score(testY, predictions6)

print('\nSkor akurasi kemampuan model Random Forest dalam mengklasifikasikan News Title adalah:', RF_accur)
# train the MLP model

print("[INFO] using '{}' model".format("mlp"))

model = models["mlp"]

model.fit(trainX, trainY)

# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

predictions7 = model.predict(testX)

MLP_accur = accuracy_score(testY, predictions7)

print('\nSkor akurasi kemampuan model MLP dalam mengklasifikasikan News Title adalah:', MLP_accur)
# train the Ensemble Mean model

print("[INFO] using '{}' from 7 models".format("Ensemble Mean"))

ensembleArr = np.stack((predictions1,predictions2,predictions3,predictions4,predictions5,predictions6,predictions7))

ensembleArr = np.around(np.mean(ensembleArr,axis=0))



# make predictions on our data and show a accuracy score

print("[INFO] evaluating...")

Ensemble_accur = accuracy_score(testY, ensembleArr)

print('\nSkor akurasi kemampuan model Ensemble Mean dalam mengklasifikasikan News Title adalah:', Ensemble_accur)
score_obj = dict(zip(['Accuracy Score'], [[KNN_accur,NB_accur,Logit_accur,SVM_accur,DT_accur,RF_accur,MLP_accur,Ensemble_accur]]))



score_df = pd.DataFrame(score_obj, index=['K-Neighbors','Naive Bayes','Logistic Regression','Support Vector Machine','Decision Tree', 'Random Forest', 'Multi-Layer Perceptron', 'Ensemble Mean'])

score_df.sort_values(by='Accuracy Score', ascending=False)