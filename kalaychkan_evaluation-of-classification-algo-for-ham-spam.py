# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Libraries

import re

import nltk # natural language tool kit

nltk.download("stopwords")

import nltk

nltk.download('punkt')

import nltk

nltk.download('wordnet')

from nltk.corpus import stopwords

import nltk as nlp

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



#Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





#Visualization

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

data = pd.concat([data.Category,data.Message],axis=1)

data.head()
data.dropna(axis = 0,inplace = True)

data.Category = [1 if each == "ham" else 0 for each in data.Category]
data.head()
first_description = data.Message[2]

description = re.sub("[^a-zA-Z]"," ",first_description)

description = description.lower() 



description
description = nltk.word_tokenize(description)



description
description = [ word for word in description if not word in set(stopwords.words("english"))]

description
lemma = nlp.WordNetLemmatizer()

description = [ lemma.lemmatize(word) for word in description] 



description = " ".join(description)

description
description_list = []

for description in data.Message:

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()   

    description = nltk.word_tokenize(description)

    description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description)

description_list
from sklearn.feature_extraction.text import CountVectorizer 

max_features = 5000

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x

print(" Commonly Used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

y = data.iloc[:,0].values   # Ham - Spam classes

x = sparce_matrix

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)

grid = {"n_neighbors":np.arange(1,5)}

knn= KNeighborsClassifier()



knn_cv = GridSearchCV(knn, grid, cv = 5)  # GridSearchCV

knn_cv.fit(x,y)



print("tuned hyperparameter K: ",knn_cv.best_params_)

print("tuned parameter (best score): ",knn_cv.best_score_)



knn = KNeighborsClassifier(n_neighbors=2)  # k = n_neighbors



accuracies = cross_val_score(estimator = knn, X = x_train, y= y_train, cv = 10) #Cross Validation

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))



#Confusion Matrix

y_pred = knn_cv.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge



logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,grid,cv = 5) #GridSearchCV

logreg_cv.fit(x,y)



print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)

print("accuracy: ",logreg_cv.best_score_)
#Confusion Matrix

y_pred = logreg_cv.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
models = []

models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 

models.append(('RandomForestClassifier',RandomForestClassifier(n_estimators = 100,random_state = 1)))



for name, model in models:

    model = model.fit(x_train,y_train)

    ACC = model.score(x_test,y_test)

    accuracies = cross_val_score(estimator = model, X = x_train, y= y_train, cv = 10) #CrossFold Validation

    print("{} -> ACC: {} ".format(name,ACC))

    print("average accuracy: ",np.mean(accuracies))

    print("average std: ",np.std(accuracies))

    #Confusion Matrix

    y_pred = model.predict(x_test)

    y_true = y_test

    cm = confusion_matrix(y_true,y_pred)

    f, ax = plt.subplots(figsize =(5,5))

    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

    plt.xlabel("y_pred")

    plt.ylabel("y_true")

    plt.show()