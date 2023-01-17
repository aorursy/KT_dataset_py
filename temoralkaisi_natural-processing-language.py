# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import matplotlib.pyplot as plt



import numpy as np

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



items = pd.read_csv("/kaggle/input/amazon-cell-phones-reviews/20191226-items.csv")

reviews = pd.read_csv("/kaggle/input/amazon-cell-phones-reviews/20191226-reviews.csv")
reviews = pd.merge(reviews, items, how="left", left_on="asin", right_on="asin")
reviews.rename(columns={"rating_x": "rating", "title_x": "title", "title_y": "item_title", "rating_y": "overall_rating"}, inplace=True)
reviews['rating'].iplot(

    kind='hist',

    xTitle='rating',

    linecolor='black',

    yTitle='count',

    title='Review Rating Distribution')
reviews = reviews.dropna()

reviews = reviews.reset_index(drop=True)
reviews.isnull().sum()
#Create Brands

apple = reviews[reviews["brand"]=="Apple"].sort_values(by=["date"], ascending=False)

xiaomi = reviews[reviews["brand"]=="Xiaomi"].sort_values(by=["date"], ascending=False)

samsung = reviews[reviews["brand"]=="Samsung"].sort_values(by=["date"], ascending=False)

Motorola = reviews[reviews["brand"]=="Motorola"].sort_values(by=["date"], ascending=False)





apple.dropna(inplace=True)

apple[reviews['rating'] != 3]

apple['Positivity'] = np.where(apple['rating'] > 3, 1, 0)

cols = ['asin', 'name', 'rating', 'date','verified', 'title', 'helpfulVotes', 'brand', 'item_title','url','image','overall_rating','reviewUrl','totalReviews','price','originalPrice']

apple.drop(cols, axis=1, inplace=True)

apple.head()
apple.rename(columns={"body": "Review"}, inplace=True)
xiaomi.dropna(inplace=True)

xiaomi[reviews['rating'] != 3]

xiaomi['Positivity'] = np.where(xiaomi['rating'] > 3, 1, 0)

cols = ['asin', 'name', 'rating', 'date', 'verified', 'title', 'helpfulVotes', 'brand', 'item_title','url','image','overall_rating','reviewUrl','totalReviews','price','originalPrice']

xiaomi.drop(cols, axis=1, inplace=True)

xiaomi.head()
xiaomi.rename(columns={"body": "Review"}, inplace=True)
samsung.dropna(inplace=True)

samsung[reviews['rating'] != 3]

samsung['Positivity'] = np.where(samsung['rating'] > 3, 1, 0)

cols = ['asin', 'name', 'rating', 'date', 'verified', 'title', 'helpfulVotes', 'brand', 'item_title','url','image','overall_rating','reviewUrl','totalReviews','price','originalPrice']

samsung.drop(cols, axis=1, inplace=True)

samsung.head()
samsung.rename(columns={"body": "Review"}, inplace=True)
Motorola.dropna(inplace=True)

Motorola[reviews['rating'] != 3]

Motorola['Positivity'] = np.where(Motorola['rating'] > 3, 1, 0)

cols = ['asin', 'name', 'rating', 'date', 'verified', 'title', 'helpfulVotes', 'brand', 'item_title','url','image','overall_rating','reviewUrl','totalReviews','price','originalPrice']

Motorola.drop(cols, axis=1, inplace=True)

Motorola.head()









Motorola.rename(columns={"body": "Review"}, inplace=True)
sns.catplot(x="Positivity", data=Motorola, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")

plt.show();
sns.catplot(x="Positivity", data=apple, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")

plt.show();
sns.catplot(x="Positivity", data=samsung, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")

plt.show();
sns.catplot(x="Positivity", data=xiaomi, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")

plt.show();
blanks = []  # start with an empty list



for i,lb,rv in apple.itertuples():  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
apple.drop(blanks, inplace=True)



len(apple)
blanks = []  # start with an empty list



for i,lb,rv in xiaomi.itertuples():  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
blanks = []  # start with an empty list



for i,lb,rv in samsung.itertuples():  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

x = apple['Review']

y = apple['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))



# Report the confusion matrix

from sklearn import metrics







print(metrics.accuracy_score(y_test,predictions))
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

x = apple['Review']

y = apple['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))





# Report the confusion matrix and showing accuracy result 

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier

x = apple['Review']

y = apple['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',DecisionTreeClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

x = apple['Review']

y = apple['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

x = apple['Review']

y = apple['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

x = samsung['Review']

y = samsung['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))



# Report the confusion matrix

from sklearn import metrics
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

x = samsung['Review']

y = samsung['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))

# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier

x = samsung['Review']

y = samsung['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',DecisionTreeClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

x = samsung['Review']

y = samsung['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

x = samsung['Review']

y = samsung['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

x = xiaomi['Review']

y = xiaomi['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))



# Report the confusion matrix

from sklearn import metrics
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

x = xiaomi['Review']

y = xiaomi['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))



# Report the confusion matrix

from sklearn import metrics
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier

x = xiaomi['Review']

y = xiaomi['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',DecisionTreeClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))



# Report the confusion matrix

from sklearn import metrics
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

x = xiaomi['Review']

y = xiaomi['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

x = Motorola['Review']

y = Motorola['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))



# Report the confusion matrix

from sklearn import metrics
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

x = Motorola['Review']

y = Motorola['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))

# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier

x = Motorola['Review']

y = Motorola['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',DecisionTreeClassifier())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

x = Motorola['Review']

y = Motorola['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

x = Motorola['Review']

y = Motorola['Positivity']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)





from sklearn.pipeline import Pipeline



text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression())])



text_clf.fit(X_train,y_train)



predictions = text_clf.predict(X_test)



print(confusion_matrix(y_test,predictions))

cm = confusion_matrix(y_test,predictions)

print(classification_report(y_test,predictions))
# Report the confusion matrix

from sklearn import metrics

print(metrics.accuracy_score(y_test,predictions))



cm = metrics.accuracy_score(y_test,predictions)