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
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')
train_data.head()
x_train=train_data.iloc[:, 1:4]

y_train=train_data.iloc[:, -1:]

x_test=test_data.iloc[:, 1:4]
import category_encoders as ce

from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = '{:.2f}'.format
ce_ord = ce.OrdinalEncoder(cols = ['keyword', 'location'])

train_data = ce_ord.fit_transform(x_train, y_train)
import string

from nltk.corpus import stopwords

text = train_data.iloc[:,-1]

stopwords=stopwords.words('english')



for n in range(len(train_data)):

    querywords = text[n].split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]

    text[n] = ' '.join(resultwords)

    

    text[n] = ''.join([c for c in text[n] if c not in (string.punctuation)])

    train_data.iloc[n, -1] = text[n]

train_data.head

    
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def cv(data):

    count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')



    emb = count_vectorizer.fit_transform(data)



    return emb, count_vectorizer



list_corpus_train = train_data["text"].tolist()

list_labels_train = y_train["target"].tolist()



list_corpus_test = test_data["text"].tolist()



X_train_final = list_corpus_train

y_train_final = list_labels_train

X_test_final = list_corpus_test



X_train, X_test, y_train, y_test = train_test_split(list_corpus_train, list_labels_train, test_size=0.2, random_state=42)

                                                                             



X_train_counts, count_vectorizer = cv(X_train)

print(count_vectorizer.vocabulary_)

X_test_counts = count_vectorizer.transform(X_test)



X_train_counts_final, count_vectorizer_final = cv(X_train_final)

X_test_final = count_vectorizer.transform(X_test_final)

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt



def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):

        lsa = TruncatedSVD(n_components=2)

        lsa.fit(test_data)

        lsa_scores = lsa.transform(test_data)

        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}

        color_column = [color_mapper[label] for label in test_labels]

        colors = ['green','red','red']

        if plot:

            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))

            red_patch = mpatches.Patch(color='green', label='Not disaster')

            green_patch = mpatches.Patch(color='red', label='Real disaster')

            plt.legend(handles=[red_patch, green_patch], prop={'size': 30})





plt.figure(figsize=(16, 16))          

plot_LSA(X_train_counts, y_train)

plt.show()
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(C=1, class_weight='balanced', solver='newton-cg', 

                         multi_class='multinomial', n_jobs=-1, random_state=40)

clf.fit(X_train_counts, y_train)



y_predicted_counts = clf.predict(X_test_counts)


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report



def get_metrics(y_test, y_predicted):  

    # true positives / (true positives+false positives)

    precision = precision_score(y_test, y_predicted, pos_label=None,

                                    average='weighted')             

    # true positives / (true positives + false negatives)

    recall = recall_score(y_test, y_predicted, pos_label=None,

                              average='weighted')

    

    # harmonic mean of precision and recall

    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    

    # true positives + true negatives/ total

    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1



accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
def tfidf(data):

    tfidf_vectorizer = TfidfVectorizer()



    train = tfidf_vectorizer.fit_transform(data)



    return train, tfidf_vectorizer



X_train_tfidf, tfidf_vectorizer = tfidf(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
fig = plt.figure(figsize=(16, 16))          

plot_LSA(X_train_tfidf, y_train)

plt.show()
clf_tfidf = LogisticRegression(C=1.0, class_weight='balanced', solver='newton-cg', 

                         multi_class='multinomial', n_jobs=-1, random_state=40)

clf_tfidf.fit(X_train_tfidf, y_train)



y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf, 

                                                                       recall_tfidf, f1_tfidf))
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=42, max_depth=135)

tree_clf.fit(X_train_counts, y_train)

y_predicted_tree = tree_clf.predict(X_test_counts)
accuracy_tree, precision_tree, recall_tree, f1_tree = get_metrics(y_test, y_predicted_tree)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tree, precision_tree, 

                                                                       recall_tree, f1_tree))
tree_reg = tree.DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train_counts, y_train)

y_predicted_tree_reg = tree_reg.predict(X_test_counts).astype(int)
accuracy_tree_reg, precision_tree_reg, recall_tree_reg, f1_tree_reg = get_metrics(y_test, y_predicted_tree_reg)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tree_reg, precision_tree_reg, 

                                                                       recall_tree_reg, f1_tree_reg))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=50, n_jobs=4,  random_state=42)

regressor.fit(X_train_counts, y_train)

y_predicted_regressor = regressor.predict(X_test_counts).astype(int)
accuracy_regressor, precision_regressor, recall_regressor, f1_regressor = get_metrics(y_test, y_predicted_regressor)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_regressor, precision_regressor, 

                                                                       recall_regressor, f1_regressor))
plt.figure(figsize=(16, 16))          

plot_LSA(X_train_counts_final, y_train_final)

plt.show()
clf_final = LogisticRegression(C=1, class_weight='balanced', solver='newton-cg', 

                         multi_class='multinomial', n_jobs=-1, random_state=40)

clf_final.fit(X_train_counts_final, y_train_final)



y_predicted_counts_final = clf.predict(X_test_counts)
print(y_predicted_counts_final)
pd.DataFrame(y_predicted_counts_final).to_csv("submission.csv")