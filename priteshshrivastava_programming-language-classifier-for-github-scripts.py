import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.metrics import plot_confusion_matrix, confusion_matrix

import eli5



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample_code = pd.read_csv("/kaggle/input/sample-github-code/sample_code.csv", 

                          lineterminator='\n')  

## read CSV error : https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err



sample_code.head()
sample_code.describe(include='all')
## Combine .C, .cpp, .cc, .h to C++ file, .cpp, say  ## Source : ## http://gcc.gnu.org/onlinedocs/gcc-4.4.1/gcc/Overall-Options.html#index-file-name-suffix-71

sample_code.loc[sample_code['type'].isin(['C', 'cpp', 'cc', 'h']), 'type'] = 'cpp'

## Combine .html & .htm to .html, say

sample_code.loc[sample_code['type'].isin(['html', 'htm']), 'type'] = 'html'
type_counts = sample_code['type'].value_counts().to_frame().reset_index()

type_counts
## Filter irrelevant files like null, .gitignore using value counts

top_languages = type_counts[type_counts['type'] >= 1000]

top_languages
sns.barplot(data=top_languages, x='index', y='type')

plt.xticks(rotation=90)
## Filtering files created by IDEs, version control & data files

top_languages = top_languages[~top_languages['index'].isin(['sublime-snippet', 'xcworkspacedata', 'gitignore', 

                                                            'project', 'properties', 'conf', 'config', 'cfg',

                                                            'meta', 'test', 'gradle', 'patch', 'ebuild', 'ini',

                                                           'csv', 'json', 'txt', 'geojson', 'svg', 

                                                            'tpl', 'less', 'cmake', 'mk', 'd'])]



sns.barplot(data=top_languages, x='index', y='type')

plt.xticks(rotation=90)
## Filter with top_languages

train = sample_code[sample_code['type'].isin(top_languages['index'].values)]



print(train.shape)

train.head()
i=10

print("\n".join(train['content'][i].split("\n")[:10]))

print(train['type'].values[i])
train_x, val_x, train_y, val_y = model_selection.train_test_split(train['content'], train['type'], test_size=0.2)



print(len(train_x), len(val_x) )
#encoder = preprocessing.LabelEncoder()  ## For neural networks

#train_y = encoder.fit_transform(train_y)

#encoder.classes_

#val_y = encoder.transform(val_y)  ## For neural networks

#print(val_y[0])

#print(encoder.inverse_transform([val_y[0]]))
vec = TfidfVectorizer(max_df = 0.6, min_df = 0.01, max_features = 10000, analyzer = 'word', 

                      #ngram_range=(2,3),

                      use_idf=True, token_pattern=r'\w{1,}')

#tfidf = TfidfTransformer()

#clf = MultinomialNB()  ## eli5 not supported

#clf = linear_model.LogisticRegression()  ## Too slow

clf = linear_model.SGDClassifier(loss = 'log', max_iter=50, tol=1e-3) ## Logistic regession only



#pipe = make_pipeline(vec, tfidf, clf)

pipe = make_pipeline(vec, clf)



pipe.fit(train_x, train_y)
pred_y = pipe.predict(val_x)

report = metrics.classification_report(val_y, pred_y)

print(report)

print("accuracy: {:0.2f}".format(metrics.accuracy_score(val_y, pred_y)))

print("F1-score (weighted): {:0.2f}".format(metrics.f1_score(val_y, pred_y, average = 'weighted')))
labels = clf.classes_.tolist()

print(type(labels))

print(list(reversed(labels)))
'''

cm = confusion_matrix(val_y, pred_y, labels)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111) 

cax = ax.matshow(cm) 

plt.title('Confusion matrix of the classifier') 

fig.colorbar(cax) 

ax.set_xticklabels([''] + labels) 

ax.set_yticklabels([''] + labels) 

plt.xlabel('Predicted') 

plt.ylabel('True') 

plt.show()

####################

ax = plt.subplot()

sns.heatmap(cm, annot=False, ax = ax, cmap="Blues"); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(list(reversed(labels)))

plt.rcParams["figure.figsize"] = (27,25)

'''
eli5.show_weights(clf, vec=vec, top=10)

## Source : https://eli5.readthedocs.io/en/latest/_notebooks/debug-sklearn-text.html
i=25

print("\n".join(val_x.values[i].split("\n")[:10]))

print("Actual class : ", val_y.values[i], "\nPredicted class : ", pred_y[i])

eli5.show_prediction(clf, val_x.values[i], vec=vec)
val = pd.concat([val_x, val_y], axis=1)

val['pred'] = pred_y

print(val.head(10))
misclassified_examples = val[val.type != val.pred]

misclassified_examples.sample(10)
eli5.show_prediction(clf, misclassified_examples['content'].values[1], vec=vec)