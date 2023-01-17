import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

import glob

from eli5 import show_weights

import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence, text

from nltk import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

import matplotlib.pyplot as plt

import nltk

import sklearn

nltk.download('punkt')



import matplotlib.pyplot as plt

plt.style.use('ggplot')
BASE_DIR = '/kaggle/input/CORD-19-research-challenge/'

PMC = os.path.join(BASE_DIR, 'custom_license/custom_license')

META = os.path.join(BASE_DIR, 'metadata.csv')
meta_df = pd.read_csv(META)

meta_df.head()
from collections import defaultdict



def parse_json(path):

    parsed = defaultdict(str)

    with open(path) as file:

        content = json.load(file)

        parsed['paper_id'] = content['paper_id']

        parsed['abstract'] = '\n'.join([entry['text'] for entry in content['abstract']])

        parsed['body_text'] = '\n'.join([entry['text'] for entry in content['body_text']])

    return parsed





# Evaluation metric



def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota



def plot_confusion_matrix(cm, class_names,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names, rotation=45)

    plt.yticks(tick_marks, class_names)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
json_files = glob.glob(f'{BASE_DIR}/**/*.json', recursive=True)



def make_df(json_files):

    data = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [],\

             'journal': [], 'abstract_word_count':[], 'body_word_count':[]}

    ids = set()

    for i, path in enumerate(json_files):

        print('\r Processing article: {} of {}'.format(i, len(json_files)), end='')

        content = parse_json(os.path.join(PMC, path))



        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content['paper_id']]

        # no metadata or the id already in the data dictionary skip the rest of the loop

        if len(meta_data) == 0 or content['paper_id'] in ids:

            continue

            

        ids.add(content['paper_id'])

        data['paper_id'].append(content['paper_id'])

        data['abstract'].append(content['abstract'])

        data['body_text'].append(content['body_text'])

        data['abstract_word_count'].append(sum([True for i in content['abstract'].split()]))

        data['body_word_count'].append(sum([True for i in content['body_text'].split()]))



    

        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content['paper_id']]

        data['authors'].append(meta_data['authors'].values[0])

        data['title'].append(meta_data['title'].values[0])

        # add the journal information

        data['journal'].append(meta_data['journal'].values[0])



    return pd.DataFrame(data, columns=data.keys())

df_covid = make_df(json_files)

df_covid.head()
df_covid.to_csv('df_covid.csv', header=True, index=False)
df_covid = pd.read_csv('df_covid.csv', keep_default_na=False) # do not parse empty sting to nan

df_covid.head()
# function to print out classification model report

def classification_report(model_name, test, pred):

    from sklearn.metrics import precision_score, recall_score

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import f1_score

    

    print(model_name, ":\n")

    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")

    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='micro')) * 100), "%")

    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='micro')) * 100), "%")

    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='micro')) * 100), "%")
df = df_covid.sample(20000, random_state=42)
#split

xtrain, xtest = train_test_split(df['abstract'].values,

                                               random_state=42,

                                               test_size=0.2)



#vectorize

ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',

                     ngram_range=(2, 3), stop_words='english')



ctv.fit(list(xtrain)+list(xtest))

xtrain_ctv = ctv.transform(xtrain)

xtest_ctv = ctv.transform(xtest)
# from sklearn.cluster import KMeans



# k = 10

# kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)

# ytrain = kmeans.fit_predict(xtrain_ctv)



from sklearn.cluster import MiniBatchKMeans



k = 10

kmeans = MiniBatchKMeans(n_clusters=k,verbose=0, batch_size=100)

ytrain = kmeans.fit_predict(xtrain_ctv)
ytest = kmeans.fit_predict(xtest_ctv)
print('Number of labels: {}'.format(max(ytrain)+1))

class_dict = dict(zip(range(10), range(10)))

class_dict
#fit logistic regression on CountVectorizer

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0, max_iter=4000)

clf.fit(xtrain_ctv, ytrain)


show_weights(clf, target_names=class_dict, vec=ctv)
#make predictions calculate logloss

import itertools

predictions = clf.predict_proba(xtest_ctv)

#get the classes from probabilities

y_pred = []

for i in predictions:

    

    y_pred.append(i.argmax())

print('F1 Score: ',sklearn.metrics.f1_score(ytest, y_pred, average='weighted'))

print('Accuracy:', sklearn.metrics.accuracy_score(ytest, y_pred))

from sklearn.metrics import confusion_matrix

class_names = class_dict.values()

print(class_names)

cnf_matrix = confusion_matrix(ytest, y_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(9, 9))

plot_confusion_matrix(cnf_matrix, class_names, True,

                      title='Confusion matrix, with normalization')

plt.show()
