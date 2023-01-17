# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "classification"



def save_fig(fig_id, tight_layout=True):

    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)
def sort_by_target(mnist):

    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]

    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]

    mnist.data[:60000] = mnist.data[reorder_train]

    mnist.target[:60000] = mnist.target[reorder_train]

    mnist.data[60000:] = mnist.data[reorder_test + 60000]

    mnist.target[60000:] = mnist.target[reorder_test + 60000]
try:

    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1, cache=True)

    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings

    sort_by_target(mnist) # fetch_openml() returns an unsorted dataset

except ImportError:

    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata('MNIST original')

mnist["data"], mnist["target"]
mnist.data.shape
X, y = mnist["data"], mnist["target"]

X.shape
y.shape
28*28
some_digit = X[36000]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary,

           interpolation="nearest")

plt.axis("off")



save_fig("some_digit_plot")

plt.show()
def plot_digit(data):

    image = data.reshape(28, 28)

    plt.imshow(image, cmap = mpl.cm.binary,

               interpolation="nearest")

    plt.axis("off")
# EXTRA

def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = mpl.cm.binary, **options)

    plt.axis("off")
plt.figure(figsize=(9,9))

example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]

plot_digits(example_images, images_per_row=10)

save_fig("more_digits_plot")

plt.show()
y[36000]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
import numpy as np



shuffle_index = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)

y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)

sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone



skfolds = StratifiedKFold(n_splits=3, random_state=42)



for train_index, test_index in skfolds.split(X_train, y_train_5):

    clone_clf = clone(sgd_clf)

    X_train_folds = X_train[train_index]

    y_train_folds = (y_train_5[train_index])

    X_test_fold = X_train[test_index]

    y_test_fold = (y_train_5[test_index])



    clone_clf.fit(X_train_folds, y_train_folds)

    y_pred = clone_clf.predict(X_test_fold)

    n_correct = sum(y_pred == y_test_fold)

    print(n_correct / len(y_pred))
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self, X):

        return np.zeros((len(X), 1), dtype=bool)
never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.model_selection import cross_val_predict



y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_train_5, y_train_pred)
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)
from sklearn.metrics import precision_score, recall_score



precision_score(y_train_5, y_train_pred)
4344 / (4344 + 1307)
recall_score(y_train_5, y_train_pred)
4344 / (4344 + 1077)
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
4344 / (4344 + (1077 + 1307)/2)
y_scores = sgd_clf.decision_function([some_digit])

y_scores
threshold = 0

y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
threshold = 200000

y_some_digit_pred = (y_scores > threshold)

y_some_digit_pred
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,

                             method="decision_function")
y_scores.shape
# hack to work around issue #9589 in Scikit-Learn 0.19.0

if y_scores.ndim == 2:

    y_scores = y_scores[:, 1]
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.xlabel("Threshold", fontsize=16)

    plt.legend(loc="upper left", fontsize=16)

    plt.ylim([0, 1])



plt.figure(figsize=(8, 4))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.xlim([-700000, 700000])

save_fig("precision_recall_vs_threshold_plot")

plt.show()
(y_train_pred == (y_scores > 0)).all()
y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)
def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])



plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions, recalls)

save_fig("precision_vs_recall_plot")

plt.show()
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)



plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr)

save_fig("roc_curve_plot")

plt.show()
from sklearn.metrics import roc_auc_score



roc_auc_score(y_train_5, y_scores)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,

                                    method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plt.legend(loc="lower right", fontsize=16)

save_fig("roc_curve_comparison_plot")

plt.show()
roc_auc_score(y_train_5, y_scores_forest)
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)

precision_score(y_train_5, y_train_pred_forest)
recall_score(y_train_5, y_train_pred_forest)
sgd_clf.fit(X_train, y_train)

sgd_clf.predict([some_digit])
some_digit_scores = sgd_clf.decision_function([some_digit])

some_digit_scores
np.argmax(some_digit_scores)
sgd_clf.classes_
sgd_clf.classes_[5]
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42))

ovo_clf.fit(X_train, y_train)

ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)
forest_clf.fit(X_train, y_train)

forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)

conf_mx
def plot_confusion_matrix(matrix):

    """If you prefer color and a colorbar"""

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)

    cax = ax.matshow(matrix)

    fig.colorbar(cax)
plt.matshow(conf_mx, cmap=plt.cm.gray)

save_fig("confusion_matrix_plot", tight_layout=False)

plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)

norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

save_fig("confusion_matrix_errors_plot", tight_layout=False)

plt.show()
cl_a, cl_b = 3, 5

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]

X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]

X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]

X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]



plt.figure(figsize=(8,8))

plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)

plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)

plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)

plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

save_fig("error_analysis_digits_plot")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



y_train_large = (y_train >= 7)

y_train_odd = (y_train % 2 == 1)

y_multilabel = np.c_[y_train_large, y_train_odd]



knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)

f1_score(y_multilabel, y_train_knn_pred, average="macro")
noise = np.random.randint(0, 100, (len(X_train), 784))

X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))

X_test_mod = X_test + noise

y_train_mod = X_train

y_test_mod = X_test
some_index = 5500

plt.subplot(121); plot_digit(X_test_mod[some_index])

plt.subplot(122); plot_digit(y_test_mod[some_index])

save_fig("noisy_digit_example_plot")

plt.show()
knn_clf.fit(X_train_mod, y_train_mod)

clean_digit = knn_clf.predict([X_test_mod[some_index]])

plot_digit(clean_digit)

save_fig("cleaned_digit_example_plot")
from sklearn.dummy import DummyClassifier

dmy_clf = DummyClassifier()

y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_dmy = y_probas_dmy[:, 1]
fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)

plot_roc_curve(fprr, tprr)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)

knn_clf.fit(X_train, y_train)
y_knn_pred = knn_clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_knn_pred)
from scipy.ndimage.interpolation import shift

def shift_digit(digit_array, dx, dy, new=0):

    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)



plot_digit(shift_digit(some_digit, 5, 1, new=100))
X_train_expanded = [X_train]

y_train_expanded = [y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)

    X_train_expanded.append(shifted_images)

    y_train_expanded.append(y_train)



X_train_expanded = np.concatenate(X_train_expanded)

y_train_expanded = np.concatenate(y_train_expanded)

X_train_expanded.shape, y_train_expanded.shape
knn_clf.fit(X_train_expanded, y_train_expanded)
y_knn_expanded_pred = knn_clf.predict(X_test)
accuracy_score(y_test, y_knn_expanded_pred)
ambiguous_digit = X_test[2589]

knn_clf.predict_proba([ambiguous_digit])
plot_digit(ambiguous_digit)
from sklearn.model_selection import GridSearchCV



param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]



knn_clf = KNeighborsClassifier()

grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)

grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
from sklearn.metrics import accuracy_score



y_pred = grid_search.predict(X_test)

accuracy_score(y_test, y_pred)
from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):

    image = image.reshape((28, 28))

    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape([-1])
image = X_train[1000]

shifted_image_down = shift_image(image, 0, 5)

shifted_image_left = shift_image(image, -5, 0)



plt.figure(figsize=(12,3))

plt.subplot(131)

plt.title("Original", fontsize=14)

plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(132)

plt.title("Shifted down", fontsize=14)

plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(133)

plt.title("Shifted left", fontsize=14)

plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.show()
X_train_augmented = [image for image in X_train]

y_train_augmented = [label for label in y_train]



for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

    for image, label in zip(X_train, y_train):

        X_train_augmented.append(shift_image(image, dx, dy))

        y_train_augmented.append(label)



X_train_augmented = np.array(X_train_augmented)

y_train_augmented = np.array(y_train_augmented)
shuffle_idx = np.random.permutation(len(X_train_augmented))

X_train_augmented = X_train_augmented[shuffle_idx]

y_train_augmented = y_train_augmented[shuffle_idx]
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)

accuracy_score(y_test, y_pred)
import os



TITANIC_PATH = os.path.join("datasets", "titanic")
import pandas as pd



def load_titanic_data(filename, titanic_path=TITANIC_PATH):

    csv_path = os.path.join(titanic_path, filename)

    return pd.read_csv(csv_path)
train_data = load_titanic_data("train.csv")

test_data = load_titanic_data("test.csv")
train_data.head()
train_data.info()
train_data.describe()
train_data["Survived"].value_counts()
train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()
from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

try:

    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

except ImportError:

    from sklearn.preprocessing import Imputer as SimpleImputer



num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

    ])
num_pipeline.fit_transform(train_data)
# Inspired from stackoverflow.com/questions/25239958

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
try:

    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20

    from sklearn.preprocessing import OneHotEncoder

except ImportError:

    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20
cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])
cat_pipeline.fit_transform(train_data)
from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])
X_train = preprocess_pipeline.fit_transform(train_data)

X_train
y_train = train_data["Survived"]
from sklearn.svm import SVC



svm_clf = SVC(gamma="auto")

svm_clf.fit(X_train, y_train)
X_test = preprocess_pipeline.transform(test_data)

y_pred = svm_clf.predict(X_test)
from sklearn.model_selection import cross_val_score



svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()
plt.figure(figsize=(8, 4))

plt.plot([1]*10, svm_scores, ".")

plt.plot([2]*10, forest_scores, ".")

plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))

plt.ylabel("Accuracy", fontsize=14)

plt.show()
train_data["AgeBucket"] = train_data["Age"] // 15 * 15

train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]

train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
import os

import tarfile

from six.moves import urllib



DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"

HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"

SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"

SPAM_PATH = os.path.join("datasets", "spam")



def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):

    if not os.path.isdir(spam_path):

        os.makedirs(spam_path)

    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):

        path = os.path.join(spam_path, filename)

        if not os.path.isfile(path):

            urllib.request.urlretrieve(url, path)

        tar_bz2_file = tarfile.open(path)

        tar_bz2_file.extractall(path=SPAM_PATH)

        tar_bz2_file.close()
fetch_spam_data()
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")

SPAM_DIR = os.path.join(SPAM_PATH, "spam")

ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]

spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]
len(ham_filenames)
len(spam_filenames)
import email

import email.policy



def load_email(is_spam, filename, spam_path=SPAM_PATH):

    directory = "spam" if is_spam else "easy_ham"

    with open(os.path.join(spam_path, directory, filename), "rb") as f:

        return email.parser.BytesParser(policy=email.policy.default).parse(f)
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]

spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
print(ham_emails[1].get_content().strip())
print(spam_emails[6].get_content().strip())
def get_email_structure(email):

    if isinstance(email, str):

        return email

    payload = email.get_payload()

    if isinstance(payload, list):

        return "multipart({})".format(", ".join([

            get_email_structure(sub_email)

            for sub_email in payload

        ]))

    else:

        return email.get_content_type()
from collections import Counter



def structures_counter(emails):

    structures = Counter()

    for email in emails:

        structure = get_email_structure(email)

        structures[structure] += 1

    return structures
structures_counter(ham_emails).most_common()
structures_counter(spam_emails).most_common()
for header, value in spam_emails[0].items():

    print(header,":",value)
spam_emails[0]["Subject"]
import numpy as np

from sklearn.model_selection import train_test_split



X = np.array(ham_emails + spam_emails)

y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import re

from html import unescape



def html_to_plain_text(html):

    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)

    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)

    text = re.sub('<.*?>', '', text, flags=re.M | re.S)

    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)

    return unescape(text)
html_spam_emails = [email for email in X_train[y_train==1]

                    if get_email_structure(email) == "text/html"]

sample_html_spam = html_spam_emails[7]

print(sample_html_spam.get_content().strip()[:1000], "...")
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
def email_to_text(email):

    html = None

    for part in email.walk():

        ctype = part.get_content_type()

        if not ctype in ("text/plain", "text/html"):

            continue

        try:

            content = part.get_content()

        except: # in case of encoding issues

            content = str(part.get_payload())

        if ctype == "text/plain":

            return content

        else:

            html = content

    if html:

        return html_to_plain_text(html)
print(email_to_text(sample_html_spam)[:100], "...")
try:

    import nltk



    stemmer = nltk.PorterStemmer()

    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):

        print(word, "=>", stemmer.stem(word))

except ImportError:

    print("Error: stemming requires the NLTK module.")

    stemmer = None
try:

    import urlextract # may require an Internet connection to download root domain names

    

    url_extractor = urlextract.URLExtract()

    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))

except ImportError:

    print("Error: replacing URLs requires the urlextract module.")

    url_extractor = None
from sklearn.base import BaseEstimator, TransformerMixin



class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,

                 replace_urls=True, replace_numbers=True, stemming=True):

        self.strip_headers = strip_headers

        self.lower_case = lower_case

        self.remove_punctuation = remove_punctuation

        self.replace_urls = replace_urls

        self.replace_numbers = replace_numbers

        self.stemming = stemming

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        X_transformed = []

        for email in X:

            text = email_to_text(email) or ""

            if self.lower_case:

                text = text.lower()

            if self.replace_urls and url_extractor is not None:

                urls = list(set(url_extractor.find_urls(text)))

                urls.sort(key=lambda url: len(url), reverse=True)

                for url in urls:

                    text = text.replace(url, " URL ")

            if self.replace_numbers:

                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)

            if self.remove_punctuation:

                text = re.sub(r'\W+', ' ', text, flags=re.M)

            word_counts = Counter(text.split())

            if self.stemming and stemmer is not None:

                stemmed_word_counts = Counter()

                for word, count in word_counts.items():

                    stemmed_word = stemmer.stem(word)

                    stemmed_word_counts[stemmed_word] += count

                word_counts = stemmed_word_counts

            X_transformed.append(word_counts)

        return np.array(X_transformed)
X_few = X_train[:3]

X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)

X_few_wordcounts
from scipy.sparse import csr_matrix



class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):

        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):

        total_count = Counter()

        for word_count in X:

            for word, count in word_count.items():

                total_count[word] += min(count, 10)

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.most_common_ = most_common

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}

        return self

    def transform(self, X, y=None):

        rows = []

        cols = []

        data = []

        for row, word_count in enumerate(X):

            for word, count in word_count.items():

                rows.append(row)

                cols.append(self.vocabulary_.get(word, 0))

                data.append(count)

        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)

X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)

X_few_vectors
X_few_vectors.toarray()
vocab_transformer.vocabulary_
from sklearn.pipeline import Pipeline



preprocess_pipeline = Pipeline([

    ("email_to_wordcount", EmailToWordCounterTransformer()),

    ("wordcount_to_vector", WordCounterToVectorTransformer()),

])



X_train_transformed = preprocess_pipeline.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



log_clf = LogisticRegression(solver="liblinear", random_state=42)

score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)

score.mean()
from sklearn.metrics import precision_score, recall_score



X_test_transformed = preprocess_pipeline.transform(X_test)



log_clf = LogisticRegression(solver="liblinear", random_state=42)

log_clf.fit(X_train_transformed, y_train)



y_pred = log_clf.predict(X_test_transformed)



print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))