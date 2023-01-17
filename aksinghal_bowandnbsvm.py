import pandas as pd



train = pd.read_csv('/kaggle/input/cleaned/train_cleaned.csv')

test = pd.read_csv('/kaggle/input/cleaned/test_cleaned.csv')



print('Train shape:', train.shape)

print('Test shape:', test.shape)
train.head()
test.head()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    train['clean_text'], train['target'], shuffle=True, test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer



bv = CountVectorizer(max_features=50000, ngram_range=(1, 2))

train_data_features = bv.fit_transform(X_train)

val_data_features = bv.transform(X_val)
vocab = bv.get_feature_names()

print(vocab[:10])
# Calculate the ratio of feature f

def pr(y_i):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
import numpy as np



x = train_data_features # these are simply counts of unigrams and bigrams as a sparse matrix

y = y_train.values # targets



r = np.log(pr(1)/pr(0)) # probability matrix for each feature based on the training set

b = np.log((y==1).mean() / (y==0).mean()) # bias
r.shape, r
y_val_pre_preds = val_data_features @ r.T + b # multiply the probability matrix with the features in the validation set

y_val_preds = y_val_pre_preds.T > 0 # get disaster tweet predictions

(y_val_preds == y_val.values).mean() # estimate accuracy
x = train_data_features.sign() # binarize

r = np.log(pr(1)/pr(0))



y_val_pre_preds = val_data_features.sign() @ r.T + b 

y_val_preds = y_val_pre_preds.T>0

(y_val_preds == y_val.values).mean()
from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression(C=0.5)

logistic.fit(train_data_features, y_train)

y_val_preds = logistic.predict(val_data_features)

(y_val_preds==y_val).mean()
logistic = LogisticRegression(C=0.5)

logistic.fit(train_data_features.sign(), y_train)

y_val_preds = logistic.predict(val_data_features.sign())

(y_val_preds==y_val).mean()
import eli5



eli5.show_weights(logistic, vec=bv, top=25)
x = train_data_features

y = y_train.values



r = np.log(pr(1)/pr(0))

x_nb = x.multiply(r)



logistic = LogisticRegression(C=0.5)

logistic.fit(x_nb, y);



val_x_nb = val_data_features.multiply(r)

y_val_preds = logistic.predict(val_x_nb)

(y_val_preds.T==y_val.values).mean()
x = train_data_features.sign()

y = y_train.values



r = np.log(pr(1)/pr(0))

x_nb = x.multiply(r)



logistic = LogisticRegression(C=0.5)

logistic.fit(x_nb, y);



val_x_nb = val_data_features.sign().multiply(r)

y_val_preds = logistic.predict(val_x_nb)

(y_val_preds.T==y_val.values).mean()
eli5.show_weights(logistic, vec=bv, top=25)
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators = 200)

forest = forest.fit(train_data_features, y_train)
# Predict on training set

y_train_pred = forest.predict(train_data_features)



# Predict on validation set

y_val_pred = forest.predict(val_data_features)
(y_val_pred == y_val).mean()
logistic = LogisticRegression(C=0.5)

logistic.fit(train_data_features.sign(), y_train)

y_val_pred = logistic.predict(val_data_features.sign())
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



def plot_confusion_matrix(y_true, y_pred, ax, vmax=None,

                          normed=True, title='Confusion matrix'):

    cm = confusion_matrix(y_true, y_pred)

    if normed:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, vmax=vmax, annot=True, square=True, ax=ax, 

                cmap='Blues', cbar=False, linecolor='k',

               linewidths=1)

    ax.set_title(title, fontsize=16)

    ax.set_ylabel('True labels', fontsize=12)

    ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)
fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

plot_confusion_matrix(y_train, y_train_pred, ax=axis1, 

                      title='Confusion matrix (train data)')

plot_confusion_matrix(y_val, y_val_pred, ax=axis2, 

                      title='Confusion matrix (validation data)')
from sklearn.metrics import classification_report, f1_score



print('Classification report on Test set: \n', classification_report(y_val, y_val_pred))
y_val_probs = logistic.predict_proba(val_data_features.sign())[:, 1]



for thresh in np.arange(0.3, 0.5, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(y_val, (y_val_probs>thresh).astype(int))))
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

train_data_features = tfidf.fit_transform(X_train)

val_data_features = tfidf.transform(X_val)
vocab = tfidf.get_feature_names()

print(vocab[:10])
from sklearn.linear_model import LogisticRegression



logistic = LogisticRegression(C=0.5)

logistic.fit(train_data_features, y_train)

y_val_preds = logistic.predict(val_data_features)

(y_val_preds==y_val).mean()
import eli5



eli5.show_weights(logistic, vec=tfidf, top=25)
x = train_data_features

y = y_train.values



r = np.log(pr(1)/pr(0))

x_nb = x.multiply(r)



logistic = LogisticRegression(C=0.5)

logistic.fit(x_nb, y);



val_x_nb = val_data_features.multiply(r)

y_val_preds = logistic.predict(val_x_nb)

(y_val_preds.T==y_val.values).mean()
eli5.show_weights(logistic, vec=tfidf, top=25)
forest = RandomForestClassifier(n_estimators = 200)

forest = forest.fit(train_data_features, y_train)
# Predict on training set

y_train_pred = forest.predict(train_data_features)



# Predict on validation set

y_val_pred = forest.predict(val_data_features)
(y_val_pred == y_val).mean()
test = pd.read_csv('/kaggle/input/cleaned/test_cleaned.csv')



test.head()
test['clean_text']
from sklearn.feature_extraction.text import CountVectorizer



train_data_features = bv.fit_transform(X_train)

test['clean_text'] = test['clean_text'].apply(lambda x: str(x))

test_data_features = bv.transform(test['clean_text'])
logistic = LogisticRegression(C=0.5)

logistic.fit(train_data_features.sign(), y_train)

y_test_pred = logistic.predict(test_data_features.sign())
y_test_probs = logistic.predict_proba(test_data_features.sign())[:, 1]
(y_test_probs>0.45).sum()/3262
test['target'] = y_test_pred

test.head()