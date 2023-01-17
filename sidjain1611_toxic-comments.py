import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import seaborn as sns

import re
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

subm = pd.read_csv('../input/sample_submission.csv')
train.head()
len_tr=train.comment_text.str.len()

sns.distplot(len_tr)
labeled_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[labeled_cols].max(axis=1)

train.describe()
train.isnull().sum()
test.isnull().sum()
def clean_text(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text
train['comment_text'] = train['comment_text'].map(lambda com : clean_text(com))
test['comment_text'] = test['comment_text'].map(lambda com : clean_text(com))
X = train.comment_text

test_X = test.comment_text
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_features=5000,stop_words='english')

X_dtm = vect.fit_transform(X)

test_X_dtm = vect.transform(test_X)
# import and instantiate the Logistic Regression model

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=12.0)



# create submission file

submission_binary = pd.read_csv('../input/sample_submission.csv')



for label in labeled_cols:

    print('... Processing {}'.format(label))

    y = train[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm, y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm)

    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))

    # compute the predicted probabilities for X_test_dtm

    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

    submission_binary[label] = test_y_prob
submission_binary.head()
submission_binary.to_csv('submission_binary.csv',index=False)
# create submission file

submission_chains = pd.read_csv('../input/sample_submission.csv')

# create a function to add features

def add_feature(X, feature_to_add):

    '''

    Returns sparse feature matrix with added feature.

    feature_to_add can also be a list of features.

    '''

    from scipy.sparse import csr_matrix, hstack

    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
for label in labeled_cols:

    print('... Processing {}'.format(label))

    y = train[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm,y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm)

    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))

    # make predictions from test_X

    test_y = logreg.predict(test_X_dtm)

    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

    submission_chains[label] = test_y_prob

    # chain current label to X_dtm

    X_dtm = add_feature(X_dtm, y)

    print('Shape of X_dtm is now {}'.format(X_dtm.shape))

    # chain current label predictions to test_X_dtm

    test_X_dtm = add_feature(test_X_dtm, test_y)

    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))
submission_chains.head()
# generate submission file

submission_chains.to_csv('submission_chains.csv', index=False)
# create submission file

submission_combined = pd.read_csv('../input/sample_submission.csv')
for label in labeled_cols:

    submission_combined[label] = 0.5*(submission_chains[label]+submission_binary[label])
submission_combined.head()
# generate submission file

submission_combined.to_csv('submission_combined.csv', index=False)