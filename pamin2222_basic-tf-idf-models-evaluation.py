import numpy as np 

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "/kaggle/input"]).decode("utf8"))
training_text_path = "/kaggle/input/training_text"

test_text_path = "/kaggle/input/test_text"

training_variants_path = "/kaggle/input/training_variants"

test_variants = "/kaggle/input/test_variants"
train_variants_df = pd.read_csv(training_variants_path)

test_variants_df = pd.read_csv(test_variants)

train_text_df = pd.read_csv(training_text_path, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test_text_df = pd.read_csv(test_text_path, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_variants_df.head()
train_text_df.head()
merged_training_df = train_variants_df.merge(train_text_df, left_on="ID", right_on="ID")
merged_training_df.head()
merged_test_df = test_variants_df.merge(test_text_df, left_on="ID", right_on="ID")
merged_test_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(

    min_df=5, max_features=16000, strip_accents='unicode', lowercase=True,

    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 

    smooth_idf=True, sublinear_tf=True, stop_words = 'english'

)
tfidf_vectorizer.fit(merged_training_df['Text'])
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, accuracy_score
X_train_tfidfmatrix = tfidf_vectorizer.transform(merged_training_df['Text'].values)

X_test_tfidfmatrix = tfidf_vectorizer.transform(merged_test_df['Text'].values)



y_train = merged_training_df['Class'].values
def evaluate(X, y, clf=None):

    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits=5, random_state=8), 

                              n_jobs=-1, method='predict_proba', verbose=2)

    pred_indices = np.argmax(probas, axis=1)

    classes = np.unique(y)

    preds = classes[pred_indices]

    print('Log loss: {}'.format(log_loss(y, probas)))

    print('Accuracy: {}'.format(accuracy_score(y, preds)))
evaluate(X_train_tfidfmatrix, y_train, clf=LogisticRegression())
evaluate(X_train_tfidfmatrix, y_train, clf=XGBClassifier())
evaluate(X_train_tfidfmatrix, y_train, clf=AdaBoostClassifier())
evaluate(X_train_tfidfmatrix, y_train, clf=MultinomialNB())
evaluate(X_train_tfidfmatrix, y_train, clf=svm.SVC(probability=True))
clf = XGBClassifier()

clf.fit(X_train_tfidfmatrix, y_train)
y_test_predicted = clf.predict_proba(X_test_tfidfmatrix)
submission_df = pd.DataFrame(y_test_predicted, columns=['class' + str(c + 1) for c in range(9)])

submission_df['ID'] = merged_test_df['ID']
submission_df.head()
submission_df.to_csv('submission.csv', index=False)