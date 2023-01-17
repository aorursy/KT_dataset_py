import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
word_vectorizer = CountVectorizer(stop_words = 'english',analyzer='word')
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
char_vectorizer = CountVectorizer(stop_words = 'english',analyzer='char')
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
losses = []
predictions = {'id': test['id']}
for class_name in class_names:
    train_target = train[class_name]
    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features=1000, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

    cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='f1_micro'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_features, train_target)
    predictions[class_name] = expit(logit(classifier.predict_proba(test_features)[:, 1]))

print('Total CV score is {}'.format(np.mean(losses)))
submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('submission.csv', index=False)