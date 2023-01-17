import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['is_train'] = 1
test['is_train'] = 0

merge = pd.concat([train, test])
merge.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'id'], axis=1, inplace=True)

X_train, X_test, y_train, y_valid = train_test_split(merge, merge['is_train'], test_size=0.2, random_state=144)
print(X_train.shape)
print(X_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(ngram_range=(1, 2),
                            analyzer='word',
                            stop_words=None,
                            max_features=200000,
                            binary=True)
train_tfidf = tfidf_vec.fit_transform(X_train['comment_text'])
test_tfidf = tfidf_vec.transform(X_test['comment_text'])
print(train_tfidf.shape)
print(test_tfidf.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = LogisticRegression(solver='sag')
model.fit(train_tfidf, y_train)
pred_test_y = model.predict_proba(test_tfidf)[:, 1]
print('AUC of guessing test: {}'.format(roc_auc_score(y_valid, pred_test_y)))
X_test['pred'] = pred_test_y
X_test.head(20)
import eli5
eli5.show_weights(model, vec=tfidf_vec)
eli5.show_weights(model, vec=tfidf_vec, top=200)
tfidf_vec_char = TfidfVectorizer(ngram_range=(2, 6),
                                 analyzer='char',
                                 stop_words='english',
                                 max_features=200000,
                                 binary=True)
train_tfidf_char = tfidf_vec_char.fit_transform(X_train['comment_text'])
test_tfidf_char = tfidf_vec_char.transform(X_test['comment_text'])
print(train_tfidf_char.shape)
print(test_tfidf_char.shape)
char_model = LogisticRegression(solver='sag')
char_model.fit(train_tfidf_char, y_train)
pred_test_y2 = model.predict_proba(test_tfidf_char)[:, 1]
print('AUC of guessing test: {}'.format(roc_auc_score(y_valid, pred_test_y2)))
eli5.show_weights(char_model, vec=tfidf_vec_char, top=200)