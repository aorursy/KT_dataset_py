import json
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
with open('../input/train.json', 'r') as f:
    train = json.loads(f.read())
with open('../input/test.json', 'r') as f:
    test = json.loads(f.read())

train = pd.DataFrame(train)
test = pd.DataFrame(test)
def to_str(x):
    return ', '.join(x)
train['ingredients'] = train['ingredients'].apply(to_str)
test['ingredients'] = test['ingredients'].apply(to_str)
th = train.shape[0]
all_data = train['ingredients'].append(test['ingredients'])
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(dtype = np.float32, ngram_range = (1, 2))
sparse_all_data = cv.fit_transform(all_data)
X_train = sparse_all_data[:th]
y_train = train['cuisine']
X_test = sparse_all_data[th:]
%%time
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    LinearSVC(max_iter = 10000, class_weight  = 'balanced', C=0.05),
    X_train,
    y_train,
    scoring='f1_macro',
    cv=5,
)
print('classifier: f1_macro={}, cv_scores={}'.format(cv_scores.mean(), cv_scores))
%%time
model = LinearSVC(max_iter = 10000, class_weight  = 'balanced', C=0.05).fit(X_train, y_train)
y_test = model.predict(X_test)
sub = pd.DataFrame({'id': test['id'], 'cuisine': y_test}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)
