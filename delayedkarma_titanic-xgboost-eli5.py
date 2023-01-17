import csv
import numpy as np
import pandas as pd

from datetime import datetime as dt

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error

from eli5 import show_weights, show_prediction
with open('../input/train.csv', 'rt') as f:
    data_train = list(csv.DictReader(f))
data_train[:1]
with open('../input/test.csv', 'rt') as f:
    data_test = list(csv.DictReader(f))
data_test[:1]
_all_xs = [{k: v for k, v in row.items() if k != 'Survived'} for row in data_train]
_all_ys = np.array([int(row['Survived']) for row in data_train])

all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
train_xs, valid_xs, train_ys, valid_ys = train_test_split(
    all_xs, all_ys, test_size=0.25, random_state=0)

print('{} items total, {:.1%} true'.format(len(all_xs), np.mean(all_ys)))
for x in all_xs:
    if x['Age']:
        x['Age'] = float(x['Age'])
    else:
        x.pop('Age')
    x['Fare'] = float(x['Fare'])
    x['SibSp'] = int(x['SibSp'])
    x['Parch'] = int(x['Parch'])
### Load the test set, do the basic pre-processing steps same as above, and make predictions on it.

all_xs_test = [{k: v for k, v in row.items()} for row in data_test]

for x in all_xs_test:
    if x['Age']:
        x['Age'] = float(x['Age'])
    else:
        x.pop('Age')
    if x['Fare']:
        x['Fare'] = float(x['Fare'])
    else:
        x.pop('Fare')
    x['SibSp'] = int(x['SibSp'])
    x['Parch'] = int(x['Parch'])
class CSCTransformer:
    def transform(self, xs):
        # work around https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
        return xs.tocsc()
    def fit(self, *args):
        return self

clf = XGBClassifier()
vec = DictVectorizer()
pipeline = make_pipeline(vec, CSCTransformer(), clf)

def evaluate(_clf):
    scores = cross_val_score(_clf, all_xs, all_ys, scoring='accuracy', cv=10)
    print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(scores), 2 * np.std(scores)))
    _clf.fit(train_xs, train_ys)  # so that parts of the original pipeline are fitted
    return np.mean(scores), _clf.predict(all_xs_test)

score_1, preds_1 = evaluate(pipeline)
show_weights(clf, vec=vec)
show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True)
no_missing = lambda feature_name, feature_value: not np.isnan(feature_value)
show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True, feature_filter=no_missing)
vec2 = FeatureUnion([
    ('Name', CountVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 4),
        preprocessor=lambda x: x['Name'],
        max_features=100,
    )),
    ('All', DictVectorizer()),
])
clf2 = XGBClassifier()

pipeline2 = make_pipeline(vec2, CSCTransformer(), clf2)

score_2, preds_2 = evaluate(pipeline2)
show_weights(clf2, vec=vec2)
# We hide missing features 
for idx in [4, 5, 7, 37, 81]:
    display(show_prediction(clf2, valid_xs[idx], vec=vec2,
                            show_feature_values=True, feature_filter=no_missing))
df_sub = pd.read_csv('../input/gender_submission.csv')

filename = 'subm_{:.6f}_{}.csv'.format(score_2, 
                     dt.now().strftime('%Y-%m-%d-%H-%M'))
print('save to {}'.format(filename))

submission = pd.DataFrame()
submission['PassengerId'] = df_sub['PassengerId']
submission['Survived'] = preds_2

submission.to_csv(filename, index=False)