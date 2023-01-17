import csv
import numpy as np

with open('../input/titanic-train.csv', 'rt') as f:
    data = list(csv.DictReader(f))
data[:1]
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

_all_xs = [{k: v for k, v in row.items() if k != 'Survived'} for row in data]
_all_ys = np.array([int(row['Survived']) for row in data])

all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
train_xs, valid_xs, train_ys, valid_ys = train_test_split(
    all_xs, all_ys, test_size=0.25, random_state=0)
print('{} items total, {:.1%} true'.format(len(all_xs), np.mean(all_ys)))
import warnings
# xgboost <= 0.6a2 shows a warning when used with scikit-learn 0.18+
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

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

evaluate(pipeline)
from eli5 import show_weights
show_weights(clf, vec=vec)
from eli5 import show_prediction
show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True)
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer

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
evaluate(pipeline2)
show_weights(clf2, vec=vec2)
from IPython.display import display

for idx in [4, 5, 7, 37, 81]:
    display(show_prediction(clf2, valid_xs[idx], vec=vec2,
                            show_feature_values=True))
