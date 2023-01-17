import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

%matplotlib inline
data = pd.read_csv('../input/mbti_1.csv')
data.head()
mapping = {
    'I': 'Introversion',
    'E': 'Extroversion',
    'N': 'Intuition',
    'S': 'Sensing',
    'T': 'Thinking',
    'F': 'Feeling',
    'J': 'Judging',
    'P': 'Perceiving',
}
X = pd.DataFrame()
for c in 'INTJESFP':
    X[c] = data['type'].apply(lambda x: 1 if c in x else 0)
_ = X.sum().sort_values().rename(lambda x: mapping[x]).plot.barh()
cv = CountVectorizer(max_features=2000, strip_accents='ascii')
result = cv.fit_transform(data['posts'])
X = pd.concat([X, pd.DataFrame(result.toarray(), columns=['w_' + k for k in cv.vocabulary_.keys()])],
              axis=1)
wcols = [col for col in X.columns if col.startswith('w_') and len(col) > 5]
XX = X[wcols].T[X[wcols].mean() >= 0.5].T
def unique_words(a, b):
    (XX[X[a] == 1].mean() / XX[X[b] == 1].mean()).sort_values().rename(lambda x: x[2:]).tail(10).plot.barh()
    plt.title(mapping[a] + ' vs ' + mapping[b])
unique_words('E', 'I')
unique_words('I', 'E')
unique_words('N', 'S')
unique_words('S', 'N')
unique_words('T', 'F')
unique_words('F', 'T')
unique_words('J', 'P')
unique_words('P', 'J')