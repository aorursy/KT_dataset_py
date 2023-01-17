import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
import time
from sklearn.datasets import fetch_20newsgroups
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from tqdm.auto import tqdm
tqdm.pandas()
dataset = fetch_20newsgroups(subset='all')

X = pd.Series(dataset['data'])
y = pd.Series(dataset['target'])
fig, ax = plt.subplots(figsize=(20, 6))
fig.suptitle('Target Class Distribution', fontsize=24)
y.apply(lambda i: dataset['target_names'][i]).value_counts().plot.pie()
fig, ax = plt.subplots(figsize=(20, 6))
wordcloud = WordCloud(height=600, width=2000, stopwords=STOPWORDS).generate(' '.join((X.values)))
ax.imshow(wordcloud)
ax.set_title('Word Cloud', fontsize=24)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig, ax = plt.subplots(5, 4, figsize=(20, 14))
fig.suptitle('Class wise multiset of words', fontsize=24)
for i in range(20):
    u = Counter((' '.join((X[y != i].values))).split())
    a = Counter((' '.join((X[y == i].values))).split())
    wordcloud = WordCloud(height=600, width=1000, stopwords=STOPWORDS).generate_from_frequencies((a-u))
    ax[i//4][i%4].imshow(wordcloud)
    ax[i//4][i%4].set_title(f'{dataset["target_names"][i]}', fontsize=18)
    ax[i//4][i%4].xaxis.set_visible(False)
    ax[i//4][i%4].yaxis.set_visible(False)    
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(shuffle=True, random_state=19)

roc_auc_scores = []
time_taken = []
n_features = []

def train_model():
    model = DecisionTreeClassifier(class_weight='balanced', random_state=19)
    scores = []

    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X_vect[train_index], X_vect[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted'))

    feats = X_vect.shape[1]
    score = round(np.mean(scores), 5)*100
        
    return feats, score
start = time.time()

X_vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)).fit_transform(X)
feats, score = train_model()

end = time.time()
t = end - start

n_features.append(feats)
time_taken.append(t)
roc_auc_scores.append(score)

print('Mean score', score)
print('Features used', feats)
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

start = time.time()

X = X.progress_apply(lambda text: re.sub(r"\s+", " ", re.sub(r"[^A-Za-z]", " ", text)))
X = X.progress_apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
X_vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)).fit_transform(X)
feats, score = train_model()

end = time.time()
t = end - start

n_features.append(feats)
time_taken.append(t)
roc_auc_scores.append(score)

print('Mean score', score)
print('Features used', feats)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

start = time.time()

X = X.progress_apply(lambda x: ' '.join(wordnet_lemmatizer.lemmatize(word).lower() for word in x.split()))
X_vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)).fit_transform(X)
feats, score = train_model()

end = time.time()
t = end - start

n_features.append(feats)
time_taken.append(t)
roc_auc_scores.append(score)

print('Mean score', score)
print('Features used', feats)
class RemoveIrrelevantFeatures():

    def __init__(self, problem_type, random_state=None):
        self.problem_type = problem_type
        self.random_state = random_state
        if problem_type not in ['regression', 'classification']:
            raise Exception('Invalid problem type')
        
    def fit(self, X, y):
        if self.problem_type == 'regression':
            model = DecisionTreeRegressor(random_state=self.random_state)
        else:
            model = DecisionTreeClassifier(class_weight='balanced', random_state=self.random_state)
        model.fit(X, y)
        self.support = (model.feature_importances_ > 0)
        self.indices = np.where(self.support)[0]
        
    def transform(self, X, y=None):
        return X[:, self.indices]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_support(self):
        return self.support
start = time.time()

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))),
    ('feature_selector', RemoveIrrelevantFeatures(problem_type='classification', random_state=19)),
])
X_vect = pipeline.fit_transform(X, y)
feats, score = train_model()

end = time.time()
t = end - start

n_features.append(feats)
time_taken.append(t)
roc_auc_scores.append(score)

print('Mean score', score)
print('Features used', feats)
for i in range(3):
    n_features[i] -= n_features[i+1]
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax = ax.tolist()
labels = ['Noise', 'Un Normalized Text', 'Irrelevant Features', 'Important Features']
explode = (0, 0, 0, 0.1)
ax[0].pie(n_features, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax[0].set_title('Feature Analysis', fontsize=18)
ax[0].axis('equal')
ax[0].legend()


t = ['Raw Text', 'Cleaned Text', 'Normalized Text', 'Selected Features']

color = 'tab:orange'
ax[1].set_ylabel('Time Taken', color=color, fontsize=12)
ax[1].plot(t, time_taken, color=color)
ax[1].tick_params(axis='y', labelcolor=color)

ax.append(ax[1].twinx())

color = 'tab:blue'
ax[2].set_ylabel('AUROC', color=color, fontsize=12)
ax[2].plot(t, roc_auc_scores, color=color)
ax[2].tick_params(axis='y', labelcolor=color)

ax[1].set_title('Comparison: Time taken and AUROC', fontsize=18)
plt.show()
