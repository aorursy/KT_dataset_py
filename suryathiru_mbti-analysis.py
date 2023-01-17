import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import spacy
import re
from pprint import pprint

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/mbti_1.csv')
personalities = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
data.head()
posts_len = data['posts'].apply(len)
ranges = pd.cut(posts_len, 10, labels=np.arange(1, 11)) # split length into ranges (1-1000, 1001-2000)
cnt = ranges.value_counts()

plt.figure(figsize=(10,5))
sns.barplot(cnt.index, cnt.values)
plt.xlabel('x1000 words')
plt.ylabel('no of examples')
plt.title('no of examples in each range of post length')

print('Average post length: ', posts_len.mean()) # can be used to decide the no of features we should consider
cnt = data.groupby(['type'])['posts'].count()
pie = go.Pie(labels=cnt.index, values=cnt.values)
fig = go.Figure(data=[pie])
py.iplot(fig)
def replace_symbols(text):
    text = re.sub('\|\|\|', ' ', text)
    text = re.sub('https?\S+', '<URL>', text)
    return text

data['cleaned_posts'] = data['posts'].apply(replace_symbols)
from wordcloud import WordCloud, STOPWORDS

STOPWORDS.add('URL') # words to not consider
labels = data['type'].unique()
row, col = 4, 4
wc = WordCloud(stopwords=STOPWORDS)

fig, ax = plt.subplots(4, 4, figsize=(20,15))

for i in range(4):
    for j in range(4):
        cur_type = labels[i*col+j]
        cur_ax = ax[i][j]
        df = data[data['type'] == cur_type]
        wordcloud = wc.generate(df['cleaned_posts'].to_string())
        cur_ax.imshow(wordcloud)
        cur_ax.axis('off')
        cur_ax.set_title(cur_type)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score # better metric due to small frequence of date for few types
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
type_enc = LabelEncoder()
type_enc.fit(data['type'])
type_enc.classes_
nlp = spacy.load('en_core_web_sm')
def tokenizer(text): # slowed the traning heavily
    doc = nlp(text)
    # preprocess during tokenizing
    tokens = [token.lemma_ for token in doc 
              if not (token.is_stop or token.is_digit or token.is_quote or token.is_space
                     or token.is_punct or token.is_bracket)]    
    return tokens

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
clf = LogisticRegression()

pipe_lr = Pipeline([('tfidf', tfidf), ('lgr', clf)])
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scoring = {'acc': 'accuracy', 'f1': 'f1_micro'}
result = cross_validate(pipe_lr, data['cleaned_posts'], type_enc.transform(data['type']), scoring=scoring,
                        cv=kfolds, n_jobs=-1, verbose=1)
print('Logistic regression model performance:')
pprint(result)

for key in result:
    print(key + ' : ', result[key].mean())
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=20)

pipe_rf = Pipeline([('tfidf', tfidf), ('rf', clf)])
scoring = {'acc': 'accuracy', 'f1': 'f1_micro'}
result = cross_validate(pipe_rf, data['cleaned_posts'], type_enc.transform(data['type']), scoring=scoring,
                        cv=kfolds, n_jobs=-1, verbose=1)
print('Random forest model performance:')
pprint(result)

for key in result:
    print(key + ' : ', result[key].mean())
import tensorflow as tf
from sklearn.model_selection import train_test_split
X = tfidf.fit_transform(data['cleaned_posts']).toarray()
Y = type_enc.transform(data['type'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
# def input_fn(features, labels, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices(({'x': features}, labels))
#     return dataset.repeat().batch(batch_size)

# feature_cols = [tf.feature_column.numeric_column(key='x', shape=[5000])]
def get_inp_fn(dataset, targets, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={'x': dataset},
        y=np.array(targets).astype(np.int32),
        num_epochs=num_epochs,
        shuffle=shuffle
    )

feature_cols = [tf.feature_column.numeric_column('x', shape=[5000])]
# run_config = tf.estimator.RunConfig(save_summary_steps=None, save_checkpoints_secs=None)
# run_config = tf.estimator.RunConfig(keep_checkpoint_max=1, save_summary_steps=None, 
#                                     save_checkpoints_steps=1000, save_checkpoints_secs=None)

clf = tf.estimator.DNNClassifier(
    feature_columns=feature_cols,
    hidden_units=[1024, 512],
    n_classes=16,
    optimizer='Adam',
    dropout=0.2
)
# clf.train(input_fn=lambda: input_fn(X_train, Y_train, 64), steps=1000)
clf.train(input_fn=get_inp_fn(X_train, Y_train), steps=2000)
# clf.evaluate(input_fn=lambda: input_fn(X_train, Y_train, 64))
result = clf.evaluate(input_fn=get_inp_fn(X_train, Y_train, 1, False))
print('Train set evaluation:')
pprint(result)

result = clf.evaluate(input_fn=get_inp_fn(X_test, Y_test, 1, False))
print('Test set evaluation:')
pprint(result)
