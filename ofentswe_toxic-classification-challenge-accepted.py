import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for visuals
import matplotlib.pyplot as plt # for plots
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
train.head()
train.shape
train.dtypes
train.isnull().any()
train.describe()
train.describe().plot(kind='bar')
sns.pairplot(train)
print(train.obscene.value_counts())
print(train.threat.value_counts())
print(train.insult.value_counts())
print(train.identity_hate.value_counts())
print(train.toxic.value_counts())
print(train.severe_toxic.value_counts())
fig, plots = plt.subplots(2,3,figsize=(18,12))
plot1, plot2, plot3, plot4, plot5, plot6 = plots.flatten()
sns.countplot(train['obscene'], palette= 'deep', ax = plot1)
sns.countplot(train['threat'], palette= 'muted', ax = plot2)
sns.countplot(train['insult'], palette = 'pastel', ax = plot3)
sns.countplot(train['identity_hate'], palette = 'dark', ax = plot4)
sns.countplot(train['toxic'], palette= 'colorblind', ax = plot5)
sns.countplot(train['severe_toxic'], palette= 'bright', ax = plot6)
structured_patterns = [
 (r'won\'t', 'will not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'i am'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')
]

class RegexpReplacer(object):
    def __init__(self, patterns=structured_patterns):
         self.patterns = [(re.compile(regex), repl) for (regex, repl) in
         patterns]
            
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
             s = re.sub(pattern, repl, s)
        return s

import re
def strip_symbols(text):
    return ' '.join(re.compile(r'\W+', re.UNICODE).split(text))
train.comment_text = train.comment_text.str.lower()
train.comment_text = train.comment_text.str.replace('\n',' ')
replacer = RegexpReplacer()
train.comment_text = train.comment_text.apply(lambda x:replacer.replace(x))
train.comment_text = train.comment_text.apply(lambda x:strip_symbols(x))
train.comment_text.head()
from wordcloud import WordCloud
wordcloud = WordCloud(width=1440, height=1080).generate(" ".join(train.comment_text.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(train.comment_text)
y = train.loc[:,'toxic':'identity_hate']
clf = BernoulliNB()
model = OneVsRestClassifier(clf)
model.fit(X, y)
## display the first few records for test dataset
test.head()
test.comment_text = test.comment_text.str.lower()
test.comment_text = test.comment_text.str.replace('\n',' ')
test.shape
test.comment_text = test.comment_text.apply(lambda x:replacer.replace(x))
test.comment_text = test.comment_text.apply(lambda x:strip_symbols(x))
X_test = vectorizer.transform(test.comment_text)
## remenber we have to make sure that the columns are the same not the rows 
print("X train shape : ",X.shape )
print("X test shape : ",X_test.shape)
probs = model.predict_proba(X_test)
submission.loc[:,'toxic':'identity_hate'] = probs
plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
sns.violinplot(x = 'toxic', y = 'insult', data = train[0:50000])
plt.subplot(1,2,2)
sns.distplot(submission[submission['identity_hate'] > 0.5]['identity_hate'][0:3000], color = 'green')
sns.distplot(submission[submission['identity_hate'] < 0.2 ]['identity_hate'][0:3000], color = 'red')
submission.to_csv('submission.csv', index=False)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(model.predict(X_test)[:,1], model.predict_proba(X_test)[:,1])
bernouli = roc_auc_score(model.predict(X_test)[:,1], model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Bernouli Naive Bayes (area = %0.2f)' % bernouli)
plt.plot([0,1], [0,1],label='Base Rate' 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
