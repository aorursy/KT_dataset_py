import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
import re

%matplotlib inline
sns.set()

plt.style.use('ggplot')
data_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data_test =  pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sample_subm = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
data_train
data_train.describe(include='all').T
data_test.describe(include='all').T
x = data_train.target.value_counts()
sns.barplot(x.index,x)
plt.title("Total amount of fake and true tweets")
combine = [data_train,data_test]
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

example="New competition launched https://www.kaggle.scom/c/nlp-getting-started"
remove_url(example)
for dataset in combine:
    dataset['text'] = dataset['text'].apply(lambda x: remove_url(x))
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""

print(remove_html(example))
for dataset in combine:
    dataset['text'] = dataset['text'].apply(lambda x: remove_html(x))
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

example = "Omg another Earthquake 😔😔"
remove_emoji(example)
for dataset in combine:
    dataset['text'] = dataset['text'].apply(lambda x: remove_emoji(x))
import string

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="That was simple: Gorgeous!"
remove_punct(example)
for dataset in combine:
    dataset['text'] = dataset['text'].apply(lambda x: remove_punct(x))
!pip install pyspellchecker
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

text = "corect me plese" 
correct_spellings(text)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
data_train['text'][1]
data_train['target'][1]
data_train['text']
X_train,X_test = data_train['text'], data_test['text']
y = data_train['target']

X_test.shape,X_train.shape
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=5,max_df=0.8,sublinear_tf=True,stop_words = 'english')
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
train_vectors.shape,test_vectors.shape
X_train, X_valid, y_train, y_valid = train_test_split(train_vectors, y, test_size=0.1, random_state=17)
X_train.shape, y_train.shape
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
logit = LogisticRegression(C=1,n_jobs=-1,random_state=17,class_weight='balanced')
logit.fit(X_train,y_train)
f1_score(y_valid,logit.predict(X_valid))
confusion_matrix(y_valid,logit.predict(X_valid))
rf = RandomForestClassifier(n_jobs=-1,random_state=17,class_weight='balanced')
rf.fit(X_train,y_train)
f1_score(y_valid,rf.predict(X_valid))
confusion_matrix(y_valid,rf.predict(X_valid))
from sklearn.model_selection import cross_val_score,GridSearchCV,StratifiedKFold,learning_curve
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=17)
C_s =  np.logspace(-1,1,100)
logit_cv = LogisticRegressionCV(Cs=C_s,cv=skf,scoring='f1',n_jobs=-1,random_state=17,verbose=1)
logit_cv.fit(X_train,y_train)
f1_score(y_valid,logit_cv.predict(X_valid)),f1_score(y_train,logit_cv.predict(X_train))
params = {'n_estimators':[70,100,150],'max_features':['auto', 'sqrt'],
          'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)],
        'min_samples_leaf':[1,2,4]}
rf_grid = GridSearchCV(RandomForestClassifier(),param_grid=params,cv=skf,n_jobs=-1,verbose=2,scoring='f1')
rf_grid.fit(X_train,y_train)
score1acc,score1f1 = rf_grid.best_score_,f1_score(y_valid,rf_grid.best_estimator_.predict(X_valid))
score1acc,score1f1
rf_grid.best_params_
rf = RandomForestClassifier(max_depth=180,max_features='sqrt',min_samples_leaf=6,n_estimators=200,random_state=17)
rf.fit(X_train,y_train)
f1_score(y_valid,rf.predict(X_valid)), f1_score(y_train,rf.predict(X_train))
confusion_matrix(y_valid, rf.predict(X_valid))
y_pred_RF = rf.predict(test_vectors)
sample_subm["target"] = y_pred_RF
sample_subm.to_csv("submissionRF.csv",index=False)
sample_subm
y_pred_LR = logit_cv.predict(test_vectors)
sample_subm["target"] = y_pred_LR
sample_subm.to_csv("submissionLR.csv",index=False)
X_train,X_test = data_train['text'], data_test['text']
y = data_train['target']

X_test.shape,X_train.shape
X_for_tfidf = pd.concat([X_train,X_test])
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=5,max_df=0.8,sublinear_tf=True,stop_words = 'english')
vectorizer.fit(X_for_tfidf)
X = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
X.shape,X_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=17)
logit = LogisticRegression(C=1.78,random_state=17)
logit.fit(X_train,y_train)
f1_score(y_valid,logit.predict(X_valid)),f1_score(y_train,logit.predict(X_train))
y_pred_LR = logit.predict(X_test)
sample_subm["target"] = y_pred_LR
sample_subm.to_csv("submissionLR.csv",index=False)
from sklearn.svm import SVC
svc = SVC(C=1,kernel='rbf')
svc.fit(X_train,y_train)
f1_score(y_valid,svc.predict(X_valid)),f1_score(y_train,svc.predict(X_train))
params = {'C':np.logspace(-1,1,10)}
svc_grid = GridSearchCV(SVC(),param_grid=params,cv=skf,n_jobs=-1,verbose=2,scoring='f1')
svc_grid.fit(X_train,y_train)
svc_grid.best_params_
svc = SVC(C=1.17,kernel='rbf')
svc.fit(X_train,y_train)
f1_score(y_valid,svc.predict(X_valid)),f1_score(y_train,svc.predict(X_train))
confusion_matrix(y_valid,svc.predict(X_valid))
y_pred_SVC = svc.predict(X_test)
sample_subm["target"] = y_pred_SVC
sample_subm.to_csv("submissionSVC.csv",index=False)
