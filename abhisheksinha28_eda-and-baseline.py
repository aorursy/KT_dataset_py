# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.info()
train.tail()
rowsum = train.iloc[:,2:].sum(axis=1)
train['clean'] = (rowsum==0)
train['clean'].sum()

total = train.iloc[:,2:].sum()
total
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.plotly as py
import plotly.graph_objs as go


plt.figure(figsize=(10,8))
sns.barplot(total.index, total.values, palette= 'dark' )
plt.title('Class Frequency')
multi_tags = rowsum.value_counts()
multi_tags
plt.figure(figsize=(8,4))
ax = sns.barplot(multi_tags.index, multi_tags.values,palette='dark')
plt.title('Number of multi-tags in a comment') ; plt.xlabel('Number of tags'); plt.ylabel('Number of comments');
rects = ax.patches
labels = multi_tags.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

temp = train.iloc[:,2:-1]
corr = temp.corr()
plt.figure(figsize=(10,8))

sns.heatmap(corr,annot= True,xticklabels=corr.columns.values,yticklabels=corr.columns.values,)
import scipy.stats as ss
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
col1="toxic"
col2="severe_toxic"
confusion_matrix = pd.crosstab(temp[col1], temp[col2])
print("Confusion matrix between toxic and severe toxic:")
print(confusion_matrix)
new_corr=cramers_corrected_stat(confusion_matrix)
print("The correlation between Toxic and Severe toxic using Cramer's stat=",new_corr)
print("Some examples : \n")
print("Toxic : \n")
print("\n1.  "+train[train.toxic ==1].iloc[4,1])
print("\nSevere Toxic : \n")
print("\n1.  "+train[train.severe_toxic ==1].iloc[4,1])
print("\nThreat : \n")
print("\n1.  "+train[train.threat ==1].iloc[4,1])
print("\nObscene : \n")
print("\n1.  "+train[train.obscene ==1].iloc[4,1])
print("\nIdentity Hate : \n")
print("\n1.  "+train[train.identity_hate ==1].iloc[4,1])
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn
stopword=set(STOPWORDS)
subset=train[train.clean==True]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,stopwords=stopword)
wc.generate(" ".join(text))

plt.figure(figsize=(12,8))
plt.axis("off")
plt.title("Words frequented in Clean Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()
subset=train[train.severe_toxic==True]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(12,8))
plt.axis("off")
plt.title("Words frequented in Severe Toxic Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()
subset=train[train.threat==True]
text=subset.comment_text.values
wc= WordCloud(background_color="black",max_words=2000,stopwords=stopword)
wc.generate(" ".join(text))
plt.figure(figsize=(12,8))
plt.axis("off")
plt.title("Words frequented in Threat Comments", fontsize=20)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()
data = pd.concat([train.iloc[:,0:2], test.iloc[:,0:2]])
data = data.reset_index(drop=True)
data.head()
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
data['count_sent'] = data["comment_text"].apply(lambda x : len(re.findall("\n",str(x))) + 1 )
data['count_words'] = data["comment_text"].apply(lambda x : len(str.split(x))) 


data['count_unique_words'] = data["comment_text"].apply(lambda x : len(set(str.split(x)))) 
data.head()
data['count_letters'] = data['comment_text'].apply(lambda x : len(str(x)))
data['count_puntuations'] = data['comment_text'].apply( lambda x : len([p for p in str(x) if p in string.punctuation]))
data['count_word_upper'] = data["comment_text"].apply(lambda x : len([i for i in str(x) if i.isupper()   ]) )
data['count_words_title'] = data['comment_text'].apply(lambda x : len([j for j in str(x) if j.istitle() ]))
eng_stopwords = set(stopwords.words("english"))
data['count_stopwords'] = data['comment_text'].apply(lambda x : len([i for i in str(x).lower().split() if i in eng_stopwords])) 
data['mean_word_length'] = data["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
data['word_unique_percent']=data['count_unique_words']*100/data['count_words']
data['punct_percent']=data['count_puntuations']*100/data['count_words']
data['count_exclamation_marks'] = data['comment_text'].apply(lambda comment: comment.count('!'))
data['count_question_marks'] = data['comment_text'].apply(lambda comment: comment.count('?'))
data['count_symbols'] = data['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in '*&$%'))
data['count_smilies'] = data['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
data['ip']=data["comment_text"].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",str(x)))
#count of ip addresses
data['count_ip']=data["ip"].apply(lambda x: len(x))

#links
data['link']=data["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
#count of links
data['count_links']=data["link"].apply(lambda x: len(x))

#article ids
data['article_id']=data["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
data['article_id_flag']=data.article_id.apply(lambda x: len(x))

#username
##              regex for     Match anything with [[User: ---------- ]]
# regexp = re.compile("\[\[User:(.*)\|")
data['username']=data["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
#count of username mentions
data['count_usernames']=data["username"].apply(lambda x: len(x))
#check if features are created
#df.username[df.count_usernames>0]

# Leaky Ip
cv = CountVectorizer()
count_feats_ip = cv.fit_transform(data["ip"].apply(lambda x : str(x)))


# Leaky usernames

cv = CountVectorizer()
count_feats_user = cv.fit_transform(data["username"].apply(lambda x : str(x)))
data.head()
data.columns
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text
data['comment_text'] = data['comment_text'].map(lambda com : clean_text(com))

data['comment_text'].head()
train_feats=data.iloc[0:len(train),]
test_feats=data.iloc[len(train):,]
#join the tags
train_tags=train.iloc[:,2:]
train_feats=pd.concat([train_feats,train_tags],axis=1)
train_feats.shape
X = train_feats.comment_text
test_X = test_feats.comment_text

print(X.shape, test_X.shape)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000,stop_words='english')
vect
X_dtm = vect.fit_transform(X)
# examine the document-term matrix created from X_train
X_dtm
test_X_dtm = vect.transform(test_X)
# examine the document-term matrix from X_test
test_X_dtm
cols_target = ['count_sent', 'count_words', 'count_unique_words',
       'count_letters', 'count_puntuations', 'count_word_upper',
       'count_words_title', 'count_stopwords', 'mean_word_length',
       'word_unique_percent', 'punct_percent', 'count_exclamation_marks',
       'count_question_marks', 'count_symbols', 'count_smilies']
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self
target_x=train_feats[cols_target]
# target_x

TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
target_y=train_tags[TARGET_COLS]
print("Using only Indirect features")
model = LogisticRegression(C=3)
X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2018)
train_loss = []
valid_loss = []
importance=[]
preds_train = np.zeros((X_train.shape[0], len(y_train)))
preds_valid = np.zeros((X_valid.shape[0], len(y_valid)))
for i, j in enumerate(TARGET_COLS):
    print('Class:= '+j)
    model.fit(X_train,y_train[j])
    preds_valid[:,i] = model.predict_proba(X_valid)[:,1]
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    train_loss_class=log_loss(y_train[j],preds_train[:,i])
    valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    importance.append(model.coef_)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))



