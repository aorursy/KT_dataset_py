# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

 

# Here we are interested in Combined_News_DJIA.csv file

df = pd.read_csv('/kaggle/input/stocknews/Combined_News_DJIA.csv')
df.head()
df.tail()
df.info()
df.describe().T
df = df.drop(['Date'], axis=1)

df.head()
df_columns = df.columns

print(df_columns)
columns = ['Top1']
df['combined_news'] = df[columns].apply(lambda row:'.'.join(row.values.astype(str)), axis=1)
df = df.drop(columns, axis=1)
df.head()
columns_2 = ['Top2','Top3', 'Top4', 'Top5','Top6', 'Top7', 'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15', 'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23','Top24', 'Top25']

df = df.drop(columns_2, axis=1)
df.head()
df = df.replace('b\"|b\'|\\\\|\\\"', '', regex=True)

df.head(2)
import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline

import seaborn as sns

sns.set()



ax = sns.countplot(x='Label', hue='Label', data=df)
from collections import Counter



# Copy df enteries in 2 separate lists based on labels



data_djia_up = df[df['Label']==1].copy()

data_djia_down = df[df['Label']==0].copy()
print(data_djia_up[:2])
print(data_djia_down[:2])
import string

print(string.punctuation)
from nltk.corpus import stopwords

print(stopwords.words('english')[10:15])
def punctuation_stopwords_removal(news_article):

    # filters charecter-by-charecter : ['h', 'e', 'e', 'l', 'o', 'o', ' ', 'm', 'y', ' ', 'n', 'a', 'm', 'e', ' ', 'i', 's', ' ', 'p', 'u', 'r', 'v', 'a']

    remove_punctuation = [ch for ch in news_article if ch not in string.punctuation]

    # convert them back to sentences and split into words

    remove_punctuation = "".join(remove_punctuation).split()

    filtered_news_article = [word.lower() for word in remove_punctuation if word.lower() not in stopwords.words('english')]

    return filtered_news_article
data_djia_up.loc[:, 'combined_news'] = data_djia_up['combined_news'].apply(punctuation_stopwords_removal)

print(data_djia_up[:1])
words_djia_up = data_djia_up['combined_news'].tolist()
words_djia_up[:3]
data_djia_down.loc[:, 'combined_news'] = data_djia_down['combined_news'].apply(punctuation_stopwords_removal)

words_djia_down = data_djia_down['combined_news'].tolist()

print(words_djia_down[:2])
djia_up_list = []

for sublist in words_djia_up:

    for words in sublist:

        djia_up_list.append(words)



djia_down_list = []

for sublist in words_djia_down:

    for words in sublist:

        djia_down_list.append(words)

        
print('DJIA up list : {}'.format(len(djia_up_list)))

print('DJIA down list : {}'.format(len(djia_down_list)))
djia_up_counter = Counter(djia_up_list)

djia_down_counter = Counter(djia_down_list)



djia_up_top_30_words = pd.DataFrame(djia_up_counter.most_common(30), columns=['word', 'count'])

djia_down_top_30_words = pd.DataFrame(djia_down_counter.most_common(30), columns=['word', 'count'])
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='word', y='count', data=djia_up_top_30_words, ax=ax)

plt.title('Top 30 words when DJIA goes up')

plt.xticks(rotation='vertical')
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='word', y='count', data=djia_down_top_30_words, ax=ax)

plt.title('Top 30 words when DJIA goes down')

plt.xticks(rotation='vertical')
df.head()
!pip install transformers
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import torch

import transformers as ppb

import warnings

warnings.filterwarnings('ignore')
# For DistilBERT

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')



# load pre-trained model/tokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)
tokenized = df['combined_news'].apply((lambda x: tokenizer.encode(x, add_special_token=True)))
print(tokenized.shape)
tokenized[:1]
max_len = 0

for i in tokenized.values:

    if len(i)>max_len:

        max_len = len(i)

print(max_len)

padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
input_ids = torch.tensor(padded)

attention_mask = torch.tensor(attention_mask)

with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:, 0, :].numpy()

labels = df['Label']
print(features[:10])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
lr_clf = LogisticRegression()

lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(test_labels, lr_clf.predict(test_features))

fpr, tpr, thresholds = roc_curve(test_labels, lr_clf.predict_proba(test_features)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('/kaggle/input/stocknews/Combined_News_DJIA.csv')
columns = ['Top1', 'Top2','Top3', 'Top4', 'Top5','Top6', 'Top7', 'Top8', 'Top9', 'Top10', 'Top11', 'Top12', 'Top13', 'Top14', 'Top15', 'Top16', 'Top17', 'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23','Top24', 'Top25']

df['combined_news'] = df[columns].apply(lambda row:'.'.join(row.values.astype(str)), axis=1)
df = df.drop(columns, axis=1)
from bs4 import BeautifulSoup



df= df.replace('b\"|b\'|\\\\|\\\"', '', regex=True)

df.head()
bow_transformer = CountVectorizer(analyzer=punctuation_stopwords_removal).fit(df['combined_news'])
len(bow_transformer.vocabulary_)
sample_djia_down = df['combined_news'][0]

bow_sample_djia_down = bow_transformer.transform([sample_djia_down])

print(sample_djia_down)

print('=====')

print(bow_sample_djia_down)
print('Printing bag-of-words for sample 1 (DJIA goes down) :')

row, cols = bow_sample_djia_down.nonzero()

for col in cols:

    print(bow_transformer.get_feature_names()[col])
print(np.shape(bow_sample_djia_down))
print('Printing bag-of-words for sample 2 (DJIA goes up/remains the same :)')

sample_djia_up = df['combined_news'][1]

bow_sample_djia_up = bow_transformer.transform([sample_djia_up])

print(sample_djia_up)

print('======')

print(bow_sample_djia_up)
from sklearn.feature_extraction.text import TfidfTransformer



bow_data = bow_transformer.transform(df['combined_news'])

print(bow_data[:1])

tfidf_transformer = TfidfTransformer().fit(bow_data)
final_tfidf = tfidf_transformer.transform(bow_data)

print(final_tfidf)
from sklearn.model_selection import train_test_split



features_train, features_test, labels_train, labels_test = train_test_split(final_tfidf, df['Label'], test_size=0.3, random_state=5)
features_train = features_train.A

features_test = features_test.A
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



djia_movement_detect_model_MNB = MultinomialNB()

djia_movement_detect_model_MNB.fit(features_train, np.asarray(labels_train, dtype="float64"))

pred_test_MNB = djia_movement_detect_model_MNB.predict(features_test)



acc_MNB = accuracy_score(np.asarray(labels_test, dtype="float64"), pred_test_MNB)

print(acc_MNB)
from sklearn.metrics import roc_curve, auc



fpr, tpr, thr = roc_curve(np.asarray(labels_test, dtype="float64"), djia_movement_detect_model_MNB.predict_proba(features_test)[:,1])

plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

auc_knn4 = auc(fpr, tpr) * 100

plt.legend(["AUC {0:.3f}".format(auc_knn4)]);