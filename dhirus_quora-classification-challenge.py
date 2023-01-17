# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.info()
train.question_text[9]
train.target.value_counts().plot(kind='bar')
def clean_text(text, remove_stopwords = False):
    text = text.lower()
    text = text.strip().replace("\n", " ").replace("\r", " ") ## remove new line chars
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)  ## remove unwanted chars
    text = re.sub(r'\'', ' ', text)
    text = re.sub('[\d+]', '', text) ## remove numerics
    text=  re.sub("\s\s+", " ", text)  ## remove white spaces
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text
clean_question=[]
for text in tqdm(train.question_text):
    textt=clean_text(text,remove_stopwords = True)
    clean_question.append(textt)
train['clean_question']=clean_question
train.head()
sincere = train[train.target==0]["clean_question"]
insincere = train[train.target==1]["clean_question"]

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=STOPWORDS,
                          max_words=50000,
                          max_font_size=30, 
                          random_state=42
                         ).generate(str(sincere))

print(wordcloud)
plt.figure(figsize=(16,13))

fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#fig.savefig("word1.png", dpi=900)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=STOPWORDS,
                          max_words=50000,
                          max_font_size=30, 
                          random_state=42
                         ).generate(str(insincere))

print(wordcloud)
plt.figure(figsize=(16,13))

fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#fig.savefig("word1.png", dpi=900)
X_train, X_test, y_train, y_test = train_test_split(train['clean_question'], 
                                                    train['target'], 
                                                    random_state=0)
tfvect=TfidfVectorizer(stop_words='english',min_df=3).fit(X_train)
x_train_tfvect=tfvect.transform(X_train)
x_test_tfvect=tfvect.transform(X_test)
name=tfvect.get_feature_names()
feature_names = np.array(tfvect.get_feature_names())
sorted_tfidf_index = x_train_tfvect.max(0).toarray()[0].argsort()
print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-100:-1]]))
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='sag')
model.fit(x_train_tfvect,y_train)
predicted= model.predict(x_test_tfvect)

accuracy=accuracy_score(y_test, predicted)

report=classification_report(y_test, predicted)
accuracy
print(report)
clean_question_test=[]
for text in tqdm(test.question_text):
    textt=clean_text(text,remove_stopwords = True)
    clean_question_test.append(textt)
x_testt_tfvect=tfvect.transform(clean_question_test)
test_pred=model.predict(x_testt_tfvect)
test_pred
out_df = pd.DataFrame({"qid":test["qid"].values})
out_df['prediction'] = test_pred
out_df.to_csv("submission.csv", index=False)