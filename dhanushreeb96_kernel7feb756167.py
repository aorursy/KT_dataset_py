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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip',sep = '\t')
test= pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip', sep='\t')
sub= pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep='\t')

train.head(10)
#Getting Class wise Counts
class_wise_cnt=train['Sentiment'].value_counts()
x=class_wise_cnt.index
y=np.array(class_wise_cnt)
#Bar chart
sns.barplot(x,y)
plt.xlabel('Sentiment')
plt.ylabel('counts')
plt.show()
#Total number of unique sentences
print("The total Unique sentences are:", format(len(train['SentenceId'].unique())))
#Creating word clouds for most common words in the whole dataset
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords=set(STOPWORDS)
stopwords.update(['movie','film']) #Since these having these two words doesn't make sense
text = " ".join(phrase for phrase in train.Phrase)
wordcloud = WordCloud(stopwords= stopwords, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear') #what is bilinear interpolation? to make the displayed image appear more smoothly
plt.axis("off")
plt.show()
#Creating word clouds for most common words within each class of sentiment
negative=" ".join(phrase for phrase in train[train['Sentiment']==0].Phrase)
somewhat_negative=" ".join(phrase for phrase in train[train['Sentiment']==1].Phrase)
neutral=" ".join(phrase for phrase in train[train['Sentiment']==2].Phrase)
somewhat_positive=" ".join(phrase for phrase in train[train['Sentiment']==3].Phrase)
positive=" ".join(phrase for phrase in train[train['Sentiment']==4].Phrase)

def make_wordcloud(text, title=None):
    wordcloud=WordCloud(stopwords= stopwords, background_color="white", random_state=1, ).generate(text)
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
#Calling the function on the sentiments:
make_wordcloud(negative)
make_wordcloud(positive)
#N-grams because words by itself may not give us many insights as compared to a sequence of n words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()
tfidfv = TfidfVectorizer(ngram_range=(1,3),tokenizer=tokenizer.tokenize) #it considers sequence of 1 words, 2 words and 3 words
tfidfv.fit(train['Phrase'])
tfidfv.fit(test['Phrase'])
train_vectorized = tfidfv.fit_transform(train['Phrase'])
test_vectorized = tfidfv.fit_transform(test['Phrase'])
#print(tfidfv.get_feature_names())

from sklearn.model_selection import train_test_split
x_train , x_val, y_train , y_val = train_test_split(train_vectorized,train['Sentiment'],test_size = 0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print(classification_report( logreg.predict(x_val) , y_val))
print(accuracy_score( logreg.predict(x_val) , y_val ))
