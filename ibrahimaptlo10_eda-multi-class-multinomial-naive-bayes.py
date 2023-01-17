import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
from numpy import nan
from bs4 import BeautifulSoup    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
train_data= pd.read_csv('../input/train.tsv',sep='\t')
test_data = pd.read_csv('../input/test.tsv',sep='\t')
train_data.info()

test_data.info()

train_data.head()

#create new column with if and for loop with the Sentiment Phrase column¶

Sentiment_words=[]
for row in train_data['Sentiment']:
    if row ==0:
        Sentiment_words.append('negative')
    elif row == 1:
        Sentiment_words.append('neutral')
    elif row == 2:
        Sentiment_words.append('somewhat negative')
    elif row == 3:
        Sentiment_words.append('somewhat positive')
    elif row == 4:
        Sentiment_words.append('positive')
    else:
        Sentiment_words.append('Failed')
train_data['Sentiment_words'] = Sentiment_words
#count values of Sentiment Phrase¶

word_count=pd.value_counts(train_data['Sentiment_words'].values, sort=False)
word_count
Index = [1,2,3,4,5]
plt.figure(figsize=(15,5))
plt.bar(Index,word_count,color = 'blue')
plt.xticks(Index,['negative','neutral','somewhat negative','somewhat positive','positive'],rotation=45)
plt.ylabel('word_count')
plt.xlabel('word')
plt.title('Count of Moods')
plt.bar(Index, word_count)
for a,b in zip(Index, word_count):
    plt.text(a, b, str(b) ,color='green', fontweight='bold')
#function to clean the column Phrase in the data set¶

def review_to_words(raw_review): 
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))
#run the function in train and test data set¶

corpus= []
for i in range(0, 156060):
    corpus.append(review_to_words(train_data['Phrase'][i]))
corpus1= []
for i in range(0, 156060):
    corpus1.append(review_to_words(train_data['Phrase'][i]))
#create new column and merge it with the new cleaning list¶

train_data['new_Phrase']=corpus

#drop the old column¶

train_data.drop(['Phrase'],axis=1,inplace=True)

train_data.head()

"""
i do it with positive Sentiment and easly o reapte it to the rest of Sentiment words
select positive Sentiment from data set"""
positive=train_data[train_data['Sentiment_words']==('positive')]

#prepare the data to split it¶

words = ' '.join(positive['new_Phrase'])
split_word = " ".join([word for word in words.split()])
#prepare the data to visual it¶

wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(split_word)
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#now here some steps to count every word in postive Sentiment
pos=positive['new_Phrase']

vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000)
pos_words = vectorizer.fit_transform(pos)

pos_words = pos_words.toarray()

pos= vectorizer.get_feature_names()

pos

print (pos_words.shape)

dist = np.sum(pos_words, axis=0)
for tag, count in zip(pos, dist):
    print (tag,count)
postive_new= pd.DataFrame(dist)

postive_new.columns=['word_count']

postive_new['word'] = pd.Series(pos, index=postive_new.index)

postive_new1=postive_new[['word','word_count']]

postive_new1.head()

postive_new1.head()

top_30_words=postive_new1.sort_values(['word_count'],ascending=[0])

top_30_words.head(30)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x__train = cv.fit_transform(corpus).toarray()
x__test= cv.fit_transform(corpus1).toarray()
y = train_data.iloc[:, 2].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x__train, y, test_size = 0.40, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_real_pred = classifier.predict(x__test)
#check accurecy of the multinomial Naive Bayes 
#using mean squre errore 
mse = ((y_pred - y_test) ** 2).mean()

mse

#also using root mean squre error
rmse = sqrt(mse)

rmse

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
y_test

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred, labels=[0, 1,2,3,4])
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()

accuracies.std()
#convert the out put array to data frame

import pandas as pd
c=y_pred
c.tolist()
df_final = pd.DataFrame(c)
my_columns = [ "Sentiment"]
df_final.columns = my_columns
df_final.head()
#convert the out put array to data frame and contact the data frame with id column in test data frame 
#to get the out put like submission file
id_test=test_data.PhraseId
type(id_test)

a=id_test
a.tolist()

df_final_id = pd.DataFrame(a)
my_columns = [ "PhraseId"]
df_final_id.columns = my_columns

submission = pd.concat([df_final_id, df_final],axis=1)
submission.head(10)
submission.to_csv('submission.csv', sep='\t',encoding='utf8')

#thank for all 
