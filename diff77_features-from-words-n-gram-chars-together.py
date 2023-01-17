import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from string import punctuation
import re
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack, hstack

train=pd.read_csv('../input/train.tsv', sep='\t')
test=pd.read_csv('../input/test.tsv', sep='\t')
sub=pd.read_csv('../input/sampleSubmission.csv')
train[train['Sentiment']==0].head(10)
test['Sentiment']=777
df=pd.concat([train, test], ignore_index=True, sort=False)
print(df.shape)
df.tail()
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
#nltk.download('punkt')
def clean_phrase(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
df['clean_review']=clean_phrase(df.Phrase.values)
df.head()
df_train=df[df.Sentiment!=777]
df_train.shape
df_test=df[df.Sentiment==777]
df_test=df_test.drop('Sentiment',axis=1)
print(df_test.shape)
df_test.head()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(df_train.Sentiment.values)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_word=TfidfVectorizer(ngram_range=(1,2),
                           stop_words = 'english',
                           max_df=0.95,min_df=10,
                           sublinear_tf=True,
                           analyzer='word',
                           max_features=18000
                          )

tfidf_word.fit(df.clean_review)

tfidf_WordTrain=tfidf_word.transform(df_train.clean_review)
tfidf_WordTrain.shape
list(tfidf_word.vocabulary_)[:10]
tfidf_char=TfidfVectorizer(ngram_range=(3,5),
                           strip_accents='unicode',
                           analyzer='char',
                           stop_words='english',
                           sublinear_tf=True,
                           #max_features=50000,
                           #dtype=np.int32 
                          )

tfidf_char.fit(df.clean_review)

tfidf_CharTrain=tfidf_char.transform(df_train.clean_review)
tfidf_CharTrain.shape
list(tfidf_char.vocabulary_)[:10]
from sklearn.model_selection import train_test_split
X_train_word,X_val_word,y_train_word,y_val_word=train_test_split(tfidf_WordTrain,y,test_size=0.3)
lr=LogisticRegression(penalty='l1', max_iter=100)
lr.fit(X_train_word,y_train_word)
y_pred_word=lr.predict(X_val_word)
print("Test Accuracy ", accuracy_score(y_pred_word, y_val_word)*100 , '%')
X_train_char,X_val_char,y_train_char,y_val_char=train_test_split(tfidf_CharTrain, y ,test_size=0.3)
lr.fit(X_train_char,y_train_char)
y_pred_char=lr.predict(X_val_char)
print("Test Accuracy ", accuracy_score(y_pred_char, y_val_char)*100 , '%')
tfidf_WordTrain.shape
tfidf_CharTrain.shape
big_train=hstack([tfidf_WordTrain, tfidf_CharTrain])
big_train.shape
X_train, X_val, y_train, y_val=train_test_split(big_train,y,test_size=0.3)
lr.fit(X_train, y_train)
y_pred=lr.predict(X_val)
print("Test Accuracy ", accuracy_score(y_pred, y_val)*100 , '%')
X_val.shape
sub_test_char=tfidf_char.transform(df_test.clean_review)
sub_test_word=tfidf_word.transform(df_test.clean_review)
test_final=hstack([sub_test_char, sub_test_word])
test_final.shape
final_pred=lr.predict(test_final)
sub.shape

sub.Sentiment=final_pred
sub.head()
sub.to_csv('submission.csv',index=False)
sub.head()