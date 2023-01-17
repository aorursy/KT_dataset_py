
# Import necessary libraries
import os
import numpy as np 
import pandas as pd 
import nltk
import re
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier




df = pd.read_csv('/kaggle/input/news-aggregator-dataset/uci-news-aggregator.csv',header='infer',sep=',')
print('Number of Rows in df = %d'%len(df))
df.shape
df.head()
title_category_df = df.loc[:,['TITLE','CATEGORY']]
del df
gc.collect()
title_category_df.head()
# check for null values

title_category_df.isnull().sum()
# count unique categories


title_category_df['TITLE'].value_counts().sort_values(ascending=False).head(10)

# Data Cleaning
# removes numbers and )| + and whitespaces
stopword_english = stopwords.words('english')
def re_sub(s):
    
    s = s.lower()
    
    s = re.sub('\d+','',s)
    s =re.sub('\s\W',' ',s)
    s= re.sub('\W\s',' ',s)
    s = re.sub('\s+',' ',s)
  
    
    return s
    
title_category_df['sent'] = [re_sub(s =sen ) for sen in title_category_df['TITLE'] ]
stemmer = nltk.PorterStemmer()
title_category_df['word_tokens']  = tuple(map(lambda x : nltk.word_tokenize(x),title_category_df['sent']))
title_category_df['word_tokens_after_sw'] = title_category_df['word_tokens'].apply(lambda x: [word for word  in x if word not in stopword_english ] )
title_category_df['word_tokens_after_sw_stemmer'] = title_category_df['word_tokens_after_sw'].apply(lambda x: [stemmer.stem(word) for word in x] )

title_category_df['word_tokens_after_sw1']  = title_category_df['word_tokens_after_sw_stemmer'].apply(lambda x : ','.join(x))

label_encode = LabelEncoder()
label_encode.fit(title_category_df['CATEGORY'])
trans_x = dict(zip(label_encode.classes_,label_encode.transform(label_encode.classes_)))
title_category_df['label_category'] = label_encode.transform(title_category_df['CATEGORY'])
title_category_df.columns
train_columns = title_category_df[['word_tokens_after_sw1','label_category']]
del title_category_df
gc.collect()
X_train,x_test,Y_train,y_test = train_test_split(train_columns['word_tokens_after_sw1'],train_columns['label_category']
                                                ,test_size=0.2)
X_train.shape
# Vectorization 
vectorizer = TfidfVectorizer(max_features=8000,min_df=1  )

x_train_features = vectorizer.fit_transform(X_train).toarray()
x_test_features = vectorizer.transform(x_test).toarray()

# Applying naive bayes

naivebayes = MultinomialNB(alpha=1)
naivebayes.fit(x_train_features,Y_train)
y_pred = naivebayes.predict(x_test_features)
np.round(accuracy_score(y_test,y_pred),2)
lr = LogisticRegression()
lr.fit(x_train_features,Y_train)
y_pred = lr.predict(x_test_features)
np.round(accuracy_score(y_test,y_pred),2)

