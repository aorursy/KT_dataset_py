#package for data reading and manipulation

import pandas as pa
#package for hstacking

from scipy.sparse import hstack
#package for Ridge regression

from sklearn.linear_model import Ridge
#package for Stopwords

from nltk.corpus import stopwords

stop_words = set (stopwords.words('english'))
#package for Stemming

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
#package for Vectorizing

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=10)
#package for Label encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
#reading the train data

train =pa.read_table("../input/train.tsv")
#reading the test data

test=pa.read_table("../input/test.tsv")
#Viewing the  missing values

train.isnull().sum()
#handling the missing values for the train data

train["category_name"].fillna(value='missing/missing/missing', inplace=True)

train["brand_name"].fillna(value="missing", inplace=True)

train["item_description"].fillna(value="No description yet", inplace =True)
#handling the missing values for the test data

test["category_name"].fillna(value='missing/missing/missing', inplace=True)

test["brand_name"].fillna(value="missing", inplace=True)

test["item_description"].fillna(value="No description yet", inplace =True)
#Splitting the category column into category_main, category_sub1, category_sub2 for train data

train['category_main']=train.category_name.str.split("/").str.get(0)

train['category_sub1']=train.category_name.str.split("/").str.get(1)

train['category_sub2']=train.category_name.str.split("/").str.get(2)
#Splitting the category column into category_main, category_sub1, category_sub2 for test data

test['category_main']=test.category_name.str.split("/").str.get(0)

test['category_sub1']=test.category_name.str.split("/").str.get(1)

test['category_sub2']=test.category_name.str.split("/").str.get(2)
#Removing the punctuations and numbers leaving out only alphabets

train['item_description']=train['item_description'].replace('[^a-zA-Z]', ' ', regex = True)

test['item_description']=test['item_description'].replace('[^a-zA-Z]', ' ', regex = True)
#function to remove stop words and tokenizing it

def stop(txt):

    words = [w for w in txt.split(" ") if not w in stop_words and len(w)>2]

    return words
#removing stop words and tokenizing the item_description

train['tokens']=train['item_description'].map(lambda x:stop(x))

test['tokens']=test['item_description'].map(lambda x:stop(x))
#Function to stem

def stemm(text):

    stemmed=[stemmer.stem(w) for w in text]

    return stemmed
#Stemming the tokenized data

train['stemmed']=train['tokens'].map(lambda x: stemm(x))

test['stemmed']=test['tokens'].map(lambda x: stemm(x))
#Function to join the stemmed tokens into a complete sentence

def join(txt):

    joinedtext=' '.join(word for word in txt)

    return joinedtext
#Joining the stemmed tokenized data into a sentence, so that it can be vectorized

train['final_desc']=train['stemmed'].map(lambda x: join(x))

test['final_desc']=test['stemmed'].map(lambda x: join(x))
#Tf-idf vectorization of the final description column

X_tfidf = vectorizer.fit_transform(train['final_desc'])

Y_tfidf = vectorizer.transform(test['final_desc'])
#length of the description is taken into ‘desc_len’

train['desc_len']=train['tokens'].map(lambda x: len(x))

test['desc_len']=test['tokens'].map(lambda x: len(x))
#length of the name column is taken into ‘name_len’

#train['name_len']=train['name'].map(lambda x: len(x))

#test['name_len']=test['name'].map(lambda x: len(x))
#label encoding the columns 'name', "brand_name", "category_main", "category_sub1", and #"category_sub2"

categorical_cols=["brand_name","category_main","category_sub1","category_sub2"]

for col in categorical_cols:

    train[col] = le.fit_transform(train[col])
#label encoding the test data's categorical columns

for col in categorical_cols:

    test[col] = le.fit_transform(test[col])
#The target column ‘price’ is stored in y

train_target = train['price']
train.head(1)
test.head(1)
#dropping the unimportant columns

train1=train.drop(train.columns[[0,1,3,5,7,11,12,13]],axis=1)

test1=test.drop(test.columns[[0,1,3,6,10,11,12]],axis=1)
train1.head(1)
test1.head(1)
#hstacking the tfidf vectorized features with the train data.

X_train = hstack([X_tfidf,train1])

Y_test = hstack([Y_tfidf,test1])
#model creation

regr = Ridge(alpha=1.0, random_state=241)

import time

start=time.clock()

regr.fit(X_train, train_target)

print(time.clock()-start)
#model prediction

import time

start=time.clock()

rslt=regr.predict(Y_test)

print(time.clock()-start)
#saving the predictied file into result.csv

rslt1=pa.DataFrame(rslt)

rslt1.columns=["price"]

rslt1["test_id"]=rslt1.index

rslt1.to_csv("sample_submission.csv", encoding='utf-8', index=False)