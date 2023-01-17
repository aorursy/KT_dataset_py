import pandas as pa

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import DictVectorizer

from scipy.sparse import hstack

from sklearn.linear_model import Ridge
from nltk.corpus import stopwords

stop_words = set (stopwords.words('english'))
train=pa.read_table("../input/train.tsv")
test=pa.read_table("../input/test.tsv")
train['item_description']=train['item_description'].str.lower()
test['item_description']=test['item_description'].str.lower()
train['item_description']=train['item_description'].replace('[^a-zA-Z]', ' ', regex = True)
test['item_description']=test['item_description'].replace('[^a-zA-Z]', ' ', regex = True)
train.isnull().sum()
train["category_name"].fillna(value='missing/missing/missing', inplace=True)

train["brand_name"].fillna(value="missing", inplace=True)

train["item_description"].fillna(value="No description yet", inplace =True)
test["category_name"].fillna(value='missing/missing/missing', inplace=True)

test["brand_name"].fillna(value="missing", inplace=True)

test["item_description"].fillna(value="No description yet", inplace =True)
train['category_main']=train.category_name.str.split("/").str.get(0)

train['category_sub1']=train.category_name.str.split("/").str.get(1)

train['category_sub2']=train.category_name.str.split("/").str.get(2)
test['category_main']=test.category_name.str.split("/").str.get(0)

test['category_sub1']=test.category_name.str.split("/").str.get(1)

test['category_sub2']=test.category_name.str.split("/").str.get(2)
train.isnull().sum()
test.isnull().sum()
def stop(txt):

    words = [w for w in txt.split(" ") if not w in stop_words and len(w)>2]

    return words
train['tokens']=train['item_description'].map(lambda x:stop(x))
test['tokens']=test['item_description'].map(lambda x:stop(x))
train['desc_len']=train['tokens'].map(lambda x: len(x))
test['desc_len']=test['tokens'].map(lambda x: len(x))
train['name_len']=train['name'].map(lambda x: len(x))
test['name_len']=test['name'].map(lambda x: len(x))
train.head()
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stemm(text):

    stemmed=[stemmer.stem(w) for w in text]

    return stemmed
train['stemmed']=train['tokens'].map(lambda x: stemm(x))
test['stemmed']=test['tokens'].map(lambda x: stemm(x))
train.head()
def join(txt):

    joinedtext=' '.join(word for word in txt)

    return joinedtext
train['final_desc']=train['stemmed'].map(lambda x: join(x))
test['final_desc']=test['stemmed'].map(lambda x: join(x))
train['final_desc'].head()
test['final_desc'].head()
vectorizer = TfidfVectorizer(min_df=10)

X_tfidf = vectorizer.fit_transform(train['final_desc']) 
X_tfidf.shape
train['name'].shape
#Avectorizer = TfidfVectorizer(min_df=10)

Y_tfidf = vectorizer.transform(test['final_desc']) 
test['name'].shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols=['name',"brand_name","category_main","category_sub1","category_sub2"]

for col in categorical_cols:

    # taking a column from dataframe, encoding it and replacing same column in the dataframe.

    train[col] = le.fit_transform(train[col])
test.head(2)
categorical_cols=['name',"brand_name","category_main","category_sub1","category_sub2"]

for col in categorical_cols:

    # taking a column from dataframe, encoding it and replacing same column in the dataframe.

    test[col] = le.fit_transform(test[col])
train.head()
y = train['price']
train.columns
test.head(2)
train1=train.drop(train.columns[[0,3,5,7,11,13,14,15]],axis=1)
train1.head(1)
test1=test.drop(test.columns[[0,3,6,8,10,13,14]],axis=1)
test1.head(2)
X = hstack([X_tfidf,train1])
Y = hstack([Y_tfidf,test1])
clf = Ridge(alpha=20.0)
import time

start=time.clock()

clf.fit(X, y)

print(time.clock()-start)
import time

start=time.clock()

rslt=clf.predict(Y)

print(time.clock()-start)
rslt.shape
test.shape
rslt1=pa.DataFrame(rslt)
rslt1.columns=["price"]
rslt1["test_id"]=rslt1.index
rslt1.to_csv("sample_submission.csv", encoding='utf-8', index=False)