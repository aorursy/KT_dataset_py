# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing packages

import pandas as pd # for dataframe

import numpy as np  # for algebric 

import nltk         #for nlp 

import matplotlib.pyplot as plt # for visualization

import seaborn as sns           # for visualization
# Importing csv files

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train[:5]
test[:2]
# viewing each unique values in each columns

for i in train.columns:

    print('Column Name  :',i)

    print(train[i].unique())
# id is not essential. so, i'm droping it

df_train = train.drop(['id'],axis=1)

df_test = test.drop(['id'],axis=1)
#Hence when it comes twitter, Hashtag is one of important thing. so, i'm created new column contains only hashtags

# for this i used regrx package.

import re

df_train['Hashtag'] = df_train['text'].map(lambda x: re.findall(r'#(\w+)',x)).apply(lambda x: ", ".join(x))

df_test['Hashtag'] = df_test['text'].map(lambda x: re.findall(r'#(\w+)',x)).apply(lambda x: ", ".join(x))

df_train['@'] = df_train['text'].map(lambda x: re.findall(r'@(\w+)',x)).apply(lambda x: ", ".join(x))

df_test['@'] = df_test['text'].map(lambda x: re.findall(r'@(\w+)',x)).apply(lambda x: ", ".join(x))



# data Cleaning

# i'm defining function to clean data. it will make work easier

def remove_punctuation(txt):

    import string

    result = txt.translate(str.maketrans('','',string.punctuation))

    return result

def lower_text(txt):

    return txt.lower()

def remove_no(txt):

    import re

    return re.sub(r"\d+","",txt)

def remove_html_tags(text):

    """Remove html tags from a string"""

    import re

    clean = re.compile('<.*?>')

    return re.sub(clean, '', text)

def removeurl(txt):

    import re

    return re.sub(r'http?\S+|www\.\S+', '',txt)



# defining all function to one for this problem

def norm(txt):

    x = remove_punctuation(txt)

    x = lower_text(x)

    x = remove_html_tags(x)

    x = remove_no(x)

    x = removeurl(x)

    return x



# applying data cleaning function for text

df_train['text'] = df_train['text'].map(lambda x:  norm(x))

df_test['text'] = df_test['text'].map(lambda x:  norm(x))



#There are some magic words "%20" which indicates space in this data.so,Removing magic words

df_train['keyword'] = df_train['keyword'].map(lambda s: s.replace('%20',' ') if isinstance(s, str) else s)

df_test['keyword'] = df_test['keyword'].map(lambda s: s.replace('%20',' ') if isinstance(s, str) else s)
# Cleaned Data

df_test

df_train[:5]
#Seperating data to explore it

train_yes = df_train[df_train["target"]==1]

train_no = df_train[df_train["target"]==0]

train_yes[:5]
# created a para from all text for each target

# Sepereate Data to two 

train_txt_yes = " ".join(str(i) for i in train_yes['text'])

train_txt_no = " ".join(str(i) for i in train_no['text'])

test_txt = " ".join(str(i) for i in df_test['text'])

#train_txt_yes
#replacing empty space with null values

train_yes["Hashtag"]= train_yes["Hashtag"].replace(r'^\s*$', np.nan, regex=True)

train_no["Hashtag"]= train_yes["Hashtag"].replace(r'^\s*$', np.nan, regex=True)

train_yes["@"]= train_yes["@"].replace(r'^\s*$', np.nan, regex=True)

train_no["@"]= train_yes["@"].replace(r'^\s*$', np.nan, regex=True)

#train_yes["Hashtag"].isnull().sum()
print("Total No. of disaster tweets                        :",train_yes["text"].count())

print("Total No. of hashtag present in disaster tweets     :",train_yes["Hashtag"].notnull().sum())

print("Total No. of non-disaster tweets                    :",train_no["text"].count())

print("Total No. of hashtag present in non-disaster tweets :",train_no["Hashtag"].notnull().sum(),"\n")



print("Total No. of disaster tweets                  :",train_yes["text"].count())

print("Total No. of @ present in disaster tweets     :",train_yes["@"].notnull().sum())

print("Total No. of non-disaster tweets              :",train_no["text"].count())

print("Total No. of @ present in non-disaster tweets :",train_no["@"].notnull().sum())
# deffining function for tokenization & lemmatize.

# Tokenization function

def Token_and_removestopword(txt):

    import nltk

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    words = nltk.word_tokenize(txt)

    without_stop_words = []

    for word in words:

        if word not in stop_words:

            without_stop_words.append(word)

    return without_stop_words

#lemmatizing tokenized words

def lemmatize_word(tokens,pos="v"): 

    import nltk

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    # provide context i.e. part-of-speech 

    lemmas = [lemmatizer.lemmatize(word, pos =pos) for word in tokens] 

    return lemmas
tok_yes = Token_and_removestopword(train_txt_yes)

tok_no = Token_and_removestopword(train_txt_no)

test_tok = Token_and_removestopword(test_txt)

tok_yes[:5]
#lem_wd = [lemmatizer.lemmatize(x, pos ='v') for x in tok]

lem_tok_yes = lemmatize_word(tok_yes)

lem_tok_no = lemmatize_word(tok_no)

lem_test = lemmatize_word(test_tok)

print(len(tok_yes))

print(len(lem_tok_yes))
def top_word_dis(lem_tok,TopN=10):

    fq = nltk.FreqDist(lem_tok)

    rslt = pd.DataFrame(fq.most_common(TopN),

                        columns=['Word', 'Frequency']).set_index('Word')

    plt.style.use('ggplot')

    rslt.plot.bar()
top_word_dis(lem_tok_yes)
top_word_dis(lem_tok_no)
print(lem_test[:10])
#TF-IDF Vectorizer

def tfidf(train_int,test_int=None,Ngram_min=1,Ngram_max=1):

    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer

    #to convert token input into text

    def toktotxt(txt_int):

        if isinstance(txt_int[0], list):

            text = txt_int.apply(lambda x :" ".join(str(i) for i in x))

        else:

            text = txt_int

        return text

    train_txt = toktotxt(train_int)  

    vectorizer = TfidfVectorizer(ngram_range = (Ngram_min,Ngram_max))

    vectorizer.fit(train_txt)

    X = vectorizer.transform(train_txt)

    train = X.toarray()

    # to get both transform for train & split

    if test_int is None:

        out = train

    else:

        test_txt = toktotxt(test_int)

        Y = vectorizer.transform(test_txt)

        test = Y.toarray()

        out = train, test

    return out



"""

Above step is not necessary. you can also use below step

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

vectorizer.fit(train_data)

vectorizer.transform(train_data)  ---> output as array

vectorizer.transform(test_data) ---> essential if you has test data & to change its shape for further process

"""
def countvectorizer(train_int,test_int=None,Ngram_min=1,Ngram_max=1):

    import pandas as pd

    from sklearn.feature_extraction.text import CountVectorizer

    def toktotxt(txt_int):

        if isinstance(txt_int[0], list):

            text = txt_int.apply(lambda x :" ".join(str(i) for i in x))

        else:

            text = txt_int

        return text

    train_txt = toktotxt(train_int)  

    vectorizer = CountVectorizer(ngram_range = (Ngram_min,Ngram_max))

    vectorizer.fit(train_txt)

    X = vectorizer.transform(train_txt)

    train = X.toarray()

    if test_int is None:

        out = train

    else:

        test_txt = toktotxt(test_int)

        Y = vectorizer.transform(test_txt)

        test = Y.toarray()

        out = train, test

    return out



"""

same as for tdidf

"""
#Dimentional Reduction

def PCA(X_train,Y_train=None,X_test=None,n=1000):

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n)

    X = pca.fit(X_train,y_train)

    train = pca.transform(X_train)

    if X_test.any() == None:

        out = train

    else:

        test = pca.transform(X_test)

        out = train, test

    return out
# ML Algorithms

def logreg(X_train, y_train,X_test,n=20):

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return preds

def naivebiase(X_train, y_train,X_test,n=20):

    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return preds

def accuracy(y_true, y_pred):

    from sklearn.metrics import accuracy_score

    return accuracy_score(y_true, y_pred)
# Cleaned Data

df_test

df_train[:5]
#apply tokenization & lemmatization to each text

df_train["tokenized_text"] = df_train["text"].apply(lambda x:lemmatize_word(Token_and_removestopword(x)))

df_test["tokenized_text"] = df_test["text"].apply(lambda x:lemmatize_word(Token_and_removestopword(x)))

df_train[:5]
"""

for vectorization in countervector & tfidf the input be in text formate.

that's why i join all text in list in above mentioned function.

"""

tfidf_train,tfidf_test = tfidf(df_train["tokenized_text"],df_test["tokenized_text"])

cnt_train,cnt_test = countvectorizer(df_train["tokenized_text"],df_test["tokenized_text"])
#check whether shape aligns or not

print(tfidf_train.shape)

print(cnt_train.shape)

print(df_train["target"].shape)
# test & train Split

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X=tfidf_train

y=df_train["target"]

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.1,random_state=42)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# dimentionality reduction ---> it is not necessary for this problem. because it is very small data

x_train,x_test = PCA(x_train,y_train,x_test,n=500)
# prediction using logistic regression model. it is defined as function above

preds = logreg(x_train, y_train,x_test)

accuracy(y_test,preds)
#prediction using naivebiase Gaussian model. it is defined as function above

preds = naivebiase(x_train, y_train,x_test)

accuracy(y_test,preds)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X=cnt_train

y=df_train["target"]

x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.1,random_state=42)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
x_train,x_test = PCA(x_train,y_train,x_test,n=500)
preds = logreg(x_train, y_train,x_test)

accuracy(y_test,preds)
preds = naivebiase(x_train, y_train,x_test)

accuracy(y_test,preds)
x_train = cnt_train

y_train = df_train["target"] 

x_test = cnt_test
# check wheather shape alligns or not

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
#if shape is not aligned check wheather do you transform test data in vectorization part or not.

#if test data is not transformed as per train data it may led to error due to mismatch in features.

x_train,x_test = PCA(x_train,y_train,x_test,n=500)
preds = logreg(x_train,y_train,x_test)
preds
submission = test[["id"]]

submission["target"] = preds
submission[:5]
submission.to_csv("submission.csv",index=False)