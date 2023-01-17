import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import emoji



#Count vectorizer for N grams

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression



# Nltk for tekenize and stopwords

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_df.head()
def missing_value_of_data(data):

    total=data.isnull().sum().sort_values(ascending=False)

    percentage=round(total/data.shape[0]*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
train_df.replace('',np.nan,inplace=True)

test_df.replace('',np.nan,inplace=True)
missing_value_of_data(train_df)
missing_value_of_data(test_df)
def count_values_in_column(data,feature):

    total=data.loc[:,feature].value_counts(dropna=False)

    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
count_values_in_column(train_df,'target')
def duplicated_values_data(data):

    dup=[]

    columns=data.columns

    for i in data.columns:

        dup.append(sum(data[i].duplicated()))

    return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])
duplicated_values_data(train_df)
duplicated_values_data(test_df)
train_df.drop_duplicates(subset="text",inplace=True)
train_df.drop(['id','keyword','location'],axis=1,inplace=True)

test_df.drop(['keyword','location'],axis=1,inplace=True)
#from functools import reduce



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def clean_nonalp(text):

    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

    

def clean_punc(text):

    return re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*','', text)



def only_words(text):

    return re.sub(r'\b\d+\b','', text)



def clean_stop(text,stop):

    output = ' '.join([word for word in text.split() if word not in stop])

    return output



def lowercasing(text):

    return str.lower(text)
stop = set(stopwords.words("english"))



train_df['text']=train_df['text'].apply(lambda x: remove_URL(x))



train_df['text']=train_df['text'].apply(lambda x: remove_html(x))



train_df['text']=train_df['text'].apply(lambda x: remove_emoji(x))



train_df['text']=train_df['text'].apply(lambda x : clean_nonalp(x))



train_df['text']=train_df['text'].apply(lambda x : clean_punc(x))



train_df['text']=train_df['text'].apply(lambda x : only_words(x))



train_df['text']=train_df['text'].apply(lambda x : clean_stop(x,stop))



train_df['text']=train_df['text'].str.lower()
stop = set(stopwords.words("english"))



test_df['text']=test_df['text'].apply(lambda x: remove_URL(x))



test_df['text']=test_df['text'].apply(lambda x: remove_html(x))



test_df['text']=test_df['text'].apply(lambda x: remove_emoji(x))



test_df['text']=test_df['text'].apply(lambda x : clean_nonalp(x))



test_df['text']=test_df['text'].apply(lambda x : clean_punc(x))



test_df['text']=test_df['text'].apply(lambda x : only_words(x))



test_df['text']=test_df['text'].apply(lambda x : clean_stop(x,stop))



test_df['text']=test_df['text'].str.lower()
missing_value_of_data(train_df)
missing_value_of_data(test_df)
train_df.tail()
test_df.tail()
text_clf = Pipeline([

                  ('vect',TfidfVectorizer(ngram_range=(1,3),min_df=5)),

                  ('clf',LogisticRegression(max_iter=1000))]



)

text_clf.fit(train_df.text.values,train_df.target.values)

pred = text_clf.predict(test_df.text.values)
test_df['target'] = pred
test_df.drop(['text'],axis=1).to_csv('submission.csv',index=False)
!pip install --upgrade pip

!pip install kaggle --upgrade
# run this chunk on console command (one by one)

# %env KAGGLE_USERNAME= your_kaggle_username #replace without punctuation

# %env KAGGLE_KEY= your_kaggle_API_Key #replace without punctuation

# !export -p | grep KAGGLE_USERNAME

# !export -p | grep KAGGLE_KEY

# !kaggle competitions submit -c nlp-getting-started -f submission.csv -m "Message"