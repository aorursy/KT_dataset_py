# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
color = sns.color_palette()

%matplotlib inline

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
## Read the train and test dataset and check the top few lines ##
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Number of rows & columns in train dataset : ",train_df.shape)
print("Number of rows $ columns in test dataset : ",test_df.shape)
train_df.head()
## check class imbalance
counts = train_df['author'].value_counts()
print(counts)
grouped_df = train_df.groupby('author')
print(grouped_df.head())
for name, group in grouped_df:
    print("\nAuthor name : ", name)
    cnt = 0
    for ind, row in group.iterrows():
        print(row["text"])
        cnt += 1
        if cnt == 5:
            break
    print("\n")
## Number of words in the text ##
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["unique_words"] = train_df["text"].apply(lambda x: len(pd.unique(str(x).split())))
test_df["unique_words"] = test_df["text"].apply(lambda x: len(pd.unique(str(x).split())))

## Number of stop words in the text ##
train_df["stop_words"] = train_df["text"].apply(lambda x: len([i for i in str(x).lower().split() if i in eng_stopwords]))
test_df["stop_words"] = test_df["text"].apply(lambda x: len([i for i in str(x).lower().split() if i in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

train_df["num_words"] = train_df["text"].apply(lambda x: np.mean([len(i) for i in str(x).split()]))
test_df["num_words"] = test_df["text"].apply(lambda x: np.mean([len(i) for i in str(x).split()]))

# summary statistics of the meta features
print("\nSummary of Training Set Numeric Variables\n")
print(train_df.groupby("author").mean())
print("\nSummary of Testing Set Numeric Variables\n",train_df.groupby("author").mean())
train_df['num_words'].loc[train_df['num_words']>80] = 80 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='author', y='num_words', data=train_df)
plt.xlabel('Author Name', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by author", fontsize=15)
plt.show()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics

encoder = LabelEncoder()
encoder.fit(train_df['author'])
train_y = encoder.transform(train_df['author'])

train_tfidf = train_df['text'].values.tolist()
test_tfidf = test_df['text'].values.tolist()

NBclassifier = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])
NBclassifier.fit(train_tfidf, train_y)

#Predict training set:
dtrain_predictions = NBclassifier.predict_proba(train_tfidf)

#Predict testing set:
y_pred_proba = NBclassifier.predict_proba(test_tfidf)

#Perform cross-validation:
cv_score = cross_validation.cross_val_score(NBclassifier, train_tfidf, train_y, cv=5, scoring='roc_auc')  
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
#id,EAP,HPL,MWS
#id07943,0.33,0.33,0.33
x = y_pred_proba[:,0]

my_submission = pd.DataFrame({'id': test_df['id'], 'EAP': y_pred_proba[:,0], 
                              'HPL': y_pred_proba[:,1], 'MWS': y_pred_proba[:,2] })
my_submission.to_csv('submission.csv', index=False)





