import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.manifold import TSNE

NB = MultinomialNB()





from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer()

vect = CountVectorizer()



import nltk

from nltk.corpus import stopwords

stopwords = stopwords.words('english')



from nltk.stem import PorterStemmer

stemmer = PorterStemmer()





import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df1 = train_df.drop(['keyword','location'], axis=1)

test_df1 = test_df.drop(['keyword','location'], axis=1)

print('train shape is {} and test shape is {}'.format(train_df1.shape,test_df1.shape))
### making the structure similar for both dataframe to merge. Also adding column to differentiate the data set

test_df1['target'], test_df1['flag'] = ' ', 'Data for test'

train_df1['flag'] = 'Data for train'
### Merge both data set vertically

train_test_df = pd.concat([train_df1,test_df1], ignore_index=True, sort=False)

print('combined data set shape is {}'.format(train_test_df.shape))

train_test_df.head()
## Checking sample tweets

print(train_test_df.loc[2,'text'])

print("Train and test dataset rows and columns {} ".format(train_test_df.shape))
print("Sample tweet which is not about disaster:-  " + train_test_df[train_test_df["target"] == 0]["text"].values[1])

print("Sample tweet which is about disaster    :-  " + train_test_df[train_test_df["target"] == 1]["text"].values[1])

fig, ax = plt.subplots()

train_data = train_test_df[train_test_df['flag']=='Data for train']

train_data['target_desc'] = train_data['target'].map({0:'No Disaster',1:'Disaster'}) 

train_data = train_data['target_desc'].value_counts()

train_data.head()



x= train_data.index

y= train_data.values

ax.bar(x,y)

plt.title("Train dataset histogram - 0:No Disaster , 1:Disaster")

plt.xlabel("Target")

plt.ylabel("Number of Recs")

plt.show()
import string

train_test_df['text'] = train_test_df['text'].str.replace('[{}]'.format(string.punctuation), '')  ## Remove punctuation



# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.

train_test_df['text_without_stop_words'] = train_test_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))



#To show select all the rows which has any digit train_test_df[train_test_df['text_without_stop_words'].str.contains(r'[0-9]')]



# Exclude numbers from the text

train_test_df['text_without_stop_words_and_number'] = train_test_df['text_without_stop_words'].apply(lambda x: ''.join([word for word in x if not word.isdigit()]))



#Splitting all the word in sentence

#train_test_df['text_without_stop_words_and_number_non_stemmed'] = train_test_df['text_without_stop_words_and_number'].str.split()



#Stemming the words

#train_test_df['text_without_stop_words_and_number_stemmed'] = train_test_df['text_without_stop_words_and_number_non_stemmed'].apply(lambda x: [stemmer.stem(y) for y in x])



#putting the stemmed word back together

#train_test_df['text_without_stop_words_and_number_stemmed'] = train_test_df['text_without_stop_words_and_number_stemmed'].apply(lambda x: ' '.join([word for word in x]))



train_test_df.head()
## Final column to be used for modelling is "text_without_stop_words_and_number" - Stemming has ruined lot of words

train_df2 = train_test_df[['target','text_without_stop_words_and_number']]

train_df2 = train_df2.rename(columns={'text_without_stop_words_and_number':'text'}) ## renaming the column to text

train_df2.head()

x = train_df2['text']



### Applying TfidfVectorizer

x_tfidf = tfid.fit_transform(x)

y = train_df2['target']



#### Dividing the final data set into original train and test data set

X_train_final = x_tfidf[:7613]

Y_train_final = y[:7613]



X_test_final = x_tfidf[7613:]

Y_test_final = y[7613:]



print("x_train {} and y_train {}".format(X_train_final.shape,Y_train_final.shape))

print("x_test {} and y_test {}".format(X_test_final.shape,Y_test_final.shape))
#### Tokenize examples

from nltk.tokenize import sent_tokenize, word_tokenize

from scipy import sparse

#print(sent_tokenize(train_df.loc[2,'text']))

#print(word_tokenize(train_df.loc[2,'text']))



x_train, x_test, y_train, y_test = train_test_split(X_train_final,Y_train_final, random_state=1)

print(sparse.csr_matrix(X_train_final[2]))

print('x train shape {} and x test shape {}'.format(x_train.shape, x_test.shape))
# Fitting Random Forest classifier with 100 trees to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

y_train=y_train.astype('int')

classifier.fit(x_train, y_train)
import sklearn.metrics as metrics

y_predict = classifier.predict(x_test)

y_test = y_test.astype('int')

y_true = np.asarray(y_test)

#np.unique(y_predict)

print("Accurecy score = {} ".format(metrics.accuracy_score(y_true,y_predict)))
y_test_predict = pd.DataFrame(list(zip(y_test,y_predict)), columns=['y_test','y_predict'])

y_test_predict.head()
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
#### Applying on the test data

sample_submission["target"] = classifier.predict(X_test_final)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)