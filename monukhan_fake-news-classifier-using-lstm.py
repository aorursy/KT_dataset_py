import pandas as pd

import numpy as np

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt



from wordcloud import WordCloud
df = pd.read_csv("../input/fake-news-dataset/train.csv")
df.head()
df.isnull().sum()
df = df.dropna() #  removing the missing values rows from dataset
df.info()


duplicateRowsDF = df[df.duplicated(["id","title","author","text"], keep="last")]

print("No of duplicates found:", duplicateRowsDF.shape[0])







#removing duplicates

# final_data = df.drop_duplicates(["id","title","author","text"], keep="first", inplace=False)
import nltk

import re

from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
text = " ".join([x for x in df.text])

wordcloud = WordCloud(background_color = 'white', stopwords = stop_words).generate(text)

plt.figure(figsize = (10,8))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.axis('off')

plt.show()
### for fake

text = " ".join([x for x in df.text[df.label == 1]])

wordcloud = WordCloud(background_color = 'white', stopwords = stop_words).generate(text)

plt.figure(figsize = (10,8))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.show()
### for Not fake (real)



text = " ".join([x for x in df.text[df.label == 0]])

wordcloud = WordCloud(background_color = 'white',  stopwords = stop_words).generate(text)

plt.figure(figsize = (10,8))

plt.imshow(wordcloud)
X = df.drop('label',axis = 1)
### dependent Features

y = df['label']
X.shape
y.shape
print('Number of 0 (not fake) :', df['label'].value_counts()[0])

print('Number of 1 (fake) :', df['label'].value_counts()[1])
sns.set_style('darkgrid')

plt.figure(figsize=(7,5))

sns.countplot(x='label',data=df)
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense
### voc size

voc_size  = 5000
messages = X.copy()
messages.reset_index(inplace = True)


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(0, len(messages)):



    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])  ### apart form a-z&A-Z replace(:,*:;) with ' '

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)

corpus[0] # list of sentences after stemming
onehot_repr = [one_hot(words, voc_size) for words in corpus]

onehot_repr[0]
# pre padding to create same size of sentences.



sent_length = 25

embedded_doc = pad_sequences(onehot_repr,padding = 'pre', maxlen= sent_length)

embedded_doc[0]
from tensorflow.keras.layers import Dropout

### creating model

embedding_vector_features = 40

model = Sequential()

model.add(Embedding(voc_size,embedding_vector_features, input_length = sent_length))

model.add(Dropout(0.3))

model.add(LSTM(200))

model.add(Dropout(0.3))

model.add(Dense(1,activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()
X_final = np.array(embedded_doc)

y_final = np.array(y)
X_final.shape
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state = 0)
model_history = model.fit(X_train,y_train, validation_data = (X_test,y_test), epochs = 10, batch_size  =64)
y_pred = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cm  = confusion_matrix(y_test, y_pred)



print(accuracy_score(y_test, y_pred),'\n')

print(classification_report(y_test, y_pred))
plt.title('LSTM Confusion Matrix')

sns.heatmap(cm,annot = True, cbar = False, fmt="d",cmap="Blues")
test_df = pd.read_csv('../input/fake-news-dataset/test.csv')

test_df.head()




test_df.isnull().sum()
test_df.shape
#the solution file that can be submitted in kaggle expects it to have 5200 rows so we can't drop rows in the test dataset

test_df.fillna('fake fake fake',inplace=True)





corpus_test = []

for i in range(0, len(test_df)):

    review = re.sub('[^a-zA-Z]', ' ',test_df['title'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus_test.append(review)


onehot_repr_test=[one_hot(words,voc_size)for words in corpus_test] 


sent_length=25



embedded_docs_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length)

print(embedded_docs_test)
X_test=np.array(embedded_docs_test)




check = model.predict_classes(X_test)
type(check)
check[0]
val = []

for i in check:

    val.append(i[0])
print('Loading the submissiion file')

submit_df = pd.read_csv('../input/fake-news-dataset/submit.csv')

print(submit_df.columns)

submit_df['label'] = val
#saving the submission file



submit_df.to_csv('submission.csv',index=False)