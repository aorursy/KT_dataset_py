import os
import re
import pandas as pd
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from importlib import reload
import sys
from imp import reload
import warnings
warnings.filterwarnings('ignore')
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
training_df = pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/train.csv')
testing_df=  pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/test.csv')


# df1 = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter="\t")
# df1 = df1.drop(['id'], axis=1)
# df1.head()
# df2 = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")
# df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)
# df2.columns = ["review","sentiment"]
# df2.head()
# df2 = df2[df2.sentiment != 'unsup']
# df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})
# df2.head()
# df = pd.concat([df1, df2]).reset_index(drop=True)
# df.head()
training_df.drop(["review_id","title","year"],axis=1,inplace=True)
training_df.head()
def rep(text):
    text = re.sub('♥♥♥♥',"worst bad horrible game",text)
    return text

training_df['user_review']=training_df.user_review.apply(rep)
testing_df['user_review']=testing_df.user_review.apply(rep)

def low(text):
    return text.lower()

training_df['user_review']=training_df.user_review.apply(low)
testing_df['user_review']=testing_df.user_review.apply(low)

def asc(text):
    text = re.sub('[^a-zA-Z]'," ",text)
    return text

training_df['user_review']=training_df.user_review.apply(asc)
testing_df['user_review']=testing_df.user_review.apply(asc)

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

training_df['Processed_Reviews'] = training_df.user_review.apply(lambda x: clean_text(x))
testing_df['Processed_Reviews'] = testing_df.user_review.apply(lambda x: clean_text(x))

testing_df['Processed_Reviews'] = testing_df.user_review.apply(lambda x: clean_text(x))
training_df.head()
training_df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 2000 #6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(training_df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(training_df['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = training_df['user_suggestion']

embed_size = 103 ###############################
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
# df_test=pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip",header=0, delimiter="\t", quoting=3)
# df_test.head()
testing_df.head()
# testing_df["user_review"]=df_test.review.apply(lambda x: clean_text(x))
# df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
# y_test = df_test["sentiment"]#######################################################################################
list_sentences_test = testing_df["Processed_Reviews"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
# from sklearn.metrics import f1_score, confusion_matrix
# print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
# print('Confusion matrix:')
# confusion_matrix(y_pred, y_test)

y_pred = y_pred.astype(int)
submission=pd.DataFrame(y_pred)
sub = pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/test.csv')
submission['review_id']=sub['review_id']
submission= submission[['review_id',0]]
submission.rename(columns = {0:'user_suggestion'}, inplace = True)
submission.to_csv('lstm.csv', index=False)
submission.head(4)
