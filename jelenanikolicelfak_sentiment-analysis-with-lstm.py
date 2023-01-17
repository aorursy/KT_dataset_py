# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns # biblioteka za grafički prikaz



import warnings

warnings.filterwarnings("ignore")



import re



from bs4 import BeautifulSoup

from tqdm import tqdm

from nltk.stem import WordNetLemmatizer



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers
all_data = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")

print("Količina podataka : ", all_data.shape)

all_data.head() # prikaz prvih 5 redova
data_filtered = all_data[all_data["Score"]!=3] #Uklanjanje neutralnih podataka, imaju Score = 3

data_filtered.shape
#pre promene

plt.figure(figsize = (20,10))

sns.countplot(data_filtered['Score'])

plt.title("Graficki prikaz odnosa sentimentnosti kritika")
data_filtered["Score"] = data_filtered["Score"].apply(lambda x : 1 if x>3 else 0)

data_filtered.head()

sorted_data=data_filtered.sort_values('ProductId', kind='quicksort', na_position='last') #sortiranje podataka po Id -u proizvoda

data_final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

data_final.shape
#posle promene

data_final['Score'].value_counts()
def decontract(text):

    text = re.sub(r"won\'t", "will not", text)

    text = re.sub(r"can\'t", "can not", text)

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)

    return text



#skup random stop reci (veznik)

stop_words= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
def preprocess_text(review):

    review = re.sub(r"http\S+", "", review)             # uklanjanje veb linkova

    review = BeautifulSoup(review, 'lxml').get_text()   # uklanjanje html tagova

    review = decontract(review)                         # decontracting

    review = re.sub("\S*\d\S*", "", review).strip()     # uklanjanje reci koji sadrze brojeve

    review = re.sub('[^A-Za-z]+', ' ', review)          # uklanjanje karaktera koji nisu reci

    review = review.lower()                             # pretvaranje sva slova u mala

    review = [word for word in review.split(" ") if not word in stop_words] # uklanjanje stop reci

    review = " ".join(review)

    review.strip()

    return review

data_final['Text'] = data_final['Text'].apply(lambda x: preprocess_text(x))

data_final['Text'].head()
train_df, test_df = train_test_split(data_final, test_size = 0.4, random_state = 42)

keep_col = ['Id','ProductId','Text','Score']

train_df = train_df[keep_col]

test_df = test_df[keep_col]

print("Podaci za treniranje : ", train_df.shape)

print("Podaci za testiranje: ", test_df.shape)
train_df.head()
test_df.head()
overlapped = pd.merge(train_df[["Text", "Score"]], test_df, on="Text", how="inner")

overlap_boolean_mask_test = test_df['Text'].isin(overlapped['Text'])
print("Dužina teksta u fajlu za treniranje")

sns.distplot(train_df['Text'].map(lambda ele: len(ele)), kde_kws={"label": "train"})



print("Dužina teksta u fajlu za treniranje")

sns.distplot(test_df[~overlap_boolean_mask_test]['Text'].map(lambda ele: len(ele)), kde_kws={"label": "test"})
top_words = 6000

tokenizer = Tokenizer(num_words=top_words)

tokenizer.fit_on_texts(train_df['Text'])

list_tokenized_train = tokenizer.texts_to_sequences(train_df['Text'])



max_review_length = 130

X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)

y_train = train_df['Score']

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train,y_train, epochs=2, batch_size=64, validation_split=0.2)
list_tokenized_test = tokenizer.texts_to_sequences(test_df['Text'])

X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)

y_test = test_df['Score']

prediction = model.predict(X_test)

y_pred = (prediction > 0.5)

print("Tačnost modela: ", accuracy_score(y_pred, y_test))