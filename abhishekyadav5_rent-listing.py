import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd



import copy



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



from sklearn.model_selection import train_test_split



import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk import word_tokenize



import re



import gensim



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.models import Model, Sequential

from keras.layers import Activation, Dense, LSTM, Dropout, Embedding, Input

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers.merge import concatenate
train = pd.read_json("../input/train.json")
train.head()
train.reset_index(inplace=True)
# Dropping old index

train = train.drop(["index"], axis=1)
train.info()
train
f, ax = plt.subplots(dpi = 70, figsize=[10,6])



train["bathrooms"].value_counts().plot(kind='bar',color='brown')



for i,v in enumerate(train["bathrooms"].value_counts().values):

    plt.text(i,v, str(v), color='black')
f, ax = plt.subplots(dpi = 70, figsize=[10,6])



train["bedrooms"].value_counts().plot(kind='bar',color='brown')



for i,v in enumerate(train["bedrooms"].value_counts().values):

    plt.text(i,v, str(v), color='black')
train["building_id"].value_counts()
building_df = train.groupby(["building_id", "interest_level"])["interest_level"].count().unstack("interest_level").fillna(0)

building_df["sum"] = building_df.sum(axis=1)
building_df = building_df[building_df.index != '0']
building_df.sort_values('sum', inplace=True, ascending=False)
building_df
building_df100 = building_df[(building_df["sum"] > 100)]
building_df100[["low", "medium", "high"]].plot(kind="barh", figsize=[7,15], stacked=True)
train["created"] = pd.to_datetime(train["created"])
train['date'] = train['created'].dt.date

train["year"] = train["created"].dt.year

train['month'] = train['created'].dt.month

train['day'] = train['created'].dt.day

train['hour'] = train['created'].dt.hour

train['day_of_week'] = train['created'].dt.weekday

train['weekend'] = ((train['day_of_week'] == 5) | (train['day_of_week'] == 6))

train['weekday'] = ((train['day_of_week'] != 5) & (train['day_of_week'] != 6))

train
date = train["date"].value_counts()
f,ax = plt.subplots(figsize=[14,6])

ax.bar(date.index, date.values, color="plum")

ax.xaxis_date()

plt.xticks(rotation=90)

plt.show()
hour_df = train.groupby(['hour', 'interest_level'])['hour'].count().unstack('interest_level').fillna(0)
hour_df[["low", "medium", "high"]].plot(kind="bar", stacked=True, figsize=(12,5))
dayweek_df = train.groupby(['day_of_week', 'interest_level'])['day_of_week'].count().unstack('interest_level').fillna(0)
dayweek_df[["low","medium","high"]].plot(kind="bar", stacked=True)
day_df = train.groupby(["day", "interest_level"])["day"].count().unstack("interest_level").fillna(0)
day_df[["low", "medium", "high"]].plot(kind="bar", stacked=True, figsize=[12,5])
train["display_address"].value_counts()
display_add_df = train.groupby(["display_address", "interest_level"])["display_address"].count().unstack("interest_level").fillna(0)
display_add_df["sum"] = display_add_df.sum(axis=1)
display_add_df100 = display_add_df[(display_add_df["sum"] > 100)]

display_add_df100.sort_values("sum", inplace=True)

display_add_df100[["low", "medium", "high"]].plot(kind="barh" ,figsize=[7,25], stacked=True)
latilongi = train[["latitude", "longitude", "interest_level"]]

latilongi = latilongi[(latilongi["latitude"] != 0) & (latilongi["longitude"] != 0)]

f, ax = plt.subplots(figsize=[15,15])



sns.scatterplot(x = "latitude", y = "longitude", hue="interest_level", data=latilongi)
manager_df = train.groupby(["manager_id", "interest_level"])["manager_id"].count().unstack("interest_level").fillna(0)
manager_df["sum"] = manager_df.sum(axis=1)
manager_df.sort_values("sum", inplace=True, ascending=False)
manager_df100 = manager_df[(manager_df["sum"] > 100)]
manager_df100[["low", "medium", "high"]].plot(kind="barh", stacked=True, figsize=[7 ,20])
train
train["photo_len"] = train["photos"].apply(len)
sns.stripplot(train["interest_level"], train["photo_len"], jitter=True)
train["price"].value_counts()
sns.stripplot(train["interest_level"], train["price"], jitter=True)
#removing outliers

Q = train["price"].quantile(0.99)

train = train[(train["price"] <= Q)]
sns.stripplot(train["interest_level"], train["price"], jitter=True)
sns.distplot(train["price"], kde=False)
# price distribution interest-wise



f, ax = plt.subplots(figsize=[9,5])



train_price_low = train[(train["interest_level"] == "low")]

sns.distplot(train_price_low["price"], kde=False, color="seagreen")



train_price_medium = train[(train["interest_level"] == "medium")]

sns.distplot(train_price_medium["price"], kde=False, color="orange")



train_price_high = train[(train["interest_level"] == "high")]

sns.distplot(train_price_high["price"], kde=False, color="blue")
train
street_df = train.groupby(["street_address", "interest_level"])["street_address"].count().unstack("interest_level").fillna(0)
street_df["sum"] = street_df.sum(axis=1)
street_df = street_df[street_df.index != "0"]
street_df.sort_values("sum", inplace=True, ascending=False)
street_df70 = street_df[(street_df["sum"] > 70)]
street_df70[["low", "medium", "high"]].plot(kind="barh", stacked=True, figsize=[6,10])
# removing weekday, weekend, photos, listing_id, created



train.drop(["weekday", "weekend", "photos", "listing_id", "created"], axis=1, inplace=True)



# and date



train.drop(["date"], axis=1, inplace=True)
train
building_id_dict = train["building_id"].value_counts().to_dict()
# This takes more than 4 hours. We have to find another way.



# %%time



# for key, value in building_id_dict.items():

#     for x,y in enumerate(train["building_id"]):

#         if y == key:

#             if value >= 250:

#                 train["building_id"][x] = "building_6"

#             elif value >= 200 & value < 250:

#                 train["building_id"][x] = "building_5"

#             elif value >= 150 & value < 200:

#                 train["building_id"][x] = "building_4"

#             elif value >= 100 & value < 150:

#                 train["building_id"][x] = "building_3"

#             elif value >= 60 & value < 100:

#                 train["building_id"][x] = "building_2"

#             elif value >= 20 & value < 60:

#                 train["building_id"][x] = "building_1"

#             else:

#                 train["building_id"][x] = "building_0"

                
for key, value in building_id_dict.items():

    if (value >= 250):

        building_id_dict[key] = "building_6"

    elif (value >= 200) & (value < 250):

        building_id_dict[key] = "building_5"

    elif (value >= 150) & (value < 200):

        building_id_dict[key] = "building_4"

    elif (value >= 100) & (value < 150):

        building_id_dict[key] = "building_3"

    elif (value >= 60) & (value < 100):

        building_id_dict[key] = "building_2"

    elif (value >= 20) & (value < 60):

        building_id_dict[key] = "building_1"

    else:

        building_id_dict[key] = "building_0"
train["building_id"] = train["building_id"].map(building_id_dict)
# one hot encoding the building_id column



one_hot_building = pd.get_dummies(train["building_id"], drop_first=True)

train = train.join(one_hot_building)
train = train.drop("building_id", axis=1)
disp_address_dict = train["display_address"].value_counts().to_dict()
for key, value in disp_address_dict.items():

    if (value >= 300):

        disp_address_dict[key] = "DA_8"

    elif (value >= 250) & (value < 300):

        disp_address_dict[key] = "DA_7"

    elif (value >= 200) & (value < 250):

        disp_address_dict[key] = "DA_6"

    elif (value >= 150) & (value < 200):

        disp_address_dict[key] = "DA_5"

    elif (value >= 100) & (value < 150):

        disp_address_dict[key] = "DA_4"

    elif (value >= 50) & (value < 100):

        disp_address_dict[key] = "DA_3"

    elif (value >= 30) & (value < 50):

        disp_address_dict[key] = "DA_2"

    elif (value >= 10) & (value < 30):

        disp_address_dict[key] = "DA_1"

    else:

        disp_address_dict[key] = "DA_0"
train["display_address"] = train["display_address"].map(disp_address_dict)
# one hot encoding the display address column



one_hot_DA = pd.get_dummies(train["display_address"])

train = train.join(one_hot_DA)
train = train.drop("display_address", axis=1)
manager_id_dict = train["manager_id"].value_counts().to_dict()
manager_id_dict
for key, value in manager_id_dict.items():

    if (value >= 1000):

        manager_id_dict[key] = "manager_9"

    elif (value >= 500) & (value < 1000):

        manager_id_dict[key] = "manager_8"

    elif (value >= 300) & (value < 500):

        manager_id_dict[key] = "manager_7"

    elif (value >= 200) & (value < 300):

        manager_id_dict[key] = "manager_6"

    elif (value >= 100) & (value < 200):

        manager_id_dict[key] = "manager_5"

    elif (value >= 70) & (value < 100):

        manager_id_dict[key] = "manager_4"

    elif (value >= 40) & (value < 70):

        manager_id_dict[key] = "manager_3"

    elif (value >= 20) & (value < 40):

        manager_id_dict[key] = "manager_2"

    elif (value >= 10) & (value < 20):

        manager_id_dict[key] = "manager_1"

    else:

        manager_id_dict[key] = "manager_0"
train["manager_id"] = train["manager_id"].map(manager_id_dict)
# one hot encoding the manager_id column



one_hot_manager = pd.get_dummies(train["manager_id"])

train = train.join(one_hot_manager)
train = train.drop("manager_id", axis=1)
street_address_dict = train["street_address"].value_counts().to_dict()
for key,value in street_address_dict.items():

    if (value >= 150):

        street_address_dict[key] = "street_8"

    elif (value >= 100) & (value < 150):

        street_address_dict[key] = "street_7"

    elif (value >= 60) & (value < 100):

        street_address_dict[key] = "street_6"

    elif (value >= 40) & (value < 60):

        street_address_dict[key] = "street_5"

    elif (value >= 25) & (value < 40):

        street_address_dict[key] = "street_4"

    elif (value >= 15) & (value < 25):

        street_address_dict[key] = "street_3"

    elif (value >= 10) & (value < 15):

        street_address_dict[key] = "street_2"

    elif (value >= 5) & (value < 10):

        street_address_dict[key] = "street_1"

    else:

        street_address_dict[key] = "street_0"
train["street_address"] = train["street_address"].map(street_address_dict)
# one hot encoding the street_address column



one_hot_street = pd.get_dummies(train["street_address"])

train = train.join(one_hot_street)
train = train.drop("street_address", axis=1)
interest_map = {"low":0, "medium":1, "high":2}
train["interest_level"] = train["interest_level"].map(interest_map)
# dropping year column



train = train.drop("year", axis=1)
train["features_len"] = train["features"].apply(len)
train
train_set, validation_set = train_test_split(train, random_state=26, test_size=0.05)
stopword = stopwords.words('english')

stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):

    text = re.sub(r"<.*>", " ", str(text).lower()).strip()

    text = re.sub(r"<.*", " ", str(text).lower()).strip()

    text = re.sub(r"w/", " with", str(text).lower()).strip()

    text = re.sub(r'[^a-zA-Z]', " ", str(text).lower()).strip() #removing sp_char

    

    tokens = []

    for token in text.split():

        if token not in stopword:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    

    return " ".join(tokens)
train_set.description = train_set.description.apply(lambda x: preprocess(x))
documents = [text.split() for text in train_set.description]
len(documents)
w2v_model = gensim.models.word2vec.Word2Vec(size = 256, window = 5, min_count = 6, workers = 4)
%%time

w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=len(documents), epochs=32)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_set.description)
len(tokenizer.word_index)
x_train = pad_sequences(tokenizer.texts_to_sequences(train_set.description), maxlen=800, padding="post", truncating="post")
y_train = train_set.interest_level.tolist()
y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)
# total features in feature column



count=0

for x in train_set["features"]:

    y = len(x)

    count = count+y



print(count)
# reseting index of train and validation set



train_set = train_set.reset_index()

validation_set = validation_set.reset_index()
train_set = train_set.drop("index", axis=1)

validation_set = validation_set.drop("index", axis=1)
%%time

new = []

for i in range(len(train_set)):

    a = " ".join(train_set.features[i])

    new.append(a)
# replacing the old features column with the new list



train_set = train_set.drop("features", axis=1)
train_set["features"] = new
train_set.description = train_set.description.apply(lambda x: preprocess(x))
documents_f = [text.split() for text in train_set.features]
len(documents_f)
w2v_model_f = gensim.models.word2vec.Word2Vec(size = 256, window = 2, min_count = 1, workers = 4)
%%time

w2v_model_f.build_vocab(documents_f)
w2v_model_f.train(documents_f, total_examples=len(documents_f), epochs=32)
tokenizer_f = Tokenizer()

tokenizer_f.fit_on_texts(train_set.features)
len(tokenizer_f.word_index)
x_train_f = pad_sequences(tokenizer_f.texts_to_sequences(train_set.features), maxlen=250, padding="post", truncating="post")
train_set_final = train_set.drop(["description", "features"], axis=1)
train_set_final.head()
# shrinking every column between 0 and 1 



def shrink(one):

    most = max(one)

    one = one/most

    return one
train_set_final["bedrooms"] = shrink(train_set_final["bedrooms"])
train_set_final["bathrooms"] = shrink(train_set_final["bathrooms"])

train_set_final["price"] = shrink(train_set_final["price"])

train_set_final["month"] = shrink(train_set_final["month"])

train_set_final["day"] = shrink(train_set_final["day"])

train_set_final["hour"] = shrink(train_set_final["hour"])

train_set_final["day_of_week"] = shrink(train_set_final["day_of_week"])

train_set_final["photo_len"] = shrink(train_set_final["photo_len"])

train_set_final["features_len"] = shrink(train_set_final["features_len"])
train_set_final["latitude"] = train_set_final["latitude"]/100

train_set_final["longitude"] = train_set_final["longitude"]/100
train_set_final = train_set_final.drop("interest_level", axis=1)
vocab_size_1 = len(tokenizer.word_index) + 1
embedding_matrix_1 = np.zeros((vocab_size_1, 256))

for word, i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix_1[i] = w2v_model.wv[word]

print(embedding_matrix_1.shape)
vocab_size_2 = len(tokenizer_f.word_index) + 1
embedding_matrix_2 = np.zeros((vocab_size_2, 256))

for word, i in tokenizer_f.word_index.items():

    if word in w2v_model_f.wv:

        embedding_matrix_2[i] = w2v_model_f.wv[word]

print(embedding_matrix_2.shape)
# Input_1 = Embedding(vocab_size_1,256, weights=[embedding_matrix_1], input_length=800, trainable=False)

# hidden_11 = Dropout(0.5)(Input_1)

# hidden_11 = Input(sh(Input_1)

# hidden_12 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(hidden_11)

# hidden_13 = Dense(10, activation="relu")(hidden_12)



# Input_2 = Embedding(vocab_size_2,256, weights=[embedding_matrix_1], input_length=250, trainable=False)

# hidden_21 = Dropout(0.5)(Input_2)

# hidden_21 = Input(shape=(256,))(Input_2)

# hidden_22 = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(hidden_21)

# hidden_23 = Dense(10, activation="relu")(hidden_22)



# Input_3 = Input(shape=(45,))

# hidden_31 = Dense(40, activation="relu")(Input_3)

# hidden_32 = Dense(20, activation="relu")(hidden_31)

# hidden_33 = Dense(10, activation="relu")(hidden_32)



# merge = concatenate([hidden_12, hidden_22, hidden_33])



# hidden_4 = Dense(10, activation="relu")(merge)

# hidden_5 = Dense(10, activation="relu")(hidden_4)

# output = Dense(1, activation="softmax")(hidden_5)



# model = Model(inputs=[hidden_11, hidden_21, Input_3], outputs=output)
Input_1 = Embedding(vocab_size_1,256, weights=[embedding_matrix_1], input_length=800, trainable=False)



model_1 = Sequential()

model_1.add(Input_1)

model_1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model_1.add(Dense(3, activation='softmax'))
Input_2 = Embedding(vocab_size_2,256, weights=[embedding_matrix_2], input_length=250, trainable=False)



model_2 = Sequential()

model_2.add(Input_2)

model_2.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model_2.add(Dense(3, activation="softmax"))
model_3 = Sequential()

model_3.add(Dense(20, input_dim=45, activation="relu"))

model_3.add(Dense(10, activation="relu"))

model_3.add(Dense(3, activation="softmax"))