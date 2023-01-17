import warnings

warnings.filterwarnings("ignore")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import spacy

import sys

import os

import copy



%matplotlib inline
import tensorflow as tf

import tensorflow_hub as hub



from tensorflow.keras.callbacks import Callback

import tensorflow.keras.layers as layers

from tensorflow.keras.layers import Input, Embedding, Flatten, BatchNormalization, LSTM, Dense, Concatenate, Bidirectional 

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Reshape, GRU, GlobalMaxPooling1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.models import Model, Sequential

import tensorflow.keras.backend as K



from sklearn.metrics import f1_score, accuracy_score, precision_score

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# !python -m spacy download en
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_data.head(10)
train_data.shape
train_data.describe(include=[np.number])
train_data.describe(include=[np.object])
train_data['keyword'] = train_data['keyword'].fillna('Unset', axis = 0)

train_data['location'] = train_data['location'].fillna('Unset', axis = 0)
train_data.head()
train_data.describe(include=[np.object])
real_num = train_data[train_data['target'] == 1].shape[0]

fake_num = train_data[train_data['target'] == 0].shape[0]



plt.figure(figsize = (7, 5))



labels = ["Real", "Fake"]

x = np.arange(len(labels))



plt.bar(x, [real_num, fake_num])

plt.xticks(x, labels)

plt.xlabel('Target')

plt.ylabel('Number of examples')

plt.title('Distribution of target classes')

plt.show()
disaster_len = [len(i) for i in train_data[train_data['target'] == 1]['text']]

fake_len = [len(i) for i in train_data[train_data['target'] == 0]['text']]
print("Max length of disaster tweets:", max(disaster_len))

print("Max length of fake tweets:", max(fake_len))
plt.hist(disaster_len, 10, facecolor='blue', edgecolor='black', linewidth=1.2)

plt.xlabel('Text lenght')

plt.ylabel('Number of examples')

plt.title('Distribution of text\'s lenghts for disaster')

plt.show()
plt.hist(fake_len, 10, facecolor='red', edgecolor='black', linewidth=1.2)

plt.xlabel('Text lenght')

plt.ylabel('Number of examples')

plt.title('Distribution of text\'s lenghts for fake')

plt.show()
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_data.head(10)
test_data.shape
test_data.describe(include=[np.number])
test_data.describe(include=[np.object])
test_data['keyword'] = test_data['keyword'].fillna('Unset', axis = 0)

test_data['location'] = test_data['location'].fillna('Unset', axis = 0)
test_data.head()
test_data.describe(include=[np.object])
text_len = [len(i) for i in test_data['text']]
print("Max length of test tweets:", max(text_len))
plt.hist(text_len, 10, facecolor='blue', edgecolor='black', linewidth=1.2)

plt.xlabel('Test text lenght')

plt.ylabel('Number of examples')

plt.title('Distribution of text\'s lenghts for test dataset')

plt.show()
MAX_LEN = 157
full_data = pd.read_csv('/kaggle/input/real-or-not/all.csv', encoding = "ISO-8859-1")

full_data.head(5)
full_data['id'] = full_data.index

full_data['target'] = (full_data['choose_one'] == 'Relevant').astype(int)
full_data = full_data[['id', 'keyword', 'location', 'text', 'target']]

full_data.head(10)
real_true = pd.merge(test_data, full_data, on='id')

real_true.head()
to_submit = real_true[['id', 'target']]

to_submit.head()
os.mkdir('result')

real_true.to_csv('result/real_true.csv', index=False)
test_data_new = real_true[['id', 'target', 'keyword_x', 'location_x', 'text_x']]

test_data_new.head()
test_data_new = test_data_new.rename(columns={"keyword_x": "keyword", "location_x": "location", "text_x": "text"})

test_data_new.head()
def clearing_data(data):

    model_lingv = spacy.load("en")

    discard = {'PUNCT','SPACE','SYM','NUM'}



    processed_text = []



    n = len(data["text"])

    for i in range(0, n):

        text = data["text"][i]

        text = text.lower()

        doc = model_lingv(text)

        words = []

        for w in doc:

            if w.pos_ not in discard and not w.like_email and not w.like_url and not w.like_num:

                words.append(w.string)

        t = ' '.join(words)

        processed_text.append(t)



        sys.stdout.write("\r%f%%" % (100*(i+1)/n))

        sys.stdout.flush()



    data["clearing_text"] = processed_text

    return data
train_data = clearing_data(train_data)

print("\nTrain data was clearing!")

test_data = clearing_data(test_data_new)

print("\nTest data was clearing!")
train_data.head()
test_data.head()
def prepare_data(data, test=False, max_text_lenth = MAX_LEN):

    text = data['clearing_text'].tolist()

    text = [' '.join(t.split()[0:max_text_lenth]) for t in text]

    text = np.array(text)

    

    label = []

    if not test:

        label = data['target'].tolist()

    label = np.array(label)

    return text, label
train_text, train_label = prepare_data(train_data)

test_text, test_label = prepare_data(test_data_new)
use_embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
train_embeddings = use_embedding(train_text)['outputs'].numpy()

test_embeddings = use_embedding(test_text)['outputs'].numpy()
svc = SVC()

svc.fit(train_embeddings, train_label)



predict_label = svc.predict(test_embeddings)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
c_array = np.linspace(1.0, 2.0, num=7)

gamma_array = np.linspace(0.0, 1.0, num=7)
svc = SVC()

grid = GridSearchCV(svc, n_jobs = -1, verbose = 2, param_grid = {'C': c_array, 'gamma': gamma_array})

grid.fit(train_embeddings, train_label)

pass
best_C = grid.best_estimator_.C

best_gamma = grid.best_estimator_.gamma

print("C =", best_C)

print("Gamma =", best_gamma)
svc = SVC(C = best_C, gamma = best_gamma).fit(train_embeddings, train_label)



predict_label = svc.predict(test_embeddings)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label

for_submit = for_submit[['id', 'target']]

    

file_name = 'submit_SVC.csv'

for_submit.to_csv('result/' + file_name, index=False)
# Thank you for idea https://www.kaggle.com/dmitri9149/transformer-svm-semantically-identical-tweets/notebook

aggregation = copy.deepcopy(train_data)



aggregation = aggregation.groupby('keyword').agg({'target':np.mean}).rename(columns={'target':'keyword_prob'})

aggregation = aggregation.sort_values('keyword_prob', ascending=False).head(20)



min_prob = 0.90

keyword_disaster = list(aggregation[aggregation['keyword_prob'] >= min_prob].index)

keyword_disaster
# Do correction    

for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label



id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

for_submit = for_submit[['id', 'target']]



file_name = 'submit_SVC_corrected.csv'

for_submit.to_csv('result/' + file_name, index=False)
embedding = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder-large/5', trainable=False)
def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    

    embedding_layer = embedding(input_text)

    

    word_embedding = BatchNormalization()(embedding_layer)

    

    x = Dense(128, activation='relu')(word_embedding)

    x2 = Dense(64, activation='relu')(x)

    

    predict = Dense(units = 1, activation='sigmoid')(x2)

    

    model = Model(inputs=[input_text], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
# Function for saving model

def save_result(model, model_name):    

    predict_label = np.asarray(model.predict(test_text, verbose=1)).round()

    predict_label = predict_label.astype(int)

    

    print("F1-score =", f1_score(test_label, predict_label))

    

    for_submit = copy.deepcopy(test_data)

    for_submit["target"] = predict_label

    for_submit = for_submit[['id', 'target']]

    

    file_name = 'submit_' + model_name + '.csv'

    for_submit.to_csv('result/' + file_name, index=False)

    

    # Do correction    

    for_submit = copy.deepcopy(test_data)

    for_submit["target"] = predict_label

    

    id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

    for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

    for_submit = for_submit[['id', 'target']]

    

    print("With correction F1-score =", f1_score(test_label, for_submit['target'].tolist()))

    

    file_name = 'submit_' + model_name + '_corrected.csv'

    for_submit.to_csv('result/' + file_name, index=False)
history = model.fit(train_text, train_label, validation_data=(test_text, test_label), 

                    initial_epoch = 0, epochs=5)

save_result(model, 'FC_v1')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    

    embedding_layer = embedding(input_text)

    

    word_embedding = BatchNormalization()(embedding_layer)

    

    x = Dense(512, activation='relu')(word_embedding)

    x2 = Dense(256, activation='relu')(x)

    x3 = Dense(64, activation='relu')(x2)

    

    predict = Dense(units = 1, activation='sigmoid')(x3)

    

    model = Model(inputs=[input_text], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit(train_text, train_label, validation_data=(test_text, test_label), 

                    initial_epoch = 0, epochs=7)

save_result(model, 'FC_v2')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    

    embedding_layer = embedding(input_text)

    

    word_embedding = BatchNormalization()(embedding_layer)

    

    x = Dense(1024, activation='relu')(word_embedding)

    x2 = Dense(512, activation='relu')(x)

    x3 = Dense(128, activation='relu')(x2)

    x4 = Dense(64, activation='relu')(x3)

    

    predict = Dense(units = 1, activation='sigmoid')(x4)

    

    model = Model(inputs=[input_text], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit(train_text, train_label, validation_data=(test_text, test_label), 

                    initial_epoch = 0, epochs=5)

save_result(model, 'FC_v3')
def prepare_data_2(data):

    processed_text = []

    text = data['clearing_text'].tolist()

    keywords = data['keyword'].tolist()

    text = [keywords[i] + " " + text[i] for i in range(0, len(text))]

    text = np.array(text)

    return text
train_text_new = prepare_data_2(train_data)

test_text_new = prepare_data_2(test_data_new)
train_embeddings = use_embedding(train_text_new)['outputs'].numpy()

test_embeddings = use_embedding(test_text_new)['outputs'].numpy()
svc = SVC()

svc.fit(train_embeddings, train_label)



predict_label = svc.predict(test_embeddings)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
c_array = np.linspace(1.0, 2.0, num=7)

gamma_array = np.linspace(0.0, 1.0, num=7)
svc = SVC()

grid = GridSearchCV(svc, n_jobs = -1, verbose = 2, param_grid = {'C': c_array, 'gamma': gamma_array})

grid.fit(train_embeddings, train_label)

pass
best_C = grid.best_estimator_.C

best_gamma = grid.best_estimator_.gamma

print("C =", best_C)

print("Gamma =", best_gamma)
svc = SVC(C = best_C, gamma = best_gamma).fit(train_embeddings, train_label)



predict_label = svc.predict(test_embeddings)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label

for_submit = for_submit[['id', 'target']]

    

file_name = 'submit_SVC_keyword.csv'

for_submit.to_csv('result/' + file_name, index=False)
# Do correction    

for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label



id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

for_submit = for_submit[['id', 'target']]



file_name = 'submit_SVC_keyword_corrected.csv'

for_submit.to_csv('result/' + file_name, index=False)
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    

    embedding_layer = embedding(input_text)

    

    word_embedding = BatchNormalization()(embedding_layer)

    

    x = Dense(128, activation='relu')(word_embedding)

    x2 = Dense(64, activation='relu')(x)

    

    predict = Dense(units = 1, activation='sigmoid')(x2)

    

    model = Model(inputs=[input_text], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit(train_text_new, train_label, validation_data=(test_text_new, test_label), 

                    initial_epoch = 0, epochs=5)

save_result(model, 'FC_v1_keyword')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    

    embedding_layer = embedding(input_text)

    

    word_embedding = BatchNormalization()(embedding_layer)

    

    x = Dense(512, activation='relu')(word_embedding)

    x2 = Dense(256, activation='relu')(x)

    x3 = Dense(64, activation='relu')(x2)

    

    predict = Dense(units = 1, activation='sigmoid')(x3)

    

    model = Model(inputs=[input_text], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit(train_text_new, train_label, validation_data=(test_text_new, test_label), 

                    initial_epoch = 0, epochs=5)

save_result(model, 'FC_v2_keyword')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    

    embedding_layer = embedding(input_text)

    

    word_embedding = BatchNormalization()(embedding_layer)

    

    x = Dense(1024, activation='relu')(word_embedding)

    x2 = Dense(512, activation='relu')(x)

    x3 = Dense(128, activation='relu')(x2)

    x4 = Dense(64, activation='relu')(x3)

    

    predict = Dense(units = 1, activation='sigmoid')(x4)

    

    model = Model(inputs=[input_text], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit(train_text_new, train_label, validation_data=(test_text_new, test_label), 

                    initial_epoch = 0, epochs=5)

save_result(model, 'FC_v3_keyword')
def vectorization_keyword(data):

    vectorization = pd.get_dummies(data['keyword'])

    vectorization = vectorization.drop('Unset', 1)

    

    keywords = [vectorization.loc[i].tolist() for i in range(0, vectorization.shape[0])]

    keywords = np.array(keywords)

    

    return keywords
train_keyword = vectorization_keyword(train_data)

test_keyword = vectorization_keyword(test_data_new)
train_embeddings = use_embedding(train_text)['outputs'].numpy()

test_embeddings = use_embedding(test_text)['outputs'].numpy()
train_emb_keyw = [np.concatenate((train_embeddings[i], train_keyword[i])) for i in range(0, len(train_keyword))]

test_emb_keyw = [np.concatenate((test_embeddings[i], test_keyword[i])) for i in range(0, len(test_keyword))]
svc = SVC()

svc.fit(train_emb_keyw, train_label)



predict_label = svc.predict(test_emb_keyw)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
c_array = np.linspace(1.0, 2.0, num=5)

gamma_array = np.linspace(0.0, 1.0, num=5)
svc = SVC()

grid = GridSearchCV(svc, n_jobs = -1, verbose = 2, param_grid = {'C': c_array, 'gamma': gamma_array})

grid.fit(train_emb_keyw, train_label)

pass
best_C = grid.best_estimator_.C

best_gamma = grid.best_estimator_.gamma

print("C =", best_C)

print("Gamma =", best_gamma)
svc = SVC(C = best_C, gamma = best_gamma).fit(train_emb_keyw, train_label)



predict_label = svc.predict(test_emb_keyw)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label

for_submit = for_submit[['id', 'target']]

    

file_name = 'submit_SVC_OHE_keyword.csv'

for_submit.to_csv('result/' + file_name, index=False)
# Do correction    

for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label



id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

for_submit = for_submit[['id', 'target']]



file_name = 'submit_SVC_OHE_keyword_corrected.csv'

for_submit.to_csv('result/' + file_name, index=False)
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    embedding_layer = embedding(input_text)

    word_embedding = BatchNormalization()(embedding_layer)

    

    input_vector = layers.Input(shape=(221))

    vector_norm = BatchNormalization()(input_vector)

    

    x = Concatenate()([word_embedding,vector_norm])

    

    x2 = Dense(128, activation='relu')(x)

    x3 = Dense(64, activation='relu')(x2)

    

    predict = Dense(units = 1, activation='sigmoid')(x3)

    

    model = Model(inputs=[input_text, input_vector], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit([train_text, train_keyword], train_label, validation_data=([test_text, test_keyword], test_label), 

                    initial_epoch = 0, epochs=5)
def save_result(model, model_name):    

    predict_label = np.asarray(model.predict([test_text, test_keyword], verbose=1)).round()

    predict_label = predict_label.astype(int)

    

    print("F1-score =", f1_score(test_label, predict_label))

    

    for_submit = copy.deepcopy(test_data)

    for_submit["target"] = predict_label

    for_submit = for_submit[['id', 'target']]

    

    file_name = 'submit_' + model_name + '.csv'

    for_submit.to_csv('result/' + file_name, index=False)

    

    # Do correction    

    for_submit = copy.deepcopy(test_data)

    for_submit["target"] = predict_label

    

    id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

    for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

    for_submit = for_submit[['id', 'target']]

    

    print("With correction F1-score =", f1_score(test_label, for_submit['target'].tolist()))

    

    file_name = 'submit_' + model_name + '_corrected.csv'

    for_submit.to_csv('result/' + file_name, index=False)
save_result(model, 'FC_v1_OHE_keyword')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    embedding_layer = embedding(input_text)

    word_embedding = BatchNormalization()(embedding_layer)

    

    input_vector = layers.Input(shape=(221))

    vector_norm = BatchNormalization()(input_vector)

    

    x = Concatenate()([word_embedding,vector_norm])

    

    x2 = Dense(512, activation='relu')(x)

    x3 = Dense(256, activation='relu')(x2)

    x4 = Dense(64, activation='relu')(x3)

    

    predict = Dense(units = 1, activation='sigmoid')(x4)

    

    model = Model(inputs=[input_text, input_vector], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit([train_text, train_keyword], train_label, validation_data=([test_text, test_keyword], test_label), 

                    initial_epoch = 0, epochs=7)

save_result(model, 'FC_v2_OHE_keyword')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    embedding_layer = embedding(input_text)

    word_embedding = BatchNormalization()(embedding_layer)

    

    input_vector = layers.Input(shape=(221))

    vector_norm = BatchNormalization()(input_vector)

    

    x = Concatenate()([word_embedding,vector_norm])

    

    x2 = Dense(1024, activation='relu')(x)

    x3 = Dense(512, activation='relu')(x2)

    x4 = Dense(128, activation='relu')(x3)

    x5 = Dense(64, activation='relu')(x4)

    

    predict = Dense(units = 1, activation='sigmoid')(x5)

    

    model = Model(inputs=[input_text, input_vector], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit([train_text, train_keyword], train_label, validation_data=([test_text, test_keyword], test_label), 

                    initial_epoch = 0, epochs=9)

save_result(model, 'FC_v3_OHE_keyword')
def vectorization_location(data_train, data_test):

    

    full = pd.concat([data_train, data_test], axis=0)

    

    vectorization = pd.get_dummies(full['location'])

    vectorization = vectorization.drop('Unset', 1)

    

    train = vectorization[:train_data.shape[0]]

    test = vectorization[train_data.shape[0]:]

    

    locations_train = [train.loc[i].tolist() for i in range(0, train.shape[0])]

    locations_train = np.array(locations_train)

    

    locations_test = [test.loc[i].tolist() for i in range(0, test.shape[0])]

    locations_test = np.array(locations_test)

    

    return locations_train, locations_test
train_location, test_location = vectorization_location(train_data, test_data_new)
train_embeddings = use_embedding(train_text)['outputs'].numpy()

test_embeddings = use_embedding(test_text)['outputs'].numpy()
train_emb_loc = [np.concatenate((train_embeddings[i], train_location[i])) for i in range(0, len(train_location))]

test_emb_loc = [np.concatenate((test_embeddings[i], test_location[i])) for i in range(0, len(test_location))]
svc = SVC()

svc.fit(train_emb_loc, train_label)



predict_label = svc.predict(test_emb_loc)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
c_array = np.linspace(1.0, 2.0, num=5)

gamma_array = np.linspace(0.0, 1.0, num=5)
svc = SVC()

grid = GridSearchCV(svc, n_jobs = -1, verbose = 2, param_grid = {'C': c_array, 'gamma': gamma_array})

grid.fit(train_emb_loc, train_label)

pass
best_C = grid.best_estimator_.C

best_gamma = grid.best_estimator_.gamma

print("C =", best_C)

print("Gamma =", best_gamma)
svc = SVC(C = best_C, gamma = best_gamma).fit(train_emb_loc, train_label)



predict_label = svc.predict(test_emb_loc)



print("Accuracy:", accuracy_score(test_label, predict_label))

print("Precision:", precision_score(test_label, predict_label))

print("F1:", f1_score(test_label, predict_label))
for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label

for_submit = for_submit[['id', 'target']]

    

file_name = 'submit_SVC_OHE_location.csv'

for_submit.to_csv('result/' + file_name, index=False)
# Do correction    

for_submit = copy.deepcopy(test_data)

for_submit["target"] = predict_label



id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

for_submit = for_submit[['id', 'target']]



file_name = 'submit_SVC_OHE_location_corrected.csv'

for_submit.to_csv('result/' + file_name, index=False)
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    embedding_layer = embedding(input_text)

    word_embedding = BatchNormalization()(embedding_layer)

    

    input_vector = layers.Input(shape=(4521))

    vector_norm = BatchNormalization()(input_vector)

    

    x = Concatenate()([word_embedding,vector_norm])

    

    x2 = Dense(128, activation='relu')(x)

    x3 = Dense(64, activation='relu')(x2)

    

    predict = Dense(units = 1, activation='sigmoid')(x3)

    

    model = Model(inputs=[input_text, input_vector], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit([train_text, train_location], train_label, validation_data=([test_text, test_location], test_label), 

                    initial_epoch = 0, epochs=7)
def save_result(model, model_name):    

    predict_label = np.asarray(model.predict([test_text, test_location], verbose=1)).round()

    predict_label = predict_label.astype(int)

    

    print("F1-score =", f1_score(test_label, predict_label))

    

    for_submit = copy.deepcopy(test_data)

    for_submit["target"] = predict_label

    for_submit = for_submit[['id', 'target']]

    

    file_name = 'submit_' + model_name + '.csv'

    for_submit.to_csv('result/' + file_name, index=False)

    

    # Do correction    

    for_submit = copy.deepcopy(test_data)

    for_submit["target"] = predict_label

    

    id_to_change = for_submit['id'][for_submit['keyword'].isin(keyword_disaster)].values

    for_submit['target'][for_submit['id'].isin(id_to_change)] = 1

    for_submit = for_submit[['id', 'target']]

    

    print("With correction F1-score =", f1_score(test_label, for_submit['target'].tolist()))

    

    file_name = 'submit_' + model_name + '_corrected.csv'

    for_submit.to_csv('result/' + file_name, index=False)
save_result(model, 'FC_v1_OHE_location')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    embedding_layer = embedding(input_text)

    word_embedding = BatchNormalization()(embedding_layer)

    

    input_vector = layers.Input(shape=(4521))

    vector_norm = BatchNormalization()(input_vector)

    

    x = Concatenate()([word_embedding,vector_norm])

    

    x2 = Dense(512, activation='relu')(x)

    x3 = Dense(256, activation='relu')(x2)

    x4 = Dense(64, activation='relu')(x3)

    

    predict = Dense(units = 1, activation='sigmoid')(x4)

    

    model = Model(inputs=[input_text, input_vector], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit([train_text, train_location], train_label, validation_data=([test_text, test_location], test_label), 

                    initial_epoch = 0, epochs=10)

save_result(model, 'FC_v2_OHE_location')
def build_model(embedding):

    input_text = layers.Input(shape=[], dtype=tf.string)

    embedding_layer = embedding(input_text)

    word_embedding = BatchNormalization()(embedding_layer)

    

    input_vector = layers.Input(shape=(4521))

    vector_norm = BatchNormalization()(input_vector)

    

    x = Concatenate()([word_embedding,vector_norm])

    

    x2 = Dense(1024, activation='relu')(x)

    x3 = Dense(512, activation='relu')(x2)

    x4 = Dense(128, activation='relu')(x3)

    x5 = Dense(64, activation='relu')(x4)

    

    predict = Dense(units = 1, activation='sigmoid')(x5)

    

    model = Model(inputs=[input_text, input_vector], outputs=predict)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1])

  

    return model
model = build_model(embedding)

model.summary()
history = model.fit([train_text, train_location], train_label, validation_data=([test_text, test_location], test_label), 

                    initial_epoch = 0, epochs=10)

save_result(model, 'FC_v3_OHE_location')