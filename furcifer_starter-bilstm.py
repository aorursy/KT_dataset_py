import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pickle
from collections import Counter
import json
with open('data.json', encoding='utf-8') as fh:
    data = json.load(fh)
data[0]
set_cats = set([a['category'] for a in data])
set_cats
len(data)
all_cats = [a['category'] for a in data]
len(all_cats)
cat_cnts = []

for cat in set_cats:
    cat_cnts.append(all_cats.count(cat))
cat_cnts
sorted(cat_cnts)[::-1]
z = zip(cat_cnts, set_cats)
z = list(z)
z
sel_cats = []

for p in z:
    if p[0] > 8000:
        sel_cats.append(p[1])
sel_cats
X_text = []
y_label = []

for p in data:
    if p['category'] in sel_cats:
        y_label.append(p['category'])
        X_text.append(p['content'])
len(X_text)
len(y_label)
X_text[0]
print(len(X_text[0]))
y_label[0]
set(y_label)
print(len(X_text))
print(len(y_label))
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
class_labels = encoder.fit_transform(y_label)
set(class_labels)
encoder.inverse_transform([[3]])
class_labels.shape
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
class_labels = class_labels.reshape((class_labels.shape[0], 1))
y_ohe = encoder.fit_transform(class_labels)
class_labels[0]
y_ohe
y_ohe.shape
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_text)

X_token = tokenizer.texts_to_sequences(X_text)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
vocab_size
tokenizer.texts_to_sequences(['দ্রব্যমূল্য নিয়ন্ত্রণে অভিযান শুরুর আগে বাজারের নাম ফাঁস হয়ে যাচ্ছে। এই অভিযোগ উঠে এসেছে বাজার তদারকি জোরদার করা নিয়ে আয়োজিত বাণিজ্য মন্ত্রণালয়ের একটি সভায়। সভায় বাণিজ্য মন্ত্রণালয়ের একজন যুগ্ম সচিব বলেন, আগে বাজারে অভিযানের দিন রিজার্ভ পুলিশের সদস্যদের পাওয়া যেত। এখন সংশ্লিষ্ট থানা থেকে সদস্যদের নিয়ে অভিযান পরিচালনা করতে হয়। এতে বাজারের নাম আগেই প্রকাশ হয়ে যায়।  বৈঠকে আগের মতো রিজার্ভ পুলিশ সদস্যদের নিয়ে অভিযান চালানোর সিদ্ধান্ত হয়। এ জন্য পুলিশ সদস্যদের মোতায়েনের জন্য ঢাকা মেট্রোপলিটন পুলিশকে (ডিএমপি) অনুরোধ জানানো হয়। সংশ্লিষ্ট ব্যক্তিরা বলছেন, অভিযানের আগে বাজারের নাম ফাঁস হয়ে গেলে তদারকিতে কোনো লাভ হয় না। ব্যবসায়ীরা আগে থেকেই সতর্ক হয়ে যান। অনেক সময় দেখা যায়, ভ্রাম্যমাণ আদালত যাওয়ার পরই বেশির ভাগ দোকান বন্ধ করে দেওয়া হয়েছে।  সভায় বাজার তদারকির ক্ষেত্রে নানা দুর্বলতা উঠে আসে। সভায় এখন থেকে বাজারে অভিযানের ক্ষেত্রে সংশ্লিষ্ট ব্যবসায়ী সমিতিকে আরও বেশি সম্পৃক্ত করার সিদ্ধান্ত নেওয়া হয়। সভার কার্যবিবরণী থেকে এসব তথ্য জানা গেছে।  বাজার তদারকি দলের কার্যক্রম জোরদার করার লক্ষ্য নিয়ে বাণিজ্য মন্ত্রণালয়ে সভাটি অনুষ্ঠিত হয় গত ২৩ জানুয়ারি। এতে সভাপতিত্ব করেন বাণিজ্যসচিব শুভাশীষ বসু। সভায় বাণিজ্য মন্ত্রণালয়, শিল্প মন্ত্রণালয়, কৃষি মন্ত্রণালয়, খাদ্য মন্ত্রণালয়, ট্যারিফ কমিশন, রপ্তানি উন্নয়ন ব্যুরো (ইপিবি) , ট্রেডিং করপোরেশন অব বাংলাদেশ (টিসিবি) , বাংলাদেশ শিল্প ও বণিক সমিতি ফেডারেশন (এফবিসিসিআই) , ঢাকা উত্তর সিটি করপোরেশন ও জাতীয় ভোক্তা অধিকার সংরক্ষণ অধিদপ্তরের প্রতিনিধিরা উপস্থিত ছিলেন।  বাণিজ্য মন্ত্রণালয় পবিত্র রমজান মাসসহ বিভিন্ন সময়ে বাজারে তদারকি দল পাঠিয়ে থাকে। এসব তদারকি দল নানা অভিযোগে ব্যবসায়ীদের জরিমানা করে।'])
print(X_text[2])
print(X_token[2])
len(X_token[0])
len(X_text[0])
from keras.preprocessing.sequence import pad_sequences
maxlen = 300
X_pad = pad_sequences(X_token, padding='post', maxlen=maxlen)
from collections import Counter

word_ls = []

for sen in X_text:
    word_ls.extend(sen.split())
len(word_ls)
Counter = Counter(word_ls)
most_occur = Counter.most_common(100)
print(most_occur)
X_pad.shape
vocab_size
maxlen
y_ohe.shape
X_pad[1]
class_labels.shape
class_labels[:,0]
c_l = list(class_labels[:,0])
c_l = set(c_l)
class_weight = {}

for c in (list(c_l)):
    print(c)
    c_w = len(class_labels)/np.sum(class_labels==c)
    print(c_w)
    class_weight[c] = c_w
from keras.models import Sequential
from keras.layers import Embedding, CuDNNLSTM, Bidirectional, Dense

embedding_dim = 8

model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
model.add(Bidirectional(CuDNNLSTM(128))) 
model.add(Dense(9, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
from keras.callbacks import LearningRateScheduler, EarlyStopping
from math import exp
def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * exp(-k*epoch)
    return lrate
lrate = LearningRateScheduler(exp_decay)
[exp_decay(i) for i in range(5)]
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
sss.get_n_splits(X_pad, y_ohe)

#print(sss)       

for train_index, test_index in sss.split(X_pad, y_ohe):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_pad[train_index], X_pad[test_index]
    y_train, y_test = y_ohe[train_index], y_ohe[test_index]
class_labels
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=1,
                    validation_split=0.2,
                    batch_size=256,
                    class_weight = class_weight)
model.evaluate(X_test, y_test)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'valid'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'])
plt.show()
model.save('lstm_best.h5')
import matplotlib.pyplot as plt
plt.hist(y_train)
plt.show()
from keras.models import load_model

model = load_model('lstm_best.h5')
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
class_labels = encoder.fit_transform(y_label)
def generate_response():
    input_sentence = input('Enter input news: ')
    Xi_token = tokenizer.texts_to_sequences([input_sentence])
    Xi_pad = pad_sequences(Xi_token, padding='post', maxlen=maxlen)
    print('Model predicts')
    preds = model.predict(Xi_pad)
    print('Confidence :')
    print(preds)
    preds = preds
    total = 0
    for k in range(len(preds[0])):
        print(encoder.inverse_transform([[k]]))
        print('%f %%' %(preds[0,k]*100))
        total += preds[0,k]*100
    #print(total)
    print('Predicted class: %s'%(encoder.inverse_transform(model.predict_classes(Xi_pad))))
generate_response()
