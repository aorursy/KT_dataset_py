#importing libaries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style("darkgrid")

sms = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin1')

sms = sms.iloc[:,[0,1]]

sms.columns = ["label", "message"]

sms.head()

sms.shape
count_Class=pd.value_counts(sms["label"], sort= True)

count_Class.plot(kind = 'bar',color = ["green","red"])

plt.title('Bar Plot')

plt.show();
#747 spam messages  are there

sms.groupby('label').describe()
#lets add length coloumn to the data

sms['length'] = sms['message'].apply(len)

sms.head()


fig = plt.figure(dpi = 120)

ax = plt.axes()

sms['length'].plot(bins=50, kind='hist',ax=ax,color = 'indigo')

ax.set(xlabel = 'Message Length Class',ylabel = 'Frequency',title = 'Length Distribution');

# comparison of spam and ham messages   

plt.figure(figsize=(12, 8))



sms[sms.label=='ham'].length.plot(bins=35, kind='hist', color='green', 

                                       label='Ham messages', alpha=0.6)

sms[sms.label=='spam'].length.plot(kind='hist', color='red', 

                                       label='Spam messages', alpha=0.6)

plt.legend()

plt.xlabel("Message Length")
from collections import Counter



count1 = Counter(" ".join(sms[sms['label']=='ham']["message"]).split()).most_common(20)

df1 = pd.DataFrame.from_dict(count1)

df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})

count2 = Counter(" ".join(sms[sms['label']=='spam']["message"]).split()).most_common(20)

df2 = pd.DataFrame.from_dict(count2)

df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})
df1.plot.bar(legend = False,color="black")

y_pos = np.arange(len(df1["words in non-spam"]))

plt.xticks(y_pos, df1["words in non-spam"])

plt.title('More frequent words in non-spam messages')

plt.xlabel('words')

plt.ylabel('number')

plt.show()





df2.plot.bar(legend = False, color = 'blue')

y_pos = np.arange(len(df2["words in spam"]))

plt.xticks(y_pos, df2["words in spam"])

plt.title('More frequent words in spam messages')

plt.xlabel('words')

plt.ylabel('number')

plt.show()

#randomaly checking a message

sms[sms.length == 200].message.iloc[0]

#importing libaries required

from keras.preprocessing.text import Tokenizer

from keras.layers import Embedding, LSTM, Dropout, Dense

from keras.models import Sequential

from keras.utils import to_categorical

vocab_size = 400

oov_tok = "<OOV>"

max_length = 250

embedding_dim = 16

encode = ({'ham': 0, 'spam': 1} )

#new dataset with replaced values

sms = sms.replace(encode)
sms.head()


X = sms['message']

Y = sms['label']



tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(X)

# convert to sequence of integers

X = tokenizer.texts_to_sequences(X)
X = np.array(X)

y = np.array(Y)

from keras.preprocessing.sequence import pad_sequences



X = pad_sequences(X, maxlen=max_length)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=7)



import tensorflow as tf



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs = 30

history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test,y_test), verbose=2)
result = model.evaluate(X_test, y_test)

# extract those

loss = result[0]

accuracy = result[1]





print(f"[+] Accuracy: {accuracy*100:.2f}%")

from keras.preprocessing import sequence

def get_predictions(txts):

    txts = tokenizer.texts_to_sequences(txts)

    txts = sequence.pad_sequences(txts, maxlen=max_length)

    preds = model.predict(txts)

    if(preds[0] > 0.5):

        print("SPAM MESSAGE")

        

    else:

        print('NOT SPAM')



    
# Spam message

txts=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"]



get_predictions(txts)
#not Spam

txts = ["Hi man, I was wondering if we can meet tomorrow."]

get_predictions(txts)