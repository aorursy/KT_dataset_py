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
import seaborn as sns

import tensorflow as tf
df = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')
df.columns
sns.countplot(df['target'])
"Percentage Real Tweets = {:.1%}".format(len(df[df['target']==1]) / len(df))



from sklearn.model_selection import train_test_split



X = df['text']

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
words = []



for item in df['text']:

    words.append(len(item.split(' ')))

    

df_wordlengths = pd.DataFrame(words)

df_wordlengths.describe()
g = sns.distplot(df_wordlengths,axlabel='Number of words in tweet')
train_labels = np.array(y_train,dtype=float) 

test_labels = np.array(y_test,dtype=float)
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



#Set Hyper Parameters

vocab_size = 10000

embedding_dim = 16

max_length = 27 # Mean of Number of Words in Tweets + (2*Std)

trunc_type = 'post'

padding_type = 'post'

oov_tok = '<OOV>'



tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index



sequences = tokenizer.texts_to_sequences(X_train)

padded = pad_sequences(sequences,max_length,truncating=trunc_type,padding=padding_type)



test_sequences = tokenizer.texts_to_sequences(X_test)

test_padded = pad_sequences(test_sequences,maxlen=max_length,truncating=trunc_type,padding=padding_type)
#Build Model

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(6,activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1,activation='sigmoid')

])



#Compile Model

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
#Fit the model

num_epochs = 10

history = model.fit(x=padded,y=train_labels,epochs=num_epochs,validation_data=(test_padded,test_labels))
#Evaluate Accuracy



test_loss, test_acc = model.evaluate(test_padded,test_labels,verbose=2)



print('\nTest Accuracy:',test_acc)
#Plot the history plot

import matplotlib.pyplot as plt



def plot_graphs(history,string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel("Ephochs")

    plt.ylabel(string)

    plt.legend([string,'validation_'+string])

    plt.show()



plot_graphs(history,"accuracy")

plot_graphs(history,"loss")
predictions = model.predict(test_padded)
def rounding(x):

    '''Round results to 0 (No Disaster) or 1 (Real Disaster)'''

    if x < 0.5:

        return 0

    else:

        return 1

    

predictions_final = [rounding(x) for x in predictions]
sns.countplot(predictions_final)
for x in range(0,5):

    print(f'Tweet: {X_test.iloc[x]},\n Prediction: {predictions[x]},\n  Target:{test_labels[x]}\n')
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(predictions_final,y_test))
submission_sequences = tokenizer.texts_to_sequences(df_test['text'])

submission_padded = pad_sequences(submission_sequences,maxlen=max_length,truncating=trunc_type,padding=padding_type)
submission_predictions = model.predict(submission_padded)
df_test['target'] = [rounding(x) for x in submission_predictions]
df_submission = df_test.drop(['keyword','location','text'],axis=1)
df_submission
df_submission.to_csv('submission.csv',index=False,header=True)