import numpy as np

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn.metrics import precision_score, recall_score , f1_score



import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau 

from tensorflow.keras.models import load_model
# have a look at dataset

df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df.head()
df.info()
# Read from pretrained Embeddings

def read_from_glove(filename):

    word_to_index = {}

    index_to_word = {}

    word_to_vec   = {}

    words         = [] 

    with open(filename,"r") as f:

        for i,line in enumerate(f,1):

            line = line.strip().split()

            word = line[0]

            vector = np.array(line[1:],dtype=np.float32)

            word_to_vec[word] = vector

            word_to_index[word] = i

            index_to_word[i] = word

            words.append(word)

    return word_to_index, index_to_word, word_to_vec, words 



word_to_index, index_to_word, word_to_vec, words = read_from_glove("/kaggle/input/glove6b50dtxt/glove.6B.50d.txt")
len(df) - df["text"].count()

# No null values.
tokenizer = Tokenizer()

tokenizer.fit_on_texts(df["text"])



sequences = tokenizer.texts_to_sequences(df["text"])

print(sequences[0])

print(len(sequences[2]))
# Configrations



MAXLEN         = max([len(e) for e in sequences])

VOCAB_SIZE     = len(word_to_vec)+1

EMBEDDING_DIM  = 50

EPOCHS         = 50

BUFFER_SIZE    = 1000

BATCH_SIZE     = 16
def create_emb_matrix(vocab_len,emb_dim):

    emb_matrix = np.zeros((vocab_len,emb_dim))

    for i,vec in enumerate(word_to_vec.values()):

        emb_vec = vec

        if emb_vec is not None:

            emb_matrix[i] = emb_vec

    return emb_matrix



embedding_matrix = create_emb_matrix(VOCAB_SIZE,EMBEDDING_DIM)
#Padding

padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="pre")



# Extracting Labels

labels=df["target"].values



#Splitting training and Test examples.

x_train,x_test,y_train,y_test = train_test_split(padded_sequences,labels,stratify=labels,random_state=42,test_size=0.3)

x_train.shape
np.bincount(y_train), np.bincount(y_test)
def create_dataset(padded_sequence,labels,batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((padded_sequence,labels))

    dataset = dataset.shuffle(BUFFER_SIZE)

    dataset = dataset.batch(batch_size,drop_remainder=True).prefetch(1)

    return dataset
train_dataset = create_dataset(x_train,y_train,BATCH_SIZE)

test_dataset  = create_dataset(x_test,y_test,BATCH_SIZE)
#Building Model



model = tf.keras.Sequential([

    tf.keras.layers.Embedding(VOCAB_SIZE,

                              EMBEDDING_DIM,

                              weights = [embedding_matrix],

                              trainable=True,

                              input_length=MAXLEN),

    

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16,return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(64,activation="tanh"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64,activation="tanh"),

    tf.keras.layers.Dense(1,activation="sigmoid")

])



model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.RMSprop(lr=0.0001,momentum=0.3),metrics=["accuracy"])
#model.load_weights("/kaggle/working/disaster_v1.h5",by_name=True)

model.summary()
def CB():

        callbacks = []

        checkpoint = ModelCheckpoint("/kaggle/working/disaster_v1.h5",

                                    monitor="val_loss",

                                    save_best_only=True,

                                    mode="min",verbose=1,

                                    )



        reducelr   = ReduceLROnPlateau(monitor="val_loss",

                                      factor=0.3,patience=5,

                                      verbose=1, mode="min", min_lr=1e-09)



        log        = CSVLogger("/kaggle/working/disaster_v1.csv")



        callbacks.append(checkpoint)

        callbacks.append(reducelr)

        callbacks.append(log)

        return callbacks



callbacks = CB()
np.bincount(labels)

# Clear Sign of Data Imbalance

patches, texts, autotexts = plt.pie(np.bincount(labels),explode=[0.1,0.1],labels=list(np.unique(labels)),radius=1.5,shadow=True,autopct='%1.1f%%')

texts[0].set_fontsize(20)

texts[1].set_fontsize(20)

plt.show()
# Fitting the model

model.fit(train_dataset,epochs=EPOCHS,

             validation_data=test_dataset,verbose=1,

             callbacks=[callbacks])
history = pd.read_csv("/kaggle/working/disaster_v1.csv")

plt.plot(history["epoch"],history["loss"],label="Training Loss",c="red")

plt.plot(history["epoch"],history["val_loss"],label="Validation loss",c="green")

plt.legend()

plt.show()
def eval(y_true,y_pred):

    print("Accuracy Score \t:",round(accuracy_score(y_true,y_pred),4))

    print("R-square Coefficient \t:",round(r2_score(y_true,y_pred),4))

    # Model's performance is worse than horizontal line

    print("Precision Score \t:",round(precision_score(y_true,y_pred),4))

    print("Recall Score \t:",round(recall_score(y_true,y_pred),4))

    print("F1 Score \t:",round(f1_score(y_true,y_pred),4))

    

y_pred = model.predict(x_test)

cat_out = np.round(y_pred)

cat_out = cat_out.ravel()

eval(y_test,cat_out)
model.save("/kaggle/working/disaster_v1_model.h5")
def create_submission_data(data):

    sub_data = tf.data.Dataset.from_tensor_slices(data)

    return sub_data
#prepairing the submission_data

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



sequences = tokenizer.texts_to_sequences(test_df["text"])

padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="pre")



sub_data = create_submission_data(padded_sequences)
sub_predictions = model.predict(sub_data)

sub_predictions = sub_predictions.ravel()
cat_sub_out = np.round(sub_predictions)

#cat_sub_out = cat_sub_out.ravel()

cat_sub_out = cat_sub_out.astype("int64")
#Submission

def predict(id,data,filename="submission.csv"):

    with open(filename,"w") as file:

        file.write("id,target\n")

        for idx,e in zip(id,data):

            file.write(f"{idx},{e}\n")

predict(test_df["id"],cat_sub_out)
sub = pd.read_csv("/kaggle/working/submission.csv")
sub