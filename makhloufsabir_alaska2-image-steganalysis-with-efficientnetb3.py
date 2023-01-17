!pip install -q efficientnet
!pip install transformers
import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.layers as l
from keras.regularizers import l2
from keras.optimizers import Adam
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets


from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

print(tpu.master())
print(tpu_strategy.num_replicas_in_sync)
# For tensorflow dataset
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

# Pass
gcs_path = KaggleDatasets().get_gcs_path('alaska2-image-steganalysis')
sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
BATCH_SIZE = 8 * tpu_strategy.num_replicas_in_sync # batch size in tpu
EPOCHS = 1

#Variables

dir_name = ['Test', 'JUNIWARD', 'JMiPOD', 'Cover', 'UERD']
df = pd.DataFrame({})
lists = []
cate = []

#get filenames
for dir_ in dir_name:
    # file name
    list_ = os.listdir("/kaggle/input/alaska2-image-steganalysis/"+dir_+"/")
    lists = lists+list_
    # category name
    cate_ = np.tile(dir_,len(list_))
    cate = np.concatenate([cate,cate_])
    
#to dataframe
df["cate"] = cate
df["name"] = lists
#add path to df
df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]

#Labeling func
def cate_label(x):
    if x["cate"] == "Cover":
        res = 0
    else:
        res = 1
    return res

#Training & test sets
Test_df = df.query("cate=='Test'").sort_values(by="name")
Train_df = df.query("cate!='Test'")

#apply Labeling func
Train_df["labled"] = df.apply(cate_label, axis=1)
print("Training set: \n",Train_df["cate"].value_counts())

print('\n', Train_df["path"].head())
print('\n',Train_df["labled"].head())
print('\n',Test_df["path"].head())
#Validating set


X = Train_df["path"]
y = Train_df["labled"]
z = Test_df["path"]

#Train & test split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=10)

#convert to numpy array
X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

#test set
X_test = np.array(Test_df["path"])
print(X_test[7])
print(X_train[7])
print(X_val[7])
print(y_train[7])
print(y_val[7])
def decode_image(filename, label=None, image_size=(512,512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
with tpu_strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB3(
            input_shape=(512, 512, 3),
            weights='imagenet',
            include_top=False
        ),
        #l.Dense(32, activation="relu",kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
        #l.Dropout(0.4),
       # l.BatchNormalization(),
        l.GlobalAveragePooling2D(),
        #l.BatchNormalization(),
        #l.Activation('relu'),
        l.Dropout(0.1),
       # l.Dense(1),
        l.Dense(1, activation='sigmoid')
    ])
    opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)    
    model.compile(
        optimizer=opt,
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )
model.summary()
STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE
#callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)] 

history = model.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=valid_dataset)
model.save("Mymodel.h5") 
#plt.clf()
#history_dict = history.history
#loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
#epochs = range(1, (len(history_dict['loss']) + 1))
#plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()
#plt.clf()
#acc_values = history_dict['accuracy']
#val_acc_values = history_dict['val_accuracy']
#epochs = range(1, (len(history_dict['accuracy']) + 1))
#plt.plot(epochs, acc_values, 'bo', label='Training acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.show()
pred = model.predict(test_dataset, verbose=1)
my_sample = sample.copy()
my_sample["Label"] = pred
my_sample.to_csv("my_sample.csv", index=False)
my_sample.head()