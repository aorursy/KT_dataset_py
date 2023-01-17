import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/heartbeat/mitbih_train.csv", header=None)
test_df = pd.read_csv("/kaggle/input/heartbeat/mitbih_test.csv", header=None)

print(train_df.shape)
print(test_df.shape)
train_df.head()
test_df.head()
print(train_df[train_df.columns[-1]].unique())
print(train_df[test_df.columns[-1]].unique())
train_x = np.array(train_df[train_df.columns[0:-1]], dtype=np.float32)
train_y = np.array(train_df[train_df.columns[-1:]], dtype=np.float32)

test_x = np.array(train_df[test_df.columns[0:-1]], dtype=np.float32)
test_y = np.array(train_df[test_df.columns[-1:]], dtype=np.float32)

print("print train set is : x = {} y = {}".format(train_x.shape, train_y.shape))
print("print test set is : x = {} y = {}".format(test_x.shape, test_y.shape))
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)
ax.plot(train_x[0], color="r")
ax.plot(train_x[1], color="g")
ax.plot(train_x[2], color="b")
plt.show()
# Return difference array
def return_diff_array_table(array, dur):
  for idx in range(array.shape[1]-dur):
    before_col = array[:,idx]
    after_col = array[:,idx+dur]
    new_col = ((after_col - before_col)+1)/2
    new_col = new_col.reshape(-1,1)
    if idx == 0:
      new_table = new_col
    else :
      new_table = np.concatenate((new_table, new_col), axis=1)
#For concat add zero padding
  padding_array = np.zeros(shape=(array.shape[0],dur))
  new_table = np.concatenate((padding_array, new_table), axis=1)
  return new_table
#Concat
def return_merge_diff_table(df, diff_dur):
  fin_table = df.reshape(-1,187,1,1)
  for dur in diff_dur:
    temp_table = return_diff_array_table(df, dur)
    fin_table = np.concatenate((fin_table, temp_table.reshape(-1,187,1,1)), axis=2)
  return fin_table

#Use "stratify" option
x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)

#Add Data
x_train = return_merge_diff_table(df=x_train, diff_dur=[1])
x_val = return_merge_diff_table(df=x_val, diff_dur=[1])

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
#For see a model's result
def return_result(model, x_train, x_test, y_train, y_test):
    y_pred = model.predict(x_test)
    train_pred = model.predict(x_train)
    pred_list=[]
    for x in y_pred:
        pred_list.append(np.argmax(x))
    train_pred_list=[]
    for x in train_pred:
        train_pred_list.append(np.argmax(x))
    test_mat = confusion_matrix(y_test, pred_list)
    train_mat = confusion_matrix(y_train, train_pred_list)
    print("In train")
    print(accuracy_score(y_train, train_pred_list))
    print(train_mat)
    print("In test")
    print(accuracy_score(y_test, pred_list))
    print(test_mat)
def return_model1():
    input_tens = tf.keras.Input(shape=(187,2,1))
    x = tf.keras.layers.Conv2D(256, kernel_size=(10,2), strides=(5,1),padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model
model1 = return_model1()
#For saving best model
checkpoint_path_best = "./best_acc_v01.ckpt"
cp_callback_best = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_best,monitor="val_accuracy",save_weights_only=True,verbose=1,save_best_only=True)

model1.fit(x_train,y_train, epochs=200, batch_size=128, validation_data=(x_val,y_val),callbacks=[cp_callback_best])
# Result is ========
return_result(model1, x_train=x_train, x_test=x_val, y_train=y_train, y_test=y_val)
def return_model2():
    input_tens = tf.keras.Input(shape=(187,2,1))
    x = tf.keras.layers.Conv2D(256, kernel_size=(10,2), strides=(5,1),padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model
model2 = return_model2()
#For saving best model
checkpoint_path_best2 = "./best_acc_v02.ckpt"
cp_callback_best2 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_best2, monitor="val_accuracy", save_weights_only=True, verbose=1, save_best_only=True)

model2.fit(x_train,y_train, epochs=200, batch_size=128, validation_data=(x_val,y_val), callbacks=[cp_callback_best2])
return_result(model2, x_train=x_train, x_test=x_val, y_train=y_train, y_test=y_val)
model1.load_weights(checkpoint_path_best)
model2.load_weights(checkpoint_path_best2)

return_result(model1, x_train=x_train, x_test=x_val, y_train=y_train, y_test=y_val)
return_result(model2, x_train=x_train, x_test=x_val, y_train=y_train, y_test=y_val)

test_input = np.array(test_df[test_df.columns[0:-1]], dtype=np.float32)
test_target = np.array(test_df[test_df.columns[-1:]], dtype=np.float32)

test_input = return_merge_diff_table(df=test_input, diff_dur=[1])

print(test_input.shape, test_target.shape)
pred_1 = model1.predict(test_input)
pred_2 = model2.predict(test_input)
pred_tot = (pred_1+pred_2)/2

pred_idx_list=[]
for pred in pred_tot:
    pred_idx_list.append(np.argmax(pred))
    
pred_idx_arr = np.array(pred_idx_list, dtype=np.float32)
print(accuracy_score(test_target, pred_idx_arr))
print(confusion_matrix(test_target, pred_idx_arr))
import seaborn as sns
#From https://www.kaggle.com/agungor2/various-confusion-matrix-plots
def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    col = ['N','S','V','F','Q']
    cm = pd.DataFrame(cm, index=col, columns=col)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    
plot_cm(test_target, pred_idx_arr)