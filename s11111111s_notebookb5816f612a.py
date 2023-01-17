import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
#read data
X_train_input=np.load("../input/my-data/X_train.npy")
y_train_input=np.load("../input/my-data/y_train.npy")
X_test_input=np.load("../input/my-data/X_test.npy")
# shape data
X_train_input.shape, y_train_input.shape
# summary signal data
stats_data = list()
for i in range(9):
    data = stats.describe(X_train_input[:,:,i].flatten())._asdict()
    data['min'],data['max']=data.pop('minmax')
    stats_data.append(pd.DataFrame(data, index=[i]))
stats_data = pd.concat(stats_data, axis=0)
stats_data
# is normal y
y_train = y_train_input
sum((y_train_input.max(axis = 1)))==len(y_train_input), sum((y_train_input.min(axis = 1)))==0

# scale input
scaler = MinMaxScaler((-1,1))
n_sample, n_time, n_feature = X_train_input.shape
X_train = X_train_input.reshape(n_sample*n_time, n_feature)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = X_train.reshape(n_sample,n_time, n_feature)
# check that all Ok
np.max(np.abs(scaler.inverse_transform(X_train[0])-X_train_input[0]))

# check stat
stats_data = list()
for i in range(9):
    data = stats.describe(X_train[:,:,i].flatten())._asdict()
    data['min'],data['max']=data.pop('minmax')
    stats_data.append(pd.DataFrame(data, index=[i]))
stats_data = pd.concat(stats_data, axis=0)
stats_data
# split data for test and validation
# X_train,X_test,y_train,y_test = train_test_split(X_train, y_train, random_state=21)
n_epoch = 100
n_batch = 200
k_fold = 4
n_sample_fold = len(X_train_input)//k_fold
# build model
def build_model():
    model = Sequential()
    model.add(LSTM(100,input_shape=(128,9)))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# cross fold
score_acc = list()
score_loss = list()
for i in range(k_fold):
    print('Fold ',i)
    X_fold_validate = X_train[i*n_sample_fold:(i+1)*n_sample_fold]
    y_fold_validate = y_train[i*n_sample_fold:(i+1)*n_sample_fold]
    X_fold_test = np.concatenate([X_train[:i*n_sample_fold],X_train[(i+1)*n_sample_fold:]], axis = 0)
    y_fold_test = np.concatenate([y_train[:i*n_sample_fold],y_train[(i+1)*n_sample_fold:]], axis = 0)
    model = build_model()
    history = model.fit(X_fold_test, y_fold_test,
                        epochs=n_epoch,
                        batch_size=n_batch,
                        verbose=True,
                        validation_data=(X_fold_validate, y_fold_validate))
    score_acc.append(history.history['val_accuracy'][-1])
    score_loss.append(history.history['val_loss'][-1])
    print('loss = {0}, acc={1}'.format(score_loss[-1], score_acc[-1]))
loss, acc = np.mean(score_loss), np.mean(score_acc)
print('\n!!! Model loss_mean = {0}, acc_mean={1}'.format(loss, acc))
# plot for reference last model
acc_temp = history.history['accuracy']
val_acc_temp = history.history['val_accuracy']
epoch =history.epoch
plt.plot(epoch, acc_temp, 'bo',label = 'Training acc')
plt.plot(epoch, val_acc_temp,'b', label = "Validation acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
input_layer = Input((32,32,3))

x = Flatten()(input_layer)

x = Dense(200, activation = 'relu')(x)
x = Dense(150, activation = 'relu')(x)

output_layer = Dense(NUM_CLASSES, activation = 'softmax')(x)

model = Model(input_layer, output_layer)
model.summary()
opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train
          , y_train
          , batch_size=32
          , epochs=10
          , shuffle=True)
model.evaluate(x_test, y_test)
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes) 
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)

