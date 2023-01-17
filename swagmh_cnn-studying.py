import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
%matplotlib inline
def chk_processting_time(start_time, end_time):
    process_time = end_time - start_time
    p_time = int(process_time)
    p_min = p_time // 60
    p_sec = p_time %  60
    print('처리시간 : {p_min}분 {p_sec}초 경과되었습니다.'.format(
            p_min = p_min, 
            p_sec = p_sec
        ))
    return process_time
train = pd.read_csv('../input/training/training.csv')
train.info()
train.dropna(inplace=True)
train.info()
train.shape
train.index = pd.RangeIndex(len(train.index))
train.tail(3)
test = pd.read_csv('../input/test/test.csv')
test.info()
test.shape
len(train), len(test)
len(train.Image[0].split(' ')), len(test.Image[0].split(' '))
train.Image[0].split(' ')[0], test.Image[0].split(' ')[0]
for x in range(len(train)):
    train.Image[x] = np.asarray(train.Image[x].split(' '), dtype=np.uint8).reshape(96, 96)
for x in range(len(test)):
    test.Image[x] = np.asarray(test.Image[x].split(' '), dtype=np.uint8).reshape(96, 96)
train.shape, test.shape
y = train.iloc[:, :-1].values
y.shape
y[1,:]
plt.imshow(train.Image[0])
plt.show()
def keypoints_show(x, y=None):
    plt.imshow(x, 'gray')
    if y is not None:
        points = np.vstack(np.split(y, 15)).T
        plt.plot(points[0], points[1], 'o', color='red')
    plt.axis('off')   
sample_idx = np.random.choice(len(train))
y[sample_idx]
train.Image[sample_idx]
keypoints_show(train.Image[sample_idx], y[sample_idx])
x = np.stack(train.Image)[..., None]
x.shape
x_t = np.stack(test.Image)[..., None]
x_t.shape
x = x / 255.0
x_t = x_t / 255.0
from IPython.display import SVG
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten, LeakyReLU, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import model_to_dot
# model = Sequential()

# model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'linear', input_shape = (96, 96, 1)))
# model.add(LeakyReLU(alpha=.001))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.7))
# model.add(Flatten())
# model.add(Dense(256, activation = 'linear'))
# model.add(LeakyReLU(alpha=.001))
# model.add(Dropout(0.7))
# model.add(Dense(128, activation = 'linear'))
# model.add(LeakyReLU(alpha=.001))
# model.add(Dropout(0.7))
# model.add(Dense(30))
# model2 = Sequential()

# model2.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'Same', activation = 'linear', input_shape = (96, 96, 1)))
# model2.add(LeakyReLU(alpha=.001))
# model2.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model2.add(Dropout(0.5))
# model2.add(Flatten())
# model2.add(Dense(256, activation = 'linear'))
# model2.add(LeakyReLU(alpha=.001))
# model2.add(Dropout(0.7))
# model2.add(Dense(30))
# model3 = Sequential()

# model3.add(Conv2D(filters = 128, kernel_size = (5,5), padding = 'Same', activation = 'linear', input_shape = (96, 96, 1)))
# model3.add(LeakyReLU(alpha=.001))
# model3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model3.add(Dropout(0.5))
# model3.add(Flatten())
# model3.add(Dense(256, activation = 'linear'))
# model3.add(LeakyReLU(alpha=.001))
# model3.add(Dropout(0.5))
# model3.add(Dense(128, activation = 'linear'))
# model3.add(LeakyReLU(alpha=.001))
# model3.add(Dropout(0.7))
# model3.add(Dense(30))
# np.random.seed(777)

# model4 = Sequential()

# model4.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'elu', input_shape = (96, 96, 1)))
# model4.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model4.add(Dropout(0.3))

# model4.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'elu'))
# model4.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model4.add(Dropout(0.5))

# model4.add(Flatten())
# model4.add(Dense(128, activation = 'relu'))
# model4.add(Dropout(0.5))
# model4.add(Dense(30, activation = 'linear'))
# np.random.seed(777)

# model5 = Sequential()

# model5.add(Conv2D(filters = 32, kernel_size = (4,4), padding = 'Same', activation = 'relu', input_shape = (96, 96, 1)))
# model5.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'Same', activation = 'relu'))
# model5.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model5.add(Dropout(0.25))
# model5.add(Flatten())
# model5.add(Dense(256, activation = 'relu'))
# model5.add(Dropout(0.5))
# model5.add(Dense(128, activation = 'relu'))
# model5.add(Dropout(0.7))
# model5.add(Dense(30))
# np.random.seed(777)

# model6 = Sequential()

# model6.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'elu', input_shape = (96, 96, 1)))
# model6.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model6.add(Dropout(0.3))

# model6.add(Conv2D(filters = 32, kernel_size = (4,4), padding = 'Same', activation = 'elu'))
# model6.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model6.add(Dropout(0.5))

# model6.add(Flatten())
# model6.add(Dense(256, activation = 'elu'))
# model6.add(Dropout(0.5))
# model6.add(Dense(128, activation = 'relu'))
# model6.add(Dropout(0.7))
# model6.add(Dense(30))
# np.random.seed(777)

# model7 = Sequential()

# model7.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (96, 96, 1)))
# model7.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model7.add(Dropout(0.5))

# model7.add(Conv2D(filters = 32, kernel_size = (4,4), padding = 'Same', activation = 'relu'))
# model7.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model7.add(Dropout(0.5))
# model7.add(Flatten())

# model7.add(Dense(128, activation = 'relu'))
# model7.add(Dropout(0.7))
# model7.add(Dense(30))
# np.random.seed(777)

# model8 = Sequential()

# model8.add(Conv2D(filters = 64, kernel_size = (6,6), padding = 'Same', activation = 'relu', input_shape = (96, 96, 1)))
# model8.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model8.add(Dropout(0.3))

# model8.add(Conv2D(filters = 32, kernel_size = (4,4), padding = 'Same', activation = 'relu'))
# model8.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model8.add(Dropout(0.5))
# model8.add(Flatten())

# model8.add(Dense(256, activation = 'relu'))
# model8.add(Dropout(0.5))
# model8.add(Dense(128, activation = 'relu'))
# model8.add(Dense(30))
# np.random.seed(777)

# model9 = Sequential()

# model9.add(Conv2D(filters = 128, kernel_size = (5,5), padding = 'Same', activation = 'linear', input_shape = (96, 96, 1)))
# model9.add(LeakyReLU(alpha=.001))
# model9.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model9.add(Dropout(0.5))
# model9.add(Flatten())
# model9.add(Dense(256, activation = 'linear'))
# model9.add(LeakyReLU(alpha=.001))
# model9.add(Dropout(0.5))
# model9.add(Dense(128, activation = 'linear'))
# model9.add(LeakyReLU(alpha=.001))
# model9.add(Dropout(0.7))
# model9.add(Dense(30))
np.random.seed(777)

model10 = Sequential()

model10.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (96, 96, 1)))
model10.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model10.add(Dropout(0.3))

model10.add(Conv2D(filters = 32, kernel_size = (4,4), padding = 'Same', activation = 'relu'))
model10.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model10.add(Dropout(0.5))

model10.add(Flatten())
model10.add(Dense(128, activation = 'relu'))
model10.add(Dropout(0.7))
model10.add(Dense(30, activation = 'relu'))
# model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'accuracy'])
# model2.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'accuracy'])
# model3.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'accuracy'])
# model4.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'accuracy'])
# model5.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
# model6.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
# model7.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
# model8.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
# model9.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
model10.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
MODEL_DIR = '../model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = '../model/{epoch:02d}-{val_loss:4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
# time1 = time.time()
# model.fit(x, y, epochs=100, batch_size=128, validation_split=0.2)
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# model2.fit(x, y, epochs=100, batch_size=100, validation_split=0.3)
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# model3.fit(x, y, epochs=100, batch_size=100, validation_split=0.2)
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# model4.fit(x, y, epochs=100, batch_size=100, validation_split=0.25)
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# history = model5.fit(x, y, validation_split=0.3, epochs=100, batch_size=100, verbose=1, callbacks=[early_stopping_callback, checkpointer])
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# history = model6.fit(x, y, validation_split=0.3, epochs=100, batch_size=100, verbose=1, callbacks=[early_stopping_callback, checkpointer])
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# history = model7.fit(x, y, validation_split=0.3, epochs=100, batch_size=100, verbose=1, callbacks=[early_stopping_callback, checkpointer])
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# history = model8.fit(x, y, validation_split=0.3, epochs=100, batch_size=100, verbose=1, callbacks=[early_stopping_callback, checkpointer])
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
# time1 = time.time()
# history = model9.fit(x, y, validation_split=0.2, epochs=100, batch_size=100, verbose=1, callbacks=[early_stopping_callback, checkpointer])
# time2 = time.time()
# print('Learning Finished!')
# chk_processting_time(time1, time2)
time1 = time.time()
history = model10.fit(x, y, validation_split=0.2, epochs=100, batch_size=100, verbose=1, callbacks=[early_stopping_callback, checkpointer])
time2 = time.time()
print('Learning Finished!')
chk_processting_time(time1, time2)
predict = model10.predict(x)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

x_len = np.arange(len(train_loss))
plt.plot(x_len, train_loss, marker='.', c='red', label='Train_loss')
plt.plot(x_len, val_loss, marker='.', c='blue', label='Val_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
def result_show(x, y, predict):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    
    for ax in axes:
        ax.imshow(x, 'gray')
        ax.axis('off')
        
    points = np.vstack(np.split(y, 15)).T
    p_points = np.vstack(np.split(predict, 15)).T
    
    axes[0].plot(p_points[0], p_points[1], 'o', color='red')
    axes[0].set_title('Predict_Keypoints', size=15)
    
    axes[1].plot(p_points[0], p_points[1], 'o', color='red')
    axes[1].plot(points[0], points[1], 'o', color='blue')
    axes[1].set_title('Result', size=15)
sample_idx = np.random.choice(len(train))
result_show(train.Image[sample_idx], y[sample_idx], predict[sample_idx])
x_t.shape
y_t = model10.predict(x_t)
sample_idx = np.random.choice(len(test))
keypoints_show(test.Image[sample_idx], y_t[sample_idx])
look_id = pd.read_csv('../input/IdLookupTable.csv')
look_id.info()
look_id.drop('Location', axis=1, inplace=True)
look_id.info()
ind = np.array(train.columns[:-1])
value = np.array(range(0,30))
maps = pd.Series(value, ind)
look_id['location_id'] = look_id.FeatureName.map(maps)
df = look_id.copy()

location = pd.DataFrame({'Location':[]})
for i in range(1,1784):
    ind = df[df.ImageId==i].location_id
    location = location.append(pd.DataFrame(y_t[i-1][list(ind)],columns=['Location']), ignore_index=True)
look_id['Location']=location
look_id[['RowId','Location']].to_csv('Predict.csv',index=False)
# !pip install kaggle
# !pip show kaggle
# !kaggle config path
# ! kaggle competitions submit -c facial-keypoints-detection -f predict.csv -m'submission
# !kaggle competitions submissions -c facial-keypoints-detection
