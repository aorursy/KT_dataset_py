import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')
np.random.seed(294056)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

target = train['label']
del train['label']
print('Training data has ' + str(train.shape[0]) + ' rows and ' + str(train.shape[0]) + ' columns')
print('Test data has ' + str(test.shape[0]) + ' rows and ' + str(test.shape[0]) + ' columns')
train.head()
target.value_counts(normalize=True)
p = sns.countplot(target)
train = train / 255.0
test = test / 255.0
train = train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
train.shape # 42000 images with (28 x 28 x 1)
test.shape # 28000 images with (28 x 28 x 1)
target_backup = np.array(target)

target = to_categorical(target, num_classes = 10)
target.shape
target
X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size = 0.2, random_state = 294056)
# # get 1 random image from each label
each_label_index = []
for eachLabel in range(10):
    index = np.array(np.where(target_backup == eachLabel)[0])
    np.random.shuffle(index)
    each_label_index.append(index[0])

plt.subplot(331)
plt.imshow(train[each_label_index[0]][:,:,0], cmap=plt.get_cmap('gray'))
plt.subplot(332)
plt.imshow(train[each_label_index[1]][:,:,0], cmap=plt.get_cmap('gray'))
plt.subplot(333)
plt.imshow(train[each_label_index[2]][:,:,0], cmap=plt.get_cmap('gray'))

plt.subplot(334)
plt.imshow(train[each_label_index[3]][:,:,0], cmap=plt.get_cmap('gray'))
plt.subplot(335)
plt.imshow(train[each_label_index[4]][:,:,0], cmap=plt.get_cmap('gray'))
plt.subplot(336)
plt.imshow(train[each_label_index[5]][:,:,0], cmap=plt.get_cmap('gray'))

plt.subplot(337)
plt.imshow(train[each_label_index[6]][:,:,0], cmap=plt.get_cmap('gray'))
plt.subplot(338)
plt.imshow(train[each_label_index[7]][:,:,0], cmap=plt.get_cmap('gray'))
plt.subplot(339)
plt.imshow(train[each_label_index[8]][:,:,0], cmap=plt.get_cmap('gray'))
plt.imshow(train[each_label_index[9]][:,:,0], cmap=plt.get_cmap('gray'))
for eachLabel in range(9):
    plt.subplot(3, 3, eachLabel + 1)
    index = np.array(np.where(target_backup == eachLabel)[0])
    plt.imshow(train[index].mean(axis = 0)[:,:,0], cmap=plt.get_cmap('gray'))
index = np.array(np.where(target_backup == 9)[0])
plt.imshow(train[index].mean(axis = 0)[:,:,0], cmap=plt.get_cmap('gray'))
for eachLabel in range(9):
    plt.subplot(3, 3, eachLabel + 1)
    index = np.array(np.where(target_backup == eachLabel)[0])
    plt.imshow(train[index].std(axis = 0)[:,:,0], cmap=plt.get_cmap('gray'))
index = np.array(np.where(target_backup == 9)[0])
plt.imshow(train[index].std(axis = 0)[:,:,0], cmap=plt.get_cmap('gray'))
def cnn_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = "Same", activation = 'relu', input_shape = (28, 28, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = "Same", activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'sigmoid'))
    # optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return(model)
model = cnn_model()
model.summary()
DA_fit = ImageDataGenerator(rotation_range = 10, zoom_range = 0.15, width_shift_range = 0.15, height_shift_range = 0.15)
DA_fit.fit(X_train)
which_image = 1
for i, j in DA_fit.flow(X_train, y_train, batch_size=32):
    plt.imshow(i[which_image][:,:,0], cmap = plt.get_cmap('gray'))
    print(j[which_image])
    break;
# # Change the epochs to 50 to get better accuracy
epochs = 1

model_fit = model.fit_generator(DA_fit.flow(X_train,y_train, batch_size=32),
                              epochs = epochs, validation_data = (X_valid, y_valid),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // 32)
# model.fit(X_train, y_train, epochs = 50, validation_data = (X_valid, y_valid), batch_size = 200)
valid_predict = model.predict_classes(X_valid)
valid_true = np.argmax(y_valid, axis = 1)
print(classification_report(valid_true, valid_predict, target_names = ['Class ' + str(i) for i in range(10)]) )
errors = (valid_predict - valid_true != 0)
X_errors = X_valid[errors]
y_errors = valid_true[errors]
y_pred_errors = valid_predict[errors]
print('> O indicates Original\n> P indicates Predicted')
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(X_errors[i][:,:,0], cmap = plt.get_cmap('gray'))
    plt.title('(O, P) : (' + str(y_errors[i]) + ', ' + str(y_pred_errors[i]) + ")")
print('> O indicates Original\n> P indicates Predicted')
ct = 0
for i in range(4, 8):
    ct = ct + 1
    plt.subplot(2, 2, ct)
    plt.imshow(X_errors[i][:,:,0], cmap = plt.get_cmap('gray'))
    plt.title('(O, P) : (' + str(y_errors[i]) + ', ' + str(y_pred_errors[i]) + ")")
print('> O indicates Original\n> P indicates Predicted')
ct = 0
for i in range(8, 12):
    ct = ct + 1
    plt.subplot(2, 2, ct)
    plt.imshow(X_errors[i][:,:,0], cmap = plt.get_cmap('gray'))
    plt.title('(O, P) : (' + str(y_errors[i]) + ', ' + str(y_pred_errors[i]) + ")")
results = model.predict_classes(test)
submission = pd.DataFrame({'ImageId' : np.arange(1, test.shape[0] + 1), 'Label' : results})
submission.to_csv('../submissions/cnn_99_3.csv', index = False)