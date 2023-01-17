import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.preprocessing import image as image_utils

from keras import applications

import cv2

import gc

from tqdm import tqdm



from sklearn.metrics import fbeta_score
# Params

input_size = 128

input_channels = 3



epochs = 40

batch_size = 128

learning_rate = 4.7e-5

lr_decay = .0029



valid_data_size = 5000  # Samples to withhold for validation



model = Sequential()

model.add( BatchNormalization( 

                  input_shape=(input_size, input_size, input_channels) ) )



vggmod = applications.VGG19(include_top=False, input_shape=(input_size, input_size, input_channels))

model.add( Sequential(layers=vggmod.layers) )

model.add(Dropout(0.2))



# top_model = Sequential()

model.add(Flatten(input_shape=model.output_shape[1:]))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(17, activation='sigmoid'))



# model.add(top_model)
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):

  def mf(x):

    p2 = np.zeros_like(p)

    for i in range(17):

      p2[:, i] = (p[:, i] > x[i]).astype(np.int)

    score = fbeta_score(y, p2, beta=2, average='samples')

    return score



  x = [0.2]*17

  for i in range(17):

    best_i2 = 0

    best_score = 0

    for i2 in range(resolution):

      i2 /= resolution

      x[i] = i2

      score = mf(x)

      if score > best_score:

        best_i2 = i2

        best_score = score

    x[i] = best_i2

    if verbose:

      print(i, best_i2, best_score)



  return x
df_train_data = pd.read_csv(r"d:\bigdata\amazon\train_v2.csv")
flatten = lambda l: [item for sublist in l for item in sublist]

labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))



label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}
x_valid = []

y_valid = []



df_valid = df_train_data[:valid_data_size]



for f, tags in tqdm(df_valid.values, miniters=100):

    img = cv2.resize(cv2.imread(r"d:\bigdata\amazon\train-jpg\{}.jpg".format(f)), (input_size, input_size))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1

    x_valid.append(img)

    y_valid.append(targets)



y_valid = np.array(y_valid, np.uint8)

x_valid = np.array(x_valid, np.float32)



gc.collect()

x_train = []

y_train = []



df_train = df_train_data[valid_data_size:]



for f, tags in tqdm(df_train.values, miniters=1000):

    img = cv2.resize(cv2.imread(r"d:\bigdata\amazon\train-jpg\{}.jpg".format(f)), (input_size, input_size))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1

    x_train.append(img)

    y_train.append(targets)

    img = cv2.flip(img, 0)  # flip vertically

    x_train.append(img)

    y_train.append(targets)

    img = cv2.flip(img, 1)  # flip horizontally

    x_train.append(img)

    y_train.append(targets)

    img = cv2.flip(img, 0)  # flip vertically

    x_train.append(img)

    y_train.append(targets)



y_train = np.array(y_train, np.uint8)

x_train = np.array(x_train, np.float32)



gc.collect()
df_test_data = pd.read_csv(r"d:\bigdata\amazon\sample_submission_v2.csv")

x_test = []

x_test1 = []

x_test2 = []

x_test3 = []



for f, tags in tqdm(df_test_data.values, miniters=1000):

    img = cv2.resize(cv2.imread(r"d:\bigdata\amazon\test-jpg\{}.jpg".format(f)), (input_size, input_size))

    x_test.append(img)

    img = cv2.flip(img, 0)  # flip vertically

    x_test1.append(img)

    img = cv2.flip(img, 1)  # flip horizontally

    x_test2.append(img)

    img = cv2.flip(img, 0)  # flip vertically

    x_test3.append(img)



    

x_test = np.array(x_test, np.float32)

x_test1 = np.array(x_test1, np.float32)

x_test2 = np.array(x_test2, np.float32)

x_test3 = np.array(x_test3, np.float32)



gc.collect()

callbacks = [EarlyStopping(monitor='val_loss',

                           patience=3,

                           verbose=0),

             TensorBoard(log_dir='logs'),

             ModelCheckpoint('weights.h5',

                             save_best_only=True)]



opt = Adam(lr=learning_rate, decay=lr_decay)



model.compile(loss='binary_crossentropy',

              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.

              optimizer=opt,

              metrics=['accuracy'])

model.fit(x_train,

          y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=2,

          callbacks=callbacks,

          validation_data=(x_valid, y_valid))

p_valid = model.predict(x_valid, batch_size=batch_size)

print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
optimise_f2_thresholds(y_valid, p_valid)
labels
y_test = []



p_test0 = model.predict(x_test, batch_size=batch_size, verbose=2)

p_test1 = model.predict(x_test1, batch_size=batch_size, verbose=2)

p_test2 = model.predict(x_test2, batch_size=batch_size, verbose=2)

p_test3 = model.predict(x_test3, batch_size=batch_size, verbose=2)

p_test = .31*p_test0 + .23*p_test1 + .23*p_test2 + .23*p_test3



y_test.append(p_test)

result = np.array(y_test[0])

result = pd.DataFrame(result, columns=labels)



preds = []



for i in tqdm(range(result.shape[0]), miniters=1000):

    a = result.iloc[[i]]

    a = a.apply(lambda x: x > 0.2, axis=1)

    a = a.transpose()

    a = a.loc[a[i] == True]

    ' '.join(list(a.index))

    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds

df_test_data.to_csv('vgg19sub.csv', index=False)

result.index = df_test_data.iloc[:,0].values

result.to_csv('vgg19prob.csv', index=False)
p_valid.shape