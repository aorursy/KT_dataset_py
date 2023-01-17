import numpy as np

import pandas as pd

import os

from glob import glob

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')



df.head()
labels_count = df['Finding Labels'].value_counts()[:15]

plt.subplots(figsize=(12, 8))

ax = sns.barplot(labels_count, labels_count.index)

ax.set(xlabel='Amount of occurrences')
df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

y = []

for x in df['Finding Labels'].unique():

    splitted = x.split('|')

    y = np.append(y, splitted)



y = [x for x in y if len(x) > 0]

y = np.unique(y)

print('All labels ({}): {}'.format(len(y), y))
for label in y:

    df[label] = df['Finding Labels'].map(lambda x: 1.0 if label in x else 0.0)



df.head()
plt.figure(figsize=(16, 18))

for i, column in enumerate(df[y]):

    plt.subplot(3, 5, i+1)

    plt.title(column)

    sns.distplot(df[df[column].where(df[column] == 1.0).notnull()]['Patient Age'])    
import matplotlib.image as mpimg



img = mpimg.imread('/kaggle/input/data/images_001/images/00000001_000.png')

imgplot = plt.imshow(img, cmap='bone')

plt.show()
def get_corresponding_label(image_index):

    return y[df.loc[df['Image Index'] == image_index][y].values.argmax()]



get_corresponding_label('00000001_000.png')
def show_images(path='/kaggle/input/data', extension='/images_001/images', labeled=False, max_images=6):

    amount = 0

    fig = plt.figure(figsize=(12, 8))

    

    for file in os.listdir(path + extension):

        if file.endswith('.png'):

            if amount == max_images:

                break

            

            img = mpimg.imread(os.path.join(path + extension, file))

            plt.subplot(231+amount)

            if labeled:

                plt.title(get_corresponding_label(file))

            imgplot = plt.imshow(img, cmap='bone')

            

            amount += 1



show_images(labeled=True)
#df['disease_vec'] = df.apply(lambda x: [x[y].values], 1).map(lambda x: x[0])

#df['disease_vec'].head()
df.shape
df = df[df['Patient Age'] < 100]

df_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('/kaggle/input/data', 'images*', '*', '*.png'))}

print('Scans found:', len(df_image_paths), ', Total Headers', df.shape[0])

df['path'] = df['Image Index'].map(df_image_paths.get)

df['Patient Age'] = df['Patient Age'].map(lambda x: int(x))
new_df = df.sample(30000)
import cv2

def read_img(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (128, 128))

    return img



from tqdm import tqdm

train_img = []

for img_path in tqdm(new_df['path'].values):

    train_img.append(read_img(img_path))
X = np.array(train_img, np.float32)/255
Y = new_df[y].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(

    X,

    Y,

    test_size=0.25,

    random_state=2018

)

print('train', X_train.shape[0], 'val', X_test.shape[0])
del X

del Y
X_train.shape
Y_train.shape
from keras.applications.mobilenet import MobileNet

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten

from keras.models import Sequential



base_mobilenet_model = MobileNet(

    input_shape=(128, 128, 3), 

    include_top=False,

    weights=None

)

model = Sequential()

model.add(base_mobilenet_model)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(512))

model.add(Dropout(0.5))

model.add(Dense(len(y), activation='sigmoid'))



model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['binary_accuracy', 'mae']

)



model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



weight_path="{}_weights.best.hdf5".format('xray_class')



checkpoint = ModelCheckpoint(

    weight_path,

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=True

)



early = EarlyStopping(

    monitor="val_loss", 

    mode="min", 

    patience=3

)



callbacks_list = [checkpoint, early]
early_stops = EarlyStopping(patience=3, monitor='val_acc')



model.fit(

    x=X_train,

    y=Y_train,

    batch_size=100,

    epochs=5,

    validation_split=0.3,

    callbacks=[early_stops]

)
for c_label, s_count in zip(y, 100*np.mean(Y_test, 0)):

    print('%s: %2.2f%%' % (c_label, s_count))
pred_Y = model.predict(X_test, batch_size=32, verbose=True)
from sklearn.metrics import roc_curve, auc



fig, c_ax = plt.subplots(1, 1, figsize = (9, 9))



for (idx, c_label) in enumerate(y):

    fpr, tpr, thresholds = roc_curve(Y_test[:, idx].astype(int), pred_Y[:, idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('barely_trained_net.png')
early_stops = EarlyStopping(patience=3, monitor='val_acc')



model.fit(x=X_train, y=Y_train, batch_size=100, epochs=8, validation_split=0.3, callbacks=[early_stops])
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1, 1, figsize = (9, 9))

for (idx, c_label) in enumerate(y):

    fpr, tpr, thresholds = roc_curve(Y_test[:, idx].astype(int), pred_Y[:, idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('barely_trained_net.png')
model_json = model.to_json()

with open("multi_disease_model.json", "w") as json_file:

    json_file.write(model_json)



# serialize weights to HDF5

model.save_weights("multi_disease_model_weight.h5")

print("Saved model to disk")