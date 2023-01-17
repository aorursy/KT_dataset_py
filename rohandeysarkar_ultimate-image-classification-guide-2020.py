import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from keras.applications import MobileNet, MobileNetV2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Dense, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
path = '../input/images/dataset/'
os.listdir(path)
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')

train_df.head()
plt.figure(figsize=(10, 10))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = mpimg.imread(path + '/Train Images/' + train_df["Image"][i])
    img = cv2.resize(img, (224, 224))
    plt.imshow(img)
    plt.title(train_df["Class"][i])
    plt.axis("off")
train_df['Class'].unique()
class_map = {
    'Food': 0,
    'Attire': 1,
    'Decorationandsignage': 2,
    'misc': 3
}

inverse_class_map = {
    0: 'Food',
    1: 'Attire',
    2: 'Decorationandsignage',
    3: 'misc'
}
sns.countplot(train_df['Class'])
train_df["Class"].value_counts()
balance_attire = (2278 - 1691)
balance_decoration = (2278 - 743) 
balance_misc = (2278 - 1271) 
balance_food = 1000
recover_balance = { 'Image': [], 'Class': [] }

while balance_food != 0:
    for i in range(train_df.shape[0]):
        if balance_food == 0:
                break
        if train_df.iloc[i]["Class"] == 'Food':
            recover_balance["Image"].append(train_df.iloc[i]["Image"])
            recover_balance["Class"].append(train_df.iloc[i]["Class"])
            balance_food -= 1
            
while balance_attire != 0:
    for i in range(train_df.shape[0]):
        if balance_attire == 0:
                break
        if train_df.iloc[i]["Class"] == 'Attire':
            recover_balance["Image"].append(train_df.iloc[i]["Image"])
            recover_balance["Class"].append(train_df.iloc[i]["Class"])
            balance_attire -= 1
            
while balance_decoration != 0:
    for i in range(train_df.shape[0]):
        if balance_decoration == 0:
                break
        if train_df.iloc[i]["Class"] == 'Decorationandsignage':
            recover_balance["Image"].append(train_df.iloc[i]["Image"])
            recover_balance["Class"].append(train_df.iloc[i]["Class"])
            balance_decoration -= 1
            
while balance_misc != 0:
    for i in range(train_df.shape[0]):
        if balance_misc == 0:
                break
        if train_df.iloc[i]["Class"] == 'misc':
            recover_balance["Image"].append(train_df.iloc[i]["Image"])
            recover_balance["Class"].append(train_df.iloc[i]["Class"])
            balance_misc -= 1
balance_df = pd.DataFrame(recover_balance)
balance_df = balance_df.sample(frac = 1)
balance_df.head()
temp_df = pd.concat([balance_df, train_df])

sns.countplot(temp_df['Class'])
train_df['Class'] = train_df['Class'].map(class_map).astype(np.uint8)
train_df.head()
balance_df['Class'] = balance_df['Class'].map(class_map).astype(np.uint8)
balance_df.head()
h, w = 224, 224
batch_size = 64
epochs = 100
train_path = path + '/Train Images/'
test_path = path + '/Test Images/'
train_images, train_labels = [], []

for i in range(train_df.shape[0]):
    train_image = cv2.imread(train_path + str(train_df.Image[i]))
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
    train_image = cv2.resize(train_image, (h, w))
    train_images.append(train_image)
    train_labels.append(train_df.Class[i])

# After adding the train_df data add balance_df inorder to balance the total training data
for i in range(balance_df.shape[0]):
    train_image = cv2.imread(train_path + str(balance_df.Image[i]))
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
    train_image = cv2.resize(train_image, (h, w))
    train_images.append(train_image)
    train_labels.append(balance_df.Class[i])

test_images = []

for i in range(test_df.shape[0]):
    test_image = cv2.imread(test_path + str(test_df.Image[i]))
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (h, w))
    test_images.append(test_image)

train_images = np.array(train_images)
test_images = np.array(test_images)
X_train, X_test, y_train, y_test = train_test_split(train_images, to_categorical(train_labels), test_size=0.3, random_state=42)
base_model = MobileNet(
    input_shape=(h, w, 3), 
    weights='imagenet',
    include_top=False, 
    pooling='avg'
)

base_model.summary()
base_model.trainable = False

output_class = 4

model = Sequential([
  base_model,
  Dense(output_class, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
earlystop = EarlyStopping(monitor='val_loss', patience=5)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), validation_data = (X_test, y_test),
                    steps_per_epoch = len(X_train) / batch_size, epochs = epochs, callbacks = callbacks)
labels = model.predict(test_images)
print(labels[:4])
label = [np.argmax(i) for i in labels]
print(label[:4])
class_label = [inverse_class_map[x] for x in label]
print(class_label[:4])
submission = pd.DataFrame({ 'Image': test_df.Image, 'Class': class_label })
submission.head()
plt.figure(figsize=(10, 10))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = mpimg.imread(path + '/Test Images/' + submission["Image"][i])
    img = cv2.resize(img, (224, 224))
    plt.imshow(img)
    plt.title(submission["Class"][i])
    plt.axis("off")
# submission.to_csv('sub.csv', index=False)