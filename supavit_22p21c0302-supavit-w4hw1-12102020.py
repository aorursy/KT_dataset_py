import pandas as pd

import numpy as np



from matplotlib import pyplot as plt



import cv2

import os
TRAIN_IMAGE_PATH = '/kaggle/input/thai-mnist-classification/train'

TRAIN_LABEL_PATH = '/kaggle/input/thai-mnist-classification/mnist.train.map.csv'
df = pd.read_csv(TRAIN_LABEL_PATH)

df
def get_image(df, index):

    image = cv2.imread(os.path.join(TRAIN_IMAGE_PATH, df.iloc[index]['id']), cv2.IMREAD_GRAYSCALE)

    image = cv2.bitwise_not(image)

    return image
from skimage.morphology import convex_hull_image



temp_img = get_image(df, 1234)

convex_img = convex_hull_image(temp_img)



fig, [ax1,ax2] = plt.subplots(1, 2)

ax1.imshow(temp_img)

ax2.imshow(convex_img)
def crop(image, pad=32):

    convex_img = convex_hull_image(image)

    x, y = np.where(convex_img)

    width = max(x) - min(x)

    height = max(y) - min(y)

    squre_width = max(width, height) + (pad*2)



    crop_img = np.zeros((squre_width, squre_width))

    crop_img[(squre_width-width)//2:(squre_width-width)//2+width, (squre_width-height)//2:(squre_width-height)//2+height] = image[min(x):max(x), min(y):max(y)]

    return crop_img
temp_img = get_image(df, 64)

convex_img = convex_hull_image(temp_img)

crop_img = crop(temp_img)



fig, [ax1,ax2,ax3] = plt.subplots(1, 3)

ax1.imshow(temp_img)

ax2.imshow(convex_img)

ax3.imshow(crop_img)
def step_resize(image, size=64, threshold=50):

    image_size = np.array(image.shape)

    step = image_size[0] // size

    

    image = ((image > threshold)*255).astype(np.uint8)

    image_size = image_size - (image_size % size)

    image = cv2.resize(image,tuple(image_size))

    #image = ((image > threshold)*255).astype(np.uint8)

    

    for i in range(1, step):

        image = ((image > threshold)*255).astype(np.uint8)

        image_size = image_size - size

        image = cv2.resize(image,tuple(image_size))

        #image = ((image > threshold)*255).astype(np.uint8)

        

    return image
temp_img = crop(get_image(df, 1234))

resize_img = cv2.resize(temp_img, (64,64))

step_resize_img = step_resize(temp_img, size=64, threshold=50)



fig, [ax1,ax2, ax3] = plt.subplots(1, 3)

ax1.imshow(temp_img)

ax2.imshow(resize_img)

ax3.imshow(step_resize_img)
def plot_image(df, n=8, figsize=(32,32)):

    fig, ax = plt.subplots(10, n, figsize=figsize)

    for num in range(10):

        selected_num = df[df['category'] == num]

        for col in range(0, n, 4):

            index = np.random.randint(0, selected_num.shape[0])

            image = get_image(selected_num, index)

            ax[num, col].imshow(image)

            ax[num, col].set_title("Raw")

            ax[num, col].set_axis_off()

            

            image = crop(image, pad=32)

            ax[num, col+1].imshow(image)

            ax[num, col+1].set_title("Crop")

            ax[num, col+1].set_axis_off()

            

            ax[num, col+2].imshow(cv2.resize(image, (32,32)))

            ax[num, col+2].set_title("cv2 resize")

            ax[num, col+2].set_axis_off()

            

            image = step_resize(image, size=32, threshold=60)

            ax[num, col+3].imshow(image)

            ax[num, col+3].set_title("Step resize")

            ax[num, col+3].set_axis_off()

            

plot_image(df)
def preprocess_image(df):

    X = np.zeros((df.shape[0], 32, 32))

    for index in range(df.shape[0]):

        image = get_image(df, index)

        image = crop(image, pad=32)

        image = step_resize(image, size=32, threshold=60)

        X[index] = image

        

        if index%500 == 0:

            print(f"Preprocess image: {index} / {df.shape[0]}")

            

    return X.reshape(-1,32,32,1)
X = preprocess_image(df)
def plot_image(df, n=10, figsize=(32,32)):

    fig, ax = plt.subplots(10, n, figsize=figsize)

    for num in range(10):

        selected_num = df[df['category'] == num]

        for col in range(n):

            index = np.random.randint(0, selected_num.shape[0])

            image = X[df['category'] == num][index,:,:,0]

            ax[num, col].imshow(image)

            ax[num, col].set_title(f"{num}")

            ax[num, col].set_axis_off()

            

plot_image(df)
import tensorflow as tf

from sklearn.model_selection import train_test_split
Y = tf.keras.utils.to_categorical(df['category'].values, num_classes=10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
import tensorflow as tf

from tensorflow import keras
model_1 = tf.keras.Sequential(name="Lenet5")

model_1.add(tf.keras.layers.Conv2D(6, (5,5), input_shape=(32, 32, 1), activation='relu'))

model_1.add(tf.keras.layers.MaxPool2D()) 

model_1.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 

model_1.add(tf.keras.layers.MaxPool2D()) 

model_1.add(tf.keras.layers.Flatten()) 

model_1.add(tf.keras.layers.Dense(120, activation='relu'))

model_1.add(tf.keras.layers.Dense(84, activation='relu'))

model_1.add(tf.keras.layers.Dense(10, activation='softmax'))

model_1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])



model_1.summary()
reduce_rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',

                                                 factor=0.1,

                                                 patience=5,

                                                 min_lr=0.0000001)



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
history = model_1.fit(X_train, Y_train, batch_size=64,validation_data=(X_test,Y_test), epochs=100,

                      callbacks=[reduce_rl, early_stop])
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
def plot_image(n=10, figsize=(32,32)):

    fig, ax = plt.subplots(10, n, figsize=figsize)

    

    predict = model_1.predict(X_test)

    for row in range(10):

        for col in range(n):

            index = np.random.randint(0, X_test.shape[0])

            image = X_test[index,:,:,0]

            ax[row, col].imshow(image)

            ax[row, col].set_title(f"{predict[index].argmax()}")

            ax[row, col].set_axis_off()

            

plot_image()
def lookup_feature(f1, f2, f3):

    if pd.isna(f1):

        return (f2+f3) % 99

    if f1 == 0:

        return (f2*f3) % 99

    if f1 == 1:

        return np.abs(f2-f3) % 99

    if f1 == 2:

        return ((f2+f3)*np.abs(f2-f3)) % 99

    if f1 == 3:

        return (np.abs((f3*(f3+1)) - (f2*(f2-1))) / 2) % 99

    if f1 == 4:

        return (50+(f2-f3)) % 99

    if f1 == 5:

        return min(f2,f3) % 99

    if f1 == 6:

        return max(f2,f3) % 99

    if f1 == 7:

        return (((f2*f3) % 9) * 11) % 99

    if f1 == 8:

        return (((f2**2)+1)*f2 + f3*(f3+1)) % 99

    if f1 == 9:

        return (50+f2) % 99
TEST_IMAGE_PATH = '/kaggle/input/thai-mnist-classification/test'

TEST_LABEL_PATH = '/kaggle/input/thai-mnist-classification/test.rules.csv'

SUBMIT_LABEL_PATH = '/kaggle/input/thai-mnist-classification/submit.csv'
test_df = pd.read_csv(TEST_LABEL_PATH)

submit_df = pd.read_csv(SUBMIT_LABEL_PATH)
for index in range(test_df.shape[0]):

    

    f1 = test_df.iloc[index]['feature1']

    if not pd.isna(f1):

        f1_img = cv2.imread(os.path.join(TEST_IMAGE_PATH, f1), cv2.IMREAD_GRAYSCALE)

        f1_img = cv2.bitwise_not(f1_img)

        f1_img = crop(f1_img, pad=32)

        f1_img = step_resize(f1_img, size=32, threshold=60)

        f1_img = np.reshape(f1_img, (1,32,32,1))

        f1 = model_1.predict(f1_img).argmax()

        

    f2 = test_df.iloc[index]['feature2']

    f2_img = cv2.imread(os.path.join(TEST_IMAGE_PATH, f2), cv2.IMREAD_GRAYSCALE)

    f2_img = cv2.bitwise_not(f2_img)

    f2_img = crop(f2_img, pad=32)

    f2_img = step_resize(f2_img, size=32, threshold=60)

    f2_img = np.reshape(f2_img, (1,32,32,1))

    f2 = model_1.predict(f2_img).argmax()

    

    f3 = test_df.iloc[index]['feature3']

    f3_img = cv2.imread(os.path.join(TEST_IMAGE_PATH, f3), cv2.IMREAD_GRAYSCALE)

    f3_img = cv2.bitwise_not(f3_img)

    f3_img = crop(f3_img, pad=32)

    f3_img = step_resize(f3_img, size=32, threshold=60)

    f3_img = np.reshape(f3_img, (1,32,32,1))

    f3 = model_1.predict(f3_img).argmax()

    

    predict = lookup_feature(f1, f2, f3)

    submit_df.iat[index, 1] = predict

    

    print(f"{index}: {f1}, {f2}, {f3}, {predict}")
submit_df.to_csv('./submit.csv', index=False)