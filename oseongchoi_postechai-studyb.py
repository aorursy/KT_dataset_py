import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

df
def load(image_id):

    """

    Steel Defect Detection 수행을 위해서 원본 이미지(철판)와 결함 이미지를 불러옵니다.

    """

    

    # 원본 이미지가 저장된 폴더의 위치를 정의한다.

    path = '../input/severstal-steel-defect-detection/train_images/'

    

    # 원본 이미지를 불러온다.

    image = plt.imread(path + image_id)

    

    # 결함 이미지를 생성한다.

    label = np.zeros(256 * 1600)

    

    where = df['ImageId'] == image_id



    # DataFrame의 각 행별로 반복 실행한다.

    for idx, (image_id, class_id, pixels) in df[where].iterrows():



        pixels = pixels.split()  # 공백단위로 분리

        pixels = np.array(pixels, dtype=int)  # 넘파이 어레이로 변환

        pixels = pixels.reshape(-1, 2)  # n X 2 의 행렬로 변환



        # 결함 정보(시작점 및 길이)에 따라서 결함 부분에 결함 종류를 기입합니다.

        for start, length in pixels:

            label[start:start + length] = class_id



    # 결함 정보를 원본 이미지와 같은 Shape로 치환 합니다.

    label = label.reshape(256, 1600, order='F')

    

    return image, label
image_id = df.sample()['ImageId'].values[0]

image, label = load(image_id)

plt.imshow(image)

plt.show()

plt.imshow(label)

plt.show()
def f1_score(y_true, y_pred, c):

    

    y_pred = tf.argmax(y_pred, axis=-1)



    pred = tf.cast(y_pred == c, dtype=tf.float32)

    true = tf.cast(y_true == c, dtype=tf.float32)



    tp = tf.reduce_sum(pred * true)

    fp = tf.reduce_sum(pred * (1 - true))

    fn = tf.reduce_sum((1 - pred) * true)



    return tp / (tp + 0.5 * (fp + fn))





def f1_0(y_true, y_pred):

    return f1_score(y_true, y_pred, 0)



def f1_1(y_true, y_pred):

    return f1_score(y_true, y_pred, 1)



def f1_2(y_true, y_pred):

    return f1_score(y_true, y_pred, 2)



def f1_3(y_true, y_pred):

    return f1_score(y_true, y_pred, 3)



def f1_4(y_true, y_pred):

    return f1_score(y_true, y_pred, 4)
tf.keras.backend.clear_session()

model = tf.keras.Sequential([

    tf.keras.layers.Input((256, 1600, 3)),

    tf.keras.layers.Conv2D(16, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.UpSampling2D(8),

    tf.keras.layers.Conv2D(5, (5, 5), padding='same', activation='softmax'),

])

metrics = [f1_0, f1_1, f1_2, f1_3, f1_4]

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics, loss_weights=[0.001, 1, 1, 1, 1])

model.summary()
class UNet(tf.keras.Model):

    

    def __init__(self):

        super(UNet, self).__init__()

        

        self.maxpool = tf.keras.layers.MaxPool2D()

        

        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')

        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')

        

        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')

        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')

    

        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        self.conv6 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        

        self.conv7 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')

        self.conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')

        

        self.conv9 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')

        self.conv10 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')

        

        self.upconv1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same', activation='relu')



        self.conv11 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')

        self.conv12 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')

        

        self.upconv2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same', activation='relu')

        

        self.conv13 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        self.conv14 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        

        self.upconv3 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same', activation='relu')

        

        self.conv15 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')

        self.conv16 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')

        

        self.upconv4 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=2, padding='same', activation='relu')

        

        self.conv17 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')

        self.conv18 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')

        

        self.conv19 = tf.keras.layers.Conv2D(5, (1, 1), padding='same', activation='softmax')

        

    def call(self, x, training=False):

        x = x / 255

        

        x = self.conv1(x)

        x = x_2 = self.conv2(x)

        

        x = self.maxpool(x)

        

        x = self.conv3(x)

        x = x_4 = self.conv4(x)

        

        x = self.maxpool(x)

        

        x = self.conv5(x)

        x = x_6 = self.conv6(x)

        

        x = self.maxpool(x)

        

        x = self.conv7(x)

        x = x_8 = self.conv8(x)

        

        x = self.maxpool(x)

        

        x = self.conv9(x)

        x = self.conv10(x)

        

        x = self.upconv1(x)

        

        x = tf.concat([x, x_8], axis=-1)

        

        x = self.conv11(x)

        x = self.conv12(x)

        

        x = self.upconv2(x)

        

        x = tf.concat([x, x_6], axis=-1)

        

        x = self.conv13(x)

        x = self.conv14(x)

        

        x = self.upconv3(x)

        

        x = tf.concat([x, x_4], axis=-1)

        

        x = self.conv15(x)

        x = self.conv16(x)

        

        x = self.upconv4(x)

        

        x = tf.concat([x, x_2], axis=-1)

        

        x = self.conv17(x)

        x = self.conv18(x)

        

        return self.conv19(x)





tf.keras.backend.clear_session()

model = UNet()

metrics = [f1_0, f1_1, f1_2, f1_3, f1_4]

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics, loss_weights=[0.001, 1, 1, 1, 1])

model.build(input_shape=(None, 256, 1600, 3))

model.summary()

model(np.random.randn(10, 256, 1600, 3))
# Train/Valid split



ids = df['ImageId']          # 인덱스만 가져오기

ids = ids.drop_duplicates()  # 중복 제거



ids_train = ids.sample(frac=0.8)

ids_valid = ids.drop(index=ids_train.index)
def gen(ids, batch_size):



    ids = ids.values             # Pandas에서 Numpy로 변경

    

    while True:



        # 이미지 아이디를 셔플 합니다.

        np.random.shuffle(ids)



        for i in range(0, ids.shape[0], batch_size):



            images = []  # 원본 이미지 담을 버퍼

            labels = []  # 결함 이미지 담을 버퍼



            for image_id in ids[i:i + batch_size]:   # 현재 미니 배치의 첫번째 부터 끝까지 반복

                image, label = load(image_id)        # image.shape == (256, 1600, 3) / label.shape == (256, 1600)

                images.append(image)

                labels.append(label)



            images = np.array(images, copy=False)    # shape == (batch_size, 256, 1600, 3)

            labels = np.array(labels, copy=False)    # shape == (batch_size, 256, 1600)



            yield images, labels
model.fit(gen(ids_train, 24), validation_data=gen(ids_valid, 24), steps_per_epoch=100, validation_steps=42, epochs=10)
image_id = ids_valid.sample().values[0]



image, label = load(image_id)



pred = model(image[np.newaxis, ...])



pred = np.argmax(pred, axis=-1)



fig, axs = plt.subplots(3, 1, figsize=[20, 10])

axs[0].imshow(image)

axs[1].imshow(label)

axs[2].imshow(pred[0])

fig.show()