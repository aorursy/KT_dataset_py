import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

df
def load(image_id):



    # 이미지가 저장된 Path를 지정합니다.

    path = '../input/severstal-steel-defect-detection/train_images'



    # Image를 Hard Drive에서 불러 옵니다.

    image = plt.imread(path + '/' + image_id)



    # 지정된 ImageId가 포함된 행에 대한 Mask를 가져옵니다.

    where = df['ImageId'] == image_id



    # 먼저 '0'으로 초기화 된 Label 배열을 만듭니다.

    label = np.zeros(256 * 1600)



    # 지정된 ImageId가 포함된 행에 대해서 행별로 반복합니다.

    for idx, (image_id, class_id, pixels) in df[where].iterrows():



        # EncodedPixel 정보를 불러와서 Numpy Array 형태로 변환합니다.

        pixels = pixels.split()

        pixels = np.array(pixels, dtype='int')

        pixels = pixels.reshape(-1, 2)



        # 모든 시작 번지와 길이에 대해서 Label을 채워 넣습니다.

        for start, length in pixels:

            label[start:(start + length)] = class_id



    # 결함 정보가 기입된 Label을 2차원 형태로 재구성 합니다.

    label = label.reshape(256, 1600, order='F')



    return image, label
image, label = load('ffcf72ecf.jpg')



plt.imshow(image)

plt.show()



plt.imshow(label)

plt.show()
def gen():

    

    # 이미지 파일들의 목록을 중복 없이 가져옵니다.

    image_ids = df['ImageId'].drop_duplicates()

    

    # Generator가 무한히 반복 될 수 있도록 합니다.

    while True:

        

        # 매 Epoch 마다 이미지 파일 목록을 Shuffle 합니다.

        image_ids = image_ids.sample(frac=1.0)

        

        # Batch Size 간격으로 Step을 진행합니다.

        for i in range(0, 6666, 32):

            

            # Image와 Label을 담을 공간을 준비합니다.

            images = []

            labels = []

            

            # Batch Size 간격으로 Image 및 Label을 불러옵니다.

            for image_id in image_ids[i:(i + 32)]:

                

                # 단일 Image와 Label을 불러와서 List에 담아둡니다.

                image, label = load(image_id)

                images.append(image)

                labels.append(label)

                

            # 리스트에 담아 놓은 Image와 Label들을 Numpy Array 형태로 변환합니다.

            images = np.array(images, copy=False)

            labels = np.array(labels, copy=False)

            

            # Batch Image와 Label을 반환 합니다.

            yield images, labels
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(256, 1600, 3)),

    tf.keras.layers.Lambda(lambda x: x / 255),

    tf.keras.layers.Conv2D(32, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.Conv2D(128, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.Conv2D(5, (5, 5), strides=2, padding='same', activation='relu'),

    tf.keras.layers.UpSampling2D(16, interpolation='bilinear'),

    tf.keras.layers.Softmax()

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], loss_weights=[0.0001, 1, 1, 1, 1])

model.summary()
model.fit(gen(), steps_per_epoch=209, epochs=10)
# train data 목록에서 랜덤으로 ImageId를 하나 가져온다.

image_id = df['ImageId'].sample().values[0]



# 가져온 ImageId로 철판 이미지와 결함 이미지를 불러온다.

image, label = load(image_id)



# 학습된 모델로 부터 결함 이미지를 예측 한다.

pred = model(image[np.newaxis, ...])

pred = np.argmax(pred[0], axis=2)



# 철판 이미지를 출력한다.

plt.imshow(image)

plt.show()



# 결함 이미지를 출력한다.

plt.imshow(label)

plt.show()



# 예측 이미지를 출력한다.

plt.imshow(pred)

plt.show()
# train data 목록에서 랜덤으로 ImageId를 하나 가져온다.

image_id = df['ImageId'].sample().values[0]



# 가져온 ImageId로 철판 이미지와 결함 이미지를 불러온다.

image, label = load(image_id)



# 학습된 모델로 부터 결함 이미지를 예측 한다.

pred = model(image[np.newaxis, ...])

pred = np.argmax(pred[0], axis=2)



# 철판 이미지를 출력한다.

plt.imshow(image)

plt.show()



# 결함 이미지를 출력한다.

plt.imshow(label)

plt.show()



# 예측 이미지를 출력한다.

plt.imshow(pred)

plt.show()
# train data 목록에서 랜덤으로 ImageId를 하나 가져온다.

image_id = df['ImageId'].sample().values[0]



# 가져온 ImageId로 철판 이미지와 결함 이미지를 불러온다.

image, label = load(image_id)



# 학습된 모델로 부터 결함 이미지를 예측 한다.

pred = model(image[np.newaxis, ...])

pred = np.argmax(pred[0], axis=2)



# 철판 이미지를 출력한다.

plt.imshow(image)

plt.show()



# 결함 이미지를 출력한다.

plt.imshow(label)

plt.show()



# 예측 이미지를 출력한다.

plt.imshow(pred)

plt.show()