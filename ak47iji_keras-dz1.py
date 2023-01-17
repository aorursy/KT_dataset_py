!cp -r ../input/keras-dz/1/* ./
from keras.layers import Dense

from keras.models import Model

from keras.metrics import top_k_categorical_accuracy

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image

from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions

from keras.callbacks import ModelCheckpoint

from keras.utils import to_categorical

import os

import random

import numpy as np

import PIL



class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper':3, 'plastic':4, 'trash':5}

data_dir = './dataset-resized/dataset-resized/'

output_path = 'list.txt'

fd = open(output_path, 'w')

for class_name in class_names_to_ids.keys():

    images_list = os.listdir(data_dir + class_name)

    for image_name in images_list:

        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))

fd.close()



# 随机选取样本做训练集和测试集

_NUM_VALIDATION = 50

_RANDOM_SEED = 0

list_path = 'list.txt'

train_list_path = 'list_train.txt'

val_list_path = 'list_val.txt'

fd = open(list_path)

lines = fd.readlines()

fd.close()

random.seed(_RANDOM_SEED)

random.shuffle(lines)

fd = open(train_list_path, 'w')

for line in lines[_NUM_VALIDATION:]:

    fd.write(line)

fd.close()

fd = open(val_list_path, 'w')

for line in lines[:_NUM_VALIDATION]:

    fd.write(line)

fd.close()



def get_train_test_data(list_file):

    list_train = open(list_file)

    x_train = []

    y_train = []

    for line in list_train.readlines():

        x_train.append(line.strip()[:-2])

        y_train.append(int(line.strip()[-1]))

        #print(line.strip())

    return x_train, y_train

x_train, y_train = get_train_test_data('list_train.txt')

x_test, y_test = get_train_test_data('list_val.txt')



def process_train_test_data(x_path, y_train, batch_size=8):

    while 1:

        images = []

        Y = []

        cnt = 0

        for image_path, y in zip(x_path, y_train):

            img_load = image.load_img(data_dir+image_path)

            img = image.img_to_array(img_load)

            ############################

            # 数据增强

            # 随机放大

            img = image.random_zoom(num_05, (0.5,0.5))

            # 随机错切

            img = image.random_shear(img, 60)

            # 随机旋转

            img = image.random_rotation(img, 180)

            # 随机平移

            img = image.random_shift(img, 0.2, 0.2)

            ############################

            img = preprocess_input(img)

            cnt += 1

            images.append(img)

            Y.append(y)

            if batch_size == cnt:

                yield np.array(images), to_categorical(Y, 6)

                cnt = 0

                images = []

                Y = []

                

train_images = process_train_test_data(x_train, y_train)

test_images = process_train_test_data(x_test, y_test)



from keras.applications.inception_resnet_v2 import InceptionResNetV2

base_model = InceptionResNetV2(include_top=False, pooling='avg')

outputs = Dense(6, activation='softmax')(base_model.output)

model = Model(base_model.inputs, outputs)



# 设置ModelCheckpoint，按照验证集的准确率进行保存

save_dir = 'train_model'

filepath = "model_{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_accuracy', verbose=1,

                             save_best_only=True , save_weights_only=False)





# 模型设置

def acc_top3(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)





def acc_top5(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', acc_top3, acc_top5])
EPOCHS = 200

# 模型训练

hist = model.fit_generator(train_images, steps_per_epoch=100, epochs=EPOCHS, shuffle=True)

model.save("fit_model.h5")



print("run ok!")
import matplotlib.pyplot as plt

epochs = range(1, EPOCHS+1)

plt.figure()

# loss

plt.plot(epochs, hist.history['loss'], 'r', label='loss')

# accuracy

plt.plot(epochs, hist.history['accuracy'], 'g', label='accuracy')

# acc_top3

plt.plot(epochs, hist.history['acc_top3'], 'b', label='acc_top3')

# acc_top5

plt.plot(epochs, hist.history['acc_top5'], 'k', label='acc_top5')

plt.grid(True)

plt.xlabel('epoch')

plt.ylabel('acc-loss')

plt.legend(loc="upper right")

plt.show()
img = next(train_images)[0][0]
plt.imshow(img)

plt.show()
y = model.predict(np.expand_dims(img, axis=0))
y
lables = {}

for key, value in class_names_to_ids.items():

    lables[value] = key
lables[np.argmax(y)]