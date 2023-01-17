import os

from keras.applications.xception import Xception

from keras.models import Model

from keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras import optimizers

from keras.callbacks import ModelCheckpoint



from keras.utils import np_utils



from keras.models import load_model



import cv2



from keras import backend as K



import matplotlib.pyplot as plt

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



import gc
img = np.loadtxt("../input/aptos2019-blindness-detection/train.csv",       # 読み込みたいファイルのパス

                  delimiter=",",    # ファイルの区切り文字

                  skiprows=1,    # 先頭の何行を無視するか（指定した行数までは読み込まない）

                  usecols=(0), # 読み込みたい列番号

                  dtype = "str"

                 )

img
label = np.loadtxt("../input/aptos2019-blindness-detection/train.csv",       # 読み込みたいファイルのパス

                  delimiter=",",    # ファイルの区切り文字

                  skiprows=1,    # 先頭の何行を無視するか（指定した行数までは読み込まない）

                  usecols=(1), # 読み込みたい列番号

                  dtype = "int"

                 )

label
img_label_trains = []

img_label_validations = []



for i in range(4):

    data_train, data_test, labels_train, labels_test = train_test_split(img, label, train_size=0.85,random_state=i*5,stratify=label)

    

    img_label_train = np.stack([data_train, labels_train],axis=1)

    img_label_validation = np.stack([data_test, labels_test],axis=1)

    

    img_label_trains.append(img_label_train)

    img_label_validations.append(img_label_validation)
#confirm dataset count(train)

print(np.count_nonzero(img_label_trains[0][:,1] == "0"))

print(np.count_nonzero(img_label_trains[0][:,1] == "1"))

print(np.count_nonzero(img_label_trains[0][:,1] == "2"))

print(np.count_nonzero(img_label_trains[0][:,1] == "3"))

print(np.count_nonzero(img_label_trains[0][:,1] == "4"))

#confirm dataset count

print(np.count_nonzero(img_label_validations[0][:,1] == "0"))

print(np.count_nonzero(img_label_validations[0][:,1] == "1"))

print(np.count_nonzero(img_label_validations[0][:,1] == "2"))

print(np.count_nonzero(img_label_validations[0][:,1] == "3"))

print(np.count_nonzero(img_label_validations[0][:,1] == "4"))
len(img_label_trains[0])
len(img_label_validations[0])
def vertical_flip(image, rate=0.5):

    if np.random.rand() < rate:

        image = image[::-1, :, :]

    return image



def horizontal_flip(image):

    image = image[:, ::-1, :]

    return image



def image_translation(img):

    params = np.random.randint(-50, 51)

    if not isinstance(params, list):

        params = [params, params]

    rows, cols, ch = img.shape



    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])

    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst



def image_shear(img):

    params = np.random.randint(-20, 21)*0.01

    rows, cols, ch = img.shape

    factor = params*(-1.0)

    M = np.float32([[1, factor, 0], [0, 1, 0]])

    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst



def image_rotation(img):

    params = np.random.randint(-30, 31)

    rows, cols, ch = img.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), params, 1)

    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst



def image_contrast(img):

    params = np.random.randint(7, 10)*0.1

    alpha = params

    new_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha

    #new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

  

    return new_img



def image_brightness2(img):

    params = np.random.randint(-21, 22)

    beta = params

    b, g, r = cv2.split(img)

    b = cv2.add(b, beta)

    g = cv2.add(g, beta)

    r = cv2.add(r, beta)

    new_img = cv2.merge((b, g, r))

    return new_img



def pca_color_augmentation_modify(image_array_input):

    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3

    assert image_array_input.dtype == np.uint8



    img = image_array_input.reshape(-1, 3).astype(np.float32)

    # 分散を計算

    ch_var = np.var(img, axis=0)

    # 分散の合計が3になるようにスケーリング

    scaling_factor = np.sqrt(3.0 / sum(ch_var))

    # 平均で引いてスケーリング

    img = (img - np.mean(img, axis=0)) * scaling_factor



    cov = np.cov(img, rowvar=False)

    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)



    rand = np.random.randn(3) * 0.1

    delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)

    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]



    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)

    return img_out
def get_random_data(image_lines_1, abs_path, img_width, img_height, data_aug):

    image_file = abs_path + image_lines_1[0] + ".png"

    label = np.eye(5)[int(image_lines_1[1])]

    

    seed_image = cv2.imread(image_file)

    seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2RGB)

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    

    if data_aug:

        

        r = np.random.rand()

        

        if r >= 0.5:

    

            seed_image = vertical_flip(seed_image)

            seed_image = horizontal_flip(seed_image)

            seed_image = image_shear(seed_image)

            seed_image = image_rotation(seed_image)

            seed_image = pca_color_augmentation_modify(seed_image)

    

    seed_image = seed_image / 255

    

    return seed_image, label
def data_generator(image_lines, batch_size, abs_path, img_width, img_height, data_aug):

    '''data generator for fit_generator'''

    n = len(image_lines)

    i = 0

    while True:

        image_data = []

        label_data = []

        for b in range(batch_size):

            if i==0:

                np.random.shuffle(image_lines)

            image, label = get_random_data(image_lines[i], abs_path, img_width, img_height, data_aug)

            image_data.append(image)

            label_data.append(label)

            i = (i+1) % n

        image_data = np.array(image_data)

        label_data = np.array(label_data)

        yield image_data, label_data



def data_generator_wrapper(image_lines, batch_size, abs_path, img_width, img_height, data_aug):

    n = len(image_lines)

    if n==0 or batch_size<=0: return None

    return data_generator(image_lines, batch_size, abs_path, img_width, img_height, data_aug)
img_width, img_height = 449, 449

num_train = len(img_label_trains[0])

num_val = len(img_label_validations[0])

batch_size = 4

print(num_train, num_val)

abs_path = "../input/aptos2019-blindness-detection/train_images/"
models = []



for i in range(4):



    input_tensor = Input(shape=(img_height, img_width, 3))



    xception_model = Xception(include_top=False, weights=None, input_tensor=input_tensor)



    xception_model.load_weights("../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")



    x = xception_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.3)(x)

    outputs = Dense(5, activation='softmax')(x)



    model = Model(inputs=xception_model.input, outputs=outputs)

    

    model.compile(optimizer=optimizers.SGD(lr=0.001,momentum=0.9),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



    model.summary()

    

    models.append(model)
models[0].fit_generator(data_generator_wrapper(img_label_trains[0], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[0], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[0].fit_generator(data_generator_wrapper(img_label_trains[0], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[0], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[0].fit_generator(data_generator_wrapper(img_label_trains[0], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[0], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[0].fit_generator(data_generator_wrapper(img_label_trains[0], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[0], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[1].fit_generator(data_generator_wrapper(img_label_trains[1], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[1], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[1].fit_generator(data_generator_wrapper(img_label_trains[1], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[1], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[1].fit_generator(data_generator_wrapper(img_label_trains[1], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[1], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[1].fit_generator(data_generator_wrapper(img_label_trains[1], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[1], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[2].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[2].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[2].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[2].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[3].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[3].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[3].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
models[3].fit_generator(data_generator_wrapper(img_label_trains[2], batch_size, abs_path, img_width, img_height, True),

        steps_per_epoch=max(1, num_train//batch_size),

        validation_data=data_generator_wrapper(img_label_validations[2], batch_size, abs_path, img_width, img_height, True),

        validation_steps=max(1, num_val//batch_size),

        epochs=5,

        initial_epoch=0,

        class_weight=[1,4.8,1.8,9.1,6.3])
img_test = np.loadtxt("../input/aptos2019-blindness-detection/test.csv",       # 読み込みたいファイルのパス

                  delimiter=",",    # ファイルの区切り文字

                  skiprows=1,    # 先頭の何行を無視するか（指定した行数までは読み込まない）

#                  usecols=(1), # 読み込みたい列番号

                  dtype = "str"

                 )
test_abs_path = "../input/aptos2019-blindness-detection/test_images/"



data = []

for i in range(len(img_test)):

    image_file = test_abs_path + img_test[i] + ".png"

    seed_image = cv2.imread(image_file)

    seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2RGB)

    seed_image = cv2.resize(seed_image, dsize=(img_width, img_height))

    seed_image = np.expand_dims(seed_image, axis=0)

    seed_image = seed_image / 255

    predict1 = models[0].predict(seed_image)

    predict2 = models[1].predict(seed_image)

    predict3 = models[2].predict(seed_image)

    predict4 = models[3].predict(seed_image)

    predict_mean = (predict1+predict2+predict3+predict4)/4

    x = np.array([img_test[i], np.argmax(predict_mean)])

    data.append(x)

    

data = np.array(data)
columns = ['id_code', 'diagnosis']

name = 'sample'



d = pd.DataFrame(data=data, columns=columns, dtype='str')
d.to_csv("submission.csv",index=False)
df = pd.read_csv("submission.csv")

print(df)