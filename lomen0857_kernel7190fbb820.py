!pip install -U efficientnet
import os
import glob
import pickle
# import shap
import math

import pandas as pd
import numpy as np
import seaborn as sns

from numpy import argmax
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical

# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.xception import preprocess_input

from keras_applications.resnext import ResNeXt50
from keras_applications.resnext import preprocess_input

# from efficientnet.tfkeras import EfficientNetB1
# from efficientnet.tfkeras import preprocess_input

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
plt.rc('font', **font)
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})
tf.__version__
arg_dict = {"backend": tf.keras.backend,
            "layers": tf.keras.layers,
            "models": tf.keras.models,
            "utils":tf.keras.utils}
  
# import torch
# import torch.nn as nn
# class FReLU(nn.Module):
#     def __init__(self, c1, k=3):  # ch_in, kernel
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
#         self.bn = nn.BatchNorm2d(c1)

#     def forward(self, x):
# #         return torch.max(x, self.bn(self.conv(x)))
#         return torch.max(x, self.conv(x))
    
# x = np.array(range(32)).reshape(1,4,4,2).astype("float32")
# x = torch.from_numpy(x).clone()
# print(x)
# FReLU(4)(x)
def expand2square(pil_img, background_color = (0,0,0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def mask_circle_solid(arr_img, background_color = (0,0,0), blur_radius = -4):
    pil_img = Image.fromarray(np.uint8(arr_img))
    background = Image.new(pil_img.mode, pil_img.size, background_color)
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((blur_radius, blur_radius, pil_img.size[0] - blur_radius, pil_img.size[1] - blur_radius), fill=255)
    masked_image = np.array(Image.composite(pil_img, background, mask))
    return masked_image

def image_rotate(arr_img, rotate_degree_range):
    rotate_degree = int(np.random.random() * rotate_degree_range)
    pil_img = Image.fromarray(np.uint8(arr_img))
    pil_img = pil_img.rotate(rotate_degree)
    rotated_image = np.array(pil_img)
    return rotated_image

def random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser
def bclearning_generator(base_generator, batch_size, sample_steps, n_steps):
    assert batch_size >= sample_steps
    assert batch_size % sample_steps == 0
    
    X_cache, y_cache = [], []
    while True:
        for i in range(n_steps):
            while True:
                current_images, current_onehots = next(base_generator)
                if current_images.shape[0] == sample_steps and current_onehots.shape[0] == sample_steps:
                    break
            current_labels = np.sum(np.arange(current_onehots.shape[1]) * current_onehots, axis=-1)
            for j in range(batch_size//sample_steps):
                for k in range(sample_steps):
                    diff_indices = np.where(current_labels != current_labels[k])[0]
                    mix_ind = np.random.choice(diff_indices)
                    rnd = np.random.rand()
                    if rnd < 0.5: rnd = 1.0 - rnd # 主画像を偏らさないために必要
                    mix_img = rnd * current_images[k] + (1.0-rnd) * current_images[mix_ind]
                    mix_onehot = rnd * current_onehots[k] + (1.0-rnd) * current_onehots[mix_ind]
                    
#                     eraser = random_eraser(v_l=mix_img.min(), v_h=mix_img.max())
#                     mix_img = eraser(mix_img)
                    
                    mix_img = preprocess_input(mix_img,**arg_dict)
                    
                    X_cache.append(mix_img)
                    y_cache.append(mix_onehot)
            X_batch = np.asarray(X_cache, dtype=np.float32)
            y_batch = np.asarray(y_cache, dtype=np.float32)
            X_cache, y_cache = [], []
            yield X_batch, y_batch
class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, featurewise_center = False, samplewise_center = False, 
                 featurewise_std_normalization = False, samplewise_std_normalization = False, 
                 zca_whitening = False, zca_epsilon = 1e-06, rotation_range = 0.0, width_shift_range = 0.0, 
                 height_shift_range = 0.0, brightness_range = None, shear_range = 0.0, zoom_range = 0.0, 
                 channel_shift_range = 0.0, fill_mode = 'nearest', cval = 0.0, horizontal_flip = False, 
                 vertical_flip = False, rescale = None, preprocessing_function = None, data_format = None, validation_split = 0.0, 
                 mask_circle = False, image_rotate_range = 0, preprocess = False, random_erase = False):
        # 親クラスのコンストラクタ
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split)
        # 拡張処理のパラメーター
        assert isinstance(mask_circle,bool)
        self.mask_circle = mask_circle
        assert isinstance(image_rotate_range,int)
        self.image_rotate_range = image_rotate_range
        assert isinstance(preprocess,bool)
        self.preprocess = preprocess
        assert isinstance(random_erase,bool)
        self.random_erase = random_erase

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = super().flow(x=x, y=y, batch_size=batch_size, shuffle=shuffle, sample_weight=sample_weight,
                               seed=seed, save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, subset=subset)

        while True:
            batch_x, batch_y = next(batches)
            
            if self.image_rotate_range:
                x = np.zeros(batch_x.shape)
                for i in range(batch_x.shape[0]):
                    x[i] = image_rotate(batch_x[i], self.image_rotate_range)
                batch_x = x
                
            if self.random_erase:
                x = np.zeros(batch_x.shape)
                for i in range(batch_x.shape[0]):
                    eraser = random_eraser(v_l=0, v_h=255)
                    x[i] = eraser(batch_x[i])
                batch_x = x
            
            if self.mask_circle:
                x = np.zeros(batch_x.shape)
                for i in range(batch_x.shape[0]):
                    x[i] = mask_circle_solid(batch_x[i])
                batch_x = x
            
            if self.preprocess:
                x = np.zeros(batch_x.shape)
                for i in range(batch_x.shape[0]):
                    x[i] = preprocess_input(batch_x[i],**arg_dict)
                batch_x = x
            
            yield batch_x , batch_y
datagen_train = MyImageDataGenerator(
        #rotation_range=180,
        zoom_range = 0.03,
        width_shift_range=0.03,
        height_shift_range=0.03,
        horizontal_flip=True,
        vertical_flip=True,
        mask_circle=True,
        brightness_range=[0.4,1.0],
        image_rotate_range=180,
        random_erase=True,
#         preprocess=True,
)

datagen_valid = MyImageDataGenerator(
        mask_circle=True,
        preprocess=True,
)
image_size = 224
X = [] #画像
Y_type = [] #種類
Y_variety = [] #品種

for variety_id_str in tqdm([dir_name[-2:] for dir_name in sorted(glob.glob("../input/fruits-category-classification/fruits_category_classification/*"))]):
    variety_id = int(variety_id_str) - 1
    #フォルダ内のファイルごと
    for img_file_path in glob.glob("../input/fruits-category-classification/fruits_category_classification/" + variety_id_str + "/*"):
        image = Image.open(img_file_path)
        image = image.convert("RGB")
        
        #縦横比を保ったまま正方形にする
        image = expand2square(image)
        
        image = image.resize((image_size, image_size))
        X.append(np.array(image))
        Y_variety.append(variety_id)
        
        if 0 <= variety_id < 8:
            Y_type.append(0)
        elif 8 <= variety_id < 14:
            Y_type.append(1)
        elif 14 <= variety_id < 22:
            Y_type.append(2)
        elif 22 <= variety_id < 30:
            Y_type.append(3)
        elif 30 <= variety_id < 36:
            Y_type.append(4)
        elif 36 <= variety_id < 41:
            Y_type.append(5)
        elif 41 <= variety_id < 46:
            Y_type.append(6)
del image
type_num = len(set(Y_type))
variety_num = len(set(Y_variety))

X = np.array(X)
Y_type = np.array(Y_type)
Y_variety = np.array(Y_variety)
for i in range(1):
    i += 100
    Y_cat = to_categorical(Y_type)
    
    batch = bclearning_generator(datagen_train.flow(X,Y_cat, batch_size=10, seed=0),
                                 batch_size=10,
                                 sample_steps=10,
                                 n_steps=X.shape[0]//10)
    
    n,nrows,ncols = 0,2,5
    fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(20,8))
    for row in range(nrows):
        for col in range(ncols):
            x,y = next(batch)
            ax[row, col].imshow(np.uint8(x[0]))
            n += 1
#     print(x[0].sum())
#     print(x[0][0])
    plt.show()
# image = Image.open("../input/fruits-category-classification/fruits_category_classification/01/01 (12).jpg")
# image = image.convert("RGB")
# plt.imshow(preprocess_input(np.array(image).astype("float32")))
# plt.show()
# preprocess_input(np.array(image).astype("float32")).sum()
# for i in range(10):
#     i += 0
#     ce_ohe = ce.OneHotEncoder(handle_unknown='impute')
#     Y_c = np.array(ce_ohe.fit_transform(Y))

#     batches = datagen_train.flow(np.array(X[i:i+1]).astype("float32"),Y_c[i:i+1],batch_size=10)

#     n = 0
#     nrows = 2
#     ncols = 5
#     fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(20, 8))
#     for row in range(nrows):
#         for col in range(ncols):
#             x,y = next(batches)
#             ax[row, col].imshow(np.uint8(x[0]))
#             n += 1
#     plt.show()
def create_model(output_num, no_trainable_layer):
#     base_model=EfficientNetB1(weights='imagenet',
#                               include_top=False,
#                               pooling="avg",
#                               input_shape=(image_size,image_size,3))
    base_model=ResNeXt50(weights = 'imagenet',
                         include_top = False,
                         pooling="avg",
                         input_shape=(image_size,image_size,3),**arg_dict)
    
    x = base_model.output
#     x = BatchNormalization()(x)
#     x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    
#     x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    prediction = Dense(output_num, activation='softmax')(x)

    for layer in base_model.layers[:-no_trainable_layer]:
#     for layer in base_model.layers:    
        layer.trainable=False
    
    return Model(inputs=base_model.input,outputs=prediction)
def accuracy_of_type(pred,true):
    acc_list = []
    
    true = np.array(true)
    pred = np.array(pred)
    
    acc_list.append(accuracy_score(true[true < 8],pred[true < 8]))
    acc_list.append(accuracy_score(true[(8 <= true) & (true < 14)],pred[(8 <= true) & (true < 14)]))
    acc_list.append(accuracy_score(true[(14 <= true) & (true < 22)],pred[(14 <= true) & (true < 22)]))
    acc_list.append(accuracy_score(true[(22 <= true) & (true < 30)],pred[(22 <= true) & (true < 30)]))
    acc_list.append(accuracy_score(true[(30 <= true) & (true < 36)],pred[(30 <= true) & (true < 36)]))
    acc_list.append(accuracy_score(true[(36 <= true) & (true < 41)],pred[(36 <= true) & (true < 41)]))
    acc_list.append(accuracy_score(true[(41 <= true) & (true < 46)],pred[(41 <= true) & (true < 46)]))
    
    return np.array(acc_list)
def model_train(X, Y, train_model=True, check_result=False, start_fold=0, bc_use=False, no_trainable_layer=70):
    np.set_printoptions(precision=4, floatmode='maxprec')
    
    acc_mean = 0
    acc_type_mean = np.zeros(7)
    error_set = set()
    
    X = X.astype("float32")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_index, test_index) in enumerate(skf.split(X,Y)):
        if fold<start_fold:
            continue
        
        print('#'*25)
        print('### FOLD %i'%(fold+1))
        print('#'*25)
        
        if fold == start_fold:
            Y_cat = to_categorical(Y)
        
#         print(test_index)
        X_train = X[train_index]
        X_valid = X[test_index]
        Y_train = Y_cat[train_index]
        Y_valid = Y_cat[test_index]

        if train_model:
        
#             datagen_train.fit(X_train)
#             datagen_valid.fit(X_valid)
            
#             print(len(set(Y)))
#             print(Y_train[0])
            
            model = create_model(len(set(Y)),no_trainable_layer)
        
            if bc_use:
                model.compile(loss="kullback_leibler_divergence", 
                              optimizer=Adam(lr=0.0005), 
                              metrics=['accuracy'])                
            else:
                model.compile(loss='categorical_crossentropy', 
                              optimizer=Adam(lr=0.0005), 
                              metrics=['accuracy'])
            
            
#             if fold == 0:
#                 model.summary()

            reduceLROnPlateau = ReduceLROnPlateau(monitor='val_accuracy',
                                                    patience=4, 
                                                    verbose=1, 
                                                    factor=0.5, 
                                                    min_lr=0.00001)

            modelCheckpoint = ModelCheckpoint(filepath = "/kaggle/working/model_" + str(fold) + ".h5",
                                                  monitor='val_accuracy',
                                                  verbose=0,
                                                  save_best_only=True,
                                                  period=1)

            earlyStopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='max')
        
            if bc_use:
                datagen_train_bc = bclearning_generator(datagen_train.flow(X_train,Y_train, batch_size=BATCH_SIZE, seed=SEED),
                                                     BATCH_SIZE,
                                                     BATCH_SIZE,
                                                     X_train.shape[0]//BATCH_SIZE)

                hist = model.fit(datagen_train_bc,
                                 validation_data = datagen_valid.flow(X_valid,Y_valid, batch_size=BATCH_SIZE, seed=SEED),
                                 epochs = EPOCHS,
                                 verbose = 2,
                                 steps_per_epoch = X_train.shape[0] // BATCH_SIZE,
                                 validation_steps = X_valid.shape[0] // BATCH_SIZE,
                                 callbacks = [reduceLROnPlateau,modelCheckpoint,earlyStopping]) 
            else:
                hist = model.fit(datagen_train.flow(X_train,Y_train, batch_size=BATCH_SIZE, seed=SEED),
                                 validation_data = datagen_valid.flow(X_valid,Y_valid, batch_size=BATCH_SIZE, seed=SEED),
                                 epochs = EPOCHS,
                                 verbose = 2,
                                 steps_per_epoch = X_train.shape[0] // BATCH_SIZE,
                                 validation_steps = X_valid.shape[0] // BATCH_SIZE,
                                 callbacks = [reduceLROnPlateau,modelCheckpoint,earlyStopping])

#             fig = plt.figure(figsize=(12, 16))
#             fig.subplots_adjust(wspace=0.5)
#             fig.suptitle("hist", fontsize=20)

#             ax1 = fig.add_subplot(211,
#                                   title="Loss:",
#                                   ylabel="Loss",
#                                   xlabel="Epoch")
#             ax1.plot(hist.history["loss"])
#             ax1.legend(["Train", "Test"], loc="upper left")

#             ax2 = fig.add_subplot(212,
#                                   title="val Acc:",
#                                   ylabel="val Acc",
#                                   xlabel="Epoch")
#             ax2.plot(hist.history['val_accuracy'])
#             ax2.legend(["Train", "Test"], loc="upper left")

#             plt.show()
        if check_result:
        
            model = load_model("/kaggle/working/model_" + str(fold) + ".h5", compile=False)

            #validをgeneratorから生成する。
            X_valid_temp = np.zeros(X_valid.shape)
            X_valid_image = X_valid
            for i in range(X_valid.shape[0]):
                temp = mask_circle_solid(X_valid[i])
                X_valid_temp[i] = preprocess_input(temp,**arg_dict)
            X_valid = X_valid_temp

            pred = model.predict(X_valid)
            Y_pred_classes = argmax(pred, axis=1)
            Y_true = argmax(Y_valid , axis=1)

            ACC_pred = accuracy_score(Y_pred_classes, Y_true)
            acc_type = accuracy_of_type(Y_pred_classes, Y_true)
            cm = confusion_matrix(Y_pred_classes, Y_true)
            plt.figure(figsize=(10, 10)) 
            sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
            plt.show()
    #         Y_pred_classes = np.argmax(pred,axis = 1) 

            errors = (Y_pred_classes - Y_true != 0)
            error_set |= set(test_index[errors])
            pickle.dump(error_set,open("error_set.sav","wb"))

            #trainで当てられなかったデータはラベル自体が異なっている可能性あり
    #         X_train_temp = np.zeros(X_train.shape)
    #         for i in range(X_train.shape[0]):
    #             temp = mask_circle_solid(X_train[i])
    #             X_train_temp[i] = preprocess_input(temp)
    #         X_train = X_train_temp

    #         pred_train = model.predict(X_train)
    #         Y_pred_train_classes = argmax(pred_train, axis=1)
    #         Y_true_train = argmax(Y_train , axis=1)

    #         errors_train = (Y_pred_train_classes - Y_true_train != 0)
    #         error_train_set |= set(test_index[errors_train])
    #         pickle.dump(error_train_set,open("error_train_set.sav","wb"))


            #error top10
            Y_pred_classes_errors = Y_pred_classes[errors]
            Y_pred_errors = pred[errors]
            Y_true_errors = Y_true[errors]
            X_val_errors = X_valid_image[errors]/255

            def display_errors(errors_index, img_errors, pred_errors, obs_errors):
                n = 0
                nrows = 2
                ncols = 5
                fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(20, 8))
                for row in range(nrows):
                    for col in range(ncols):
                        error = errors_index[n]
                        ax[row, col].imshow((img_errors[error]).reshape((image_size, image_size,3)))
                        ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
                        n += 1
                plt.show()

            Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
            true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
            delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
            sorted_dela_errors = np.argsort(delta_pred_true_errors)
            most_important_errors = sorted_dela_errors[-10:]

            display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

            #shap
#             def map2layer(x, layer):
#                 feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
#                 return (model.layers[layer].input, feed_dict)
            
#             explainer = shap.DeepExplainer(
#                 (model.layers[layer].input, model.layers[-1].output),
#                 map2layer(X_train[0:100], 7),
#             )

#             #explainer = shap.DeepExplainer(model, (X_train[0:100]))
#             for i in most_important_errors:
#                 shap_values = explainer.shap_values(X_val_errors[[i]])
#                 index_names = np.array([str(x) + "\n" + '{:>7.3%}'.format(Y_pred_errors[i][x]) for x in range(output_num)]).reshape(1,output_num)
#                 print("Predicted label :{}\nTrue label :{}".format(Y_pred_classes_errors[i],Y_true_errors[i]))
                
#                 shap.image_plot(shap_values, X_val_errors[[i]] ,index_names)
#                 plt.gcf().set_size_inches(20, 20)

            acc_mean += ACC_pred/5
            acc_type_mean += acc_type/5
            print('val Acc =' , ACC_pred)
            print('val type Acc =', acc_type)
            print()
            del model
        del X_train
        del X_valid
        del Y_train
        del Y_valid
        
    print('#'*25)
    print('### RESULT')
    print('#'*25)
    print('>>>> Acc mean =', acc_mean)
    print('>>>> Acc type mean =', acc_type_mean)


EPOCHS = 80
BATCH_SIZE = 16
SEED = 0

model_train(X,Y_variety,
            train_model=False,
            check_result=True,
            start_fold=0,
            bc_use=True,
            no_trainable_layer=109)


image_size
X_valid = []
image1 = Image.open("../input/fruits-category-classification/fruits_category_classification/22/22 (1).jpg")
image1 = image1.convert("RGB")
image1 = image1.resize((image_size, image_size))
X_valid.append(np.array(image1))

image2 = Image.open("../input/fruits-category-classification/fruits_category_classification/22/22 (11).jpg")
image2 = image2.convert("RGB")
image2 = image2.resize((image_size, image_size))
X_valid.append(np.array(image2))

image3 = Image.open("../input/fruits-category-classification/fruits_category_classification/22/22 (12).jpg")
image3 = image3.convert("RGB")
image3 = image3.resize((image_size, image_size))
X_valid.append(np.array(image3))
X_valid[0][50][14]
plt.imshow(X_valid[0])
X_valid = np.array(X_valid).astype("float32")

model = load_model("/kaggle/working/model_" + str(0) + ".h5", compile=False)

X_valid_temp = np.zeros(X_valid.shape)
for i in range(X_valid.shape[0]):
    temp_x = mask_circle_solid(X_valid[i])
#     print(temp.sum())
    X_valid_temp[i] = preprocess_input(temp_x)
X_valid_test = X_valid_temp

#model = load_model("../input/model-0/model_" + str(0) + ".h5", compile=False)
print(X_valid_test)
pred = model.predict(X_valid_test)
X_valid_test[0].sum()
pred
argmax(pred, axis=1)

!conda list
