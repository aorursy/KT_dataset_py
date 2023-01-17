import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

!git clone https://github.com/aielawady/horaira.git
import horaira.augmentors

import horaira.im_proc

import horaira.utils
# Removing .git dir because Kaggle gives error related to the depth of the dir.

!rm -r horaira/.git/
preprocessed_path = 'preprocess_images/'

competition_base_path = '../input/aptos2019-blindness-detection/'

image_size = 300

scale_value = 1

aspect_ratio = 1

input_aspect_ratio = 1

NUM_CLASSES = 5
df_train = pd.read_csv(competition_base_path + 'train.csv')

df_test = pd.read_csv(competition_base_path + 'test.csv')

x = df_train['id_code'].values

y = df_train['diagnosis'].values



train_x,train_y,valid_x,valid_y = horaira.utils.balanced_valid_set_splitter(x, y, valid_n_per_class = 100, debug=True)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)
encoding_decoding_methods = {

    'train_encoding':'one',        # 'one', 'all_lower_ones' or 'pairs'

    'valid_encoding': 'one',       # 'one', 'all_lower_ones' or 'pairs'

    'decoding':'max'               # 'max' or 'highest_true'

}



tmp = np.arange(5)

print("\nSample input 1:")

print(tmp)

tmp = horaira.utils.classes_encoder(tmp,5,method=encoding_decoding_methods['train_encoding'])

print("\nEncoding using '{}' method:".format(encoding_decoding_methods['train_encoding']))

print(tmp)

tmp = horaira.utils.classes_decoder(tmp,method=encoding_decoding_methods['decoding'])

print("\nDecoding using '{}' method:".format(encoding_decoding_methods['decoding']))

print(tmp)

tmp = horaira.utils.np.random.rand(6,5)

print("\nSample input 2:")

print(tmp)

tmp = horaira.utils.classes_decoder(tmp,method=encoding_decoding_methods['decoding'])

print("\nDecoding using '{}' method:".format(encoding_decoding_methods['decoding']))

print(tmp)

circle_centering_params = {

    'circle_detection_method':'moments',      # 'enclosing_circle', 'moments' or 'max_dim'

    'scale_value' : scale_value,

    'aspect_ratio' : aspect_ratio,

    'width' : image_size, 

    'gray_threshold' : 10

}





preprocess_sequence_step = [horaira.im_proc.circle_centering, horaira.im_proc.veins_spots_highlighter]

preprocess_params_step = [circle_centering_params, {}]



lister = np.random.choice(df_train['id_code'],8)



imgs_orig = horaira.utils.apply_preprocess(lister,src_path=competition_base_path+'train_images/', preprocessing_function={}, preprocessing_params={})

imgs = horaira.utils.apply_preprocess(lister,src_path=competition_base_path+'train_images/', preprocessing_function=preprocess_sequence_step, preprocessing_params=preprocess_params_step)

plt.figure(figsize=(20,20))

for i in range(len(imgs) * 2):

    plt.subplot(4,4,i+1)

    if i%2 == 0:

        plt.imshow(imgs_orig[i//2].astype('uint8'))

    else:

        plt.imshow(imgs[i//2].astype('uint8'))

preprocess_sequence_step = [horaira.im_proc.circle_centering]

preprocess_params_step = [circle_centering_params]



lister = np.random.choice(df_train['id_code'],8)



imgs_orig = horaira.utils.apply_preprocess(lister,src_path=competition_base_path+'train_images/', preprocessing_function={}, preprocessing_params={})

imgs = horaira.utils.apply_preprocess(lister,src_path=competition_base_path+'train_images/', preprocessing_function=preprocess_sequence_step, preprocessing_params=preprocess_params_step)

plt.figure(figsize=(20,20))

for i in range(len(imgs) * 2):

    plt.subplot(4,4,i+1)

    if i%2 == 0:

        plt.imshow(imgs_orig[i//2].astype('uint8'))

    else:

        plt.imshow(imgs[i//2].astype('uint8'))

!mkdir -p $preprocessed_path/train

horaira.utils.apply_preprocess(df_train['id_code'][:100].values,src_path=competition_base_path+'train_images/', dst_path=preprocessed_path+'train/',

                 preprocessing_function=preprocess_sequence_step, preprocessing_params=preprocess_params_step, write=True)
train_x_aug, train_y_aug = horaira.augmentors.crop_augmentor(df_train['id_code'].values[:100], df_train['diagnosis'].values[:100], preprocessed_path, 10)
plt.figure(figsize=(20,20))

for i in range(4):

    plt.subplot(2,2,i+1)

    plt.imshow(plt.imread(preprocessed_path+'train/'+train_x_aug[-i]+'.png'))