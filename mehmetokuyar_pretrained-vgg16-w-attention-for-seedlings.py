!mkdir ~/.keras
!mkdir ~/.keras/models
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob 
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from skimage.util.montage import montage2d
from skimage.io import imread
base_dir = os.path.join('..', 'input', 'plant-seedlings-classification')

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
cat_enc = LabelEncoder()
img_paths = glob(os.path.join(base_dir, 'train', '*', '*.*'))
print('Image Files', len(img_paths))
all_paths_df = pd.DataFrame(dict(path = img_paths))
all_paths_df['category'] = all_paths_df['path'].map(lambda x: x.split('/')[-2])
cat_enc.fit(all_paths_df['category'])
all_paths_df['cat_id'] = all_paths_df['category'].map(lambda x: 
                                                      cat_enc.transform([x])[0])
all_paths_df['cat_vec'] = all_paths_df['cat_id'].map(lambda x: to_categorical(x, len(cat_enc.classes_)))
all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_paths_df.sample(5)
all_paths_df[['cat_id']].hist(figsize = (10, 5))
from sklearn.model_selection import train_test_split
raw_train_df, valid_df = train_test_split(all_paths_df, 
                                   test_size = 0.02, 
                                   random_state = 2018,
                                   stratify = all_paths_df[['cat_id']])
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
raw_train_df.sample(1)
train_df = raw_train_df.groupby(['cat_id']).apply(lambda x: x.sample(500, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
train_df[['cat_id']].hist(bins = len(cat_enc.classes_), figsize = (10, 5))
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
ppi = lambda x: Image.fromarray(preprocess_input(np.array(x).astype(np.float32)))
IMG_SIZE = (224, 224) # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.2)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'cat_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 32)
valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'cat_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'cat_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 1024)) # one big batch
t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:].clip(0, 255).astype(np.uint8))
    c_ax.set_title('%s' % cat_enc.classes_[np.argmax(c_y)])
    c_ax.axis('off')
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
in_lay = Input(t_x.shape[1:])
base_pretrained_model = VGG16(input_shape =  t_x.shape[1:], 
                              include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
from keras.layers import BatchNormalization
bn_features = BatchNormalization()(pt_features)

# here we do an attention mechanism to turn pixels in the GAP on an off

attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_features)
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(1, 
                    kernel_size = (1,1), 
                    padding = 'valid', 
                    activation = 'sigmoid')(attn_layer)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.5)(Dense(128, activation = 'elu')(gap_dr))
out_layer = Dense(len(cat_enc.classes_), activation = 'softmax')(dr_steps)
tb_model = Model(inputs = [in_lay], outputs = [out_layer])

tb_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])

tb_model.summary()
!rm -rf ~/.keras # clean up the model / make space for other things
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('seedlings')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
tb_model.fit_generator(train_gen, 
                      steps_per_epoch = 40,
                      validation_data = (test_X, test_Y), 
                      epochs = 8, 
                      callbacks = callbacks_list)
# load the best version of the model
tb_model.load_weights(weight_path)
# get the attention layer since it is the only one with a single output dim
for attn_layer in tb_model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break
import keras.backend as K
rand_idx = np.random.choice(range(len(test_X)), size = 6)
attn_func = K.function(inputs = [tb_model.get_input_at(0), K.learning_phase()],
           outputs = [attn_layer.get_output_at(0)]
          )
fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
    cur_img = test_X[c_idx:(c_idx+1)]
    attn_img = attn_func([cur_img, 0])[0]
    img_ax.imshow(cur_img[0,:,:].clip(0, 255).astype(np.uint8))
    attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis', 
                   vmin = 0, vmax = 1, 
                   interpolation = 'lanczos')
    real_label = test_Y[c_idx]
    img_ax.set_title('TB\nClass:%s' % (cat_enc.classes_[np.argmax(real_label)]))
    pred_confidence = tb_model.predict(cur_img)[0]
    attn_ax.set_title('Attention Map\nPred:%2.1f%%' % (100*pred_confidence[np.argmax(real_label)]))
fig.savefig('attention_map.png', dpi = 300)
pred_Y = tb_model.predict(test_X, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
from sklearn.metrics import classification_report, confusion_matrix
plt.matshow(confusion_matrix(np.argmax(test_Y,-1), pred_Y_cat))
print(classification_report(np.argmax(test_Y,-1), pred_Y_cat.astype(int), 
                            target_names = cat_enc.classes_))
dense_layers = [x for x in tb_model.layers if isinstance(x, Dense)]
last_layer = bn_features
for i, c_d in enumerate(dense_layers):
    W, b = c_d.get_weights()
    in_dim, out_dim = W.shape
    new_W = np.expand_dims(np.expand_dims(W,0),0)
    new_b = b
    last_layer = Conv2D(out_dim, 
                        kernel_size = (1,1), 
                        weights = (new_W, new_b),
                        activation = c_d.activation, 
                        name = 'd2cv_{}'.format(i))(last_layer)
viz_model = Model(inputs = [in_lay], 
                  outputs = [last_layer],
                  name = 'viz_model')
viz_model.summary()
rand_idx = np.random.choice(range(len(test_X)), size = 4)
ch_count = viz_model.get_output_shape_at(0)[-1]
fig, m_axs = plt.subplots(len(rand_idx), 
                          1+ch_count, 
                          figsize = (4*(1+ch_count), 
                                     4*len(rand_idx)))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_idx, n_axs in zip(rand_idx, m_axs):
    cur_img = test_X[c_idx:(c_idx+1)]
    attn_img = viz_model.predict([cur_img])
    pred_confidence = tb_model.predict(cur_img)[0]
    n_axs[0].imshow(cur_img[0,:,:].clip(0, 255).astype(np.uint8))
    real_label = np.argmax(test_Y[c_idx])
    n_axs[0].set_title('Class:%s' % (cat_enc.classes_[real_label]))
    for i, c_ax in enumerate(n_axs[1:]):
        c_ax.imshow(attn_img[0, :, :, i], cmap = 'viridis', 
                       vmin = 0, vmax = 1, 
                       interpolation = 'lanczos')
        c_ax.set_title('%s Map\nPred:%2.1f%%' % (cat_enc.classes_[i], 
                                                 100*pred_confidence[i]))
    
fig.savefig('positive_map.png', dpi = 300)
test_paths = glob(os.path.join(base_dir, 'test', '*.*'))
print('Image Files', len(test_paths))
test_paths_df = pd.DataFrame(dict(path = test_paths))
test_paths_df['file'] = test_paths_df['path'].map(os.path.basename)
test_paths_df.sample(3)
# used a fixed dataset for evaluating the algorithm
test_X, _ = next(flow_from_dataframe(core_idg, 
                               test_paths_df, 
                             path_col = 'path',
                            y_col = 'path', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                             shuffle = False,
                            batch_size = len(test_paths))) # one big batch
pred_Y = tb_model.predict(test_X, batch_size = 8, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_paths_df['species'] = [cat_enc.classes_[x] for x in pred_Y_cat]
test_paths_df[['file', 'species']].to_csv('submission.csv', index=False)
test_paths_df[['file', 'species']].head(5)
