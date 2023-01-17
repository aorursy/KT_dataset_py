import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
# not needed in Kaggle, but required in Jupyter
%matplotlib inline 
data_dir = os.path.join('..', 'input')
all_files = glob(os.path.join(data_dir, '*', '*'))
all_df = pd.DataFrame(dict(path = [x for x in all_files if x.endswith('png') or x.endswith('gui')]))
all_df['source'] = all_df['path'].map(lambda x: x.split('/')[-2])
all_df['filetype'] = all_df['path'].map(lambda x: os.path.splitext(x)[1][1:])
all_df['fileid'] = all_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_df.sample(3)
all_data_df = all_df.pivot_table(index=['source', 'fileid'], 
                                 columns = ['filetype'], 
                                 values = 'path',
                                 aggfunc='first').reset_index()
print(all_data_df.shape[0], 'samples for training and validation')
all_data_df.sample(3)
from PIL import Image as PImage
def read_text_file(in_path):
    with open(in_path, 'r') as f:
        return f.read()

def imread_scale(in_path):
    return np.array(PImage.open(in_path).resize((64, 64), PImage.ANTIALIAS))[:,:,:3]

clear_sample_df = all_data_df.groupby(['source']).apply(lambda x: x.sample(1)).reset_index(drop = True)

fig, m_axs = plt.subplots(3, clear_sample_df.shape[0], figsize = (12, 12))
for (_, c_row), (im_ax, im_lr, gui_ax) in zip(clear_sample_df.iterrows(), m_axs.T):
    
    im_ax.imshow(imread(c_row['png']))
    im_ax.axis('off')
    im_ax.set_title(c_row['source'])
    
    im_lr.imshow(imread_scale(c_row['png']), interpolation = 'none')
    im_lr.set_title('LowRes')
    
    gui_ax.text(0, 0, read_text_file(c_row['gui']), 
            style='italic',
            bbox={'facecolor':'yellow', 'alpha':0.1, 'pad':10},
               fontsize = 7)
    gui_ax.axis('off')

%%time
all_data_df['code'] = all_data_df['gui'].map(read_text_file)
all_data_df['img'] = all_data_df['png'].map(imread_scale)
all_data_df.sample(2)
from keras.preprocessing.text import Tokenizer
w_token = Tokenizer()
w_token.fit_on_texts(all_data_df['code'].values)
from sklearn.preprocessing import LabelEncoder
source_lab_enc = LabelEncoder()
source_lab_enc.fit(all_data_df['source'])
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Dense, Input, Embedding, concatenate, Dropout
v_in = Input((None,), name = 'SequenceIn')
cat_in = Input((len(source_lab_enc.classes_),), name = 'SourceIn')
cat_vec = Dense(64)(cat_in)
em_in = Embedding(len(w_token.word_index)+1, 
                  len(w_token.word_index)+1)(v_in)
lstm_1 = Bidirectional(LSTM(128, return_sequences = True))(em_in)
lstm_2 = LSTM(512, return_sequences = False)(lstm_1)
lstm_2 = Dropout(0.5)(lstm_2)
comb_vec = concatenate([lstm_2, cat_vec])
dr_vec = Dropout(0.5)(comb_vec)
seq_model = Model(inputs = [v_in, cat_in], outputs = [dr_vec])
seq_model.summary()
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, BatchNormalization, Conv2D, UpSampling2D
model = Sequential(name = 'seq_to_img')
base_size = (8, 8, 96)
model.add(Dense(np.prod(base_size), input_dim = seq_model.get_output_shape_at(0)[1], activation = 'linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Reshape(base_size, input_shape=(96*8*8,)))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation = 'linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation = 'linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(3, (3, 3), padding='same', activation = 'sigmoid'))
model.summary()
full_out = model(dr_vec)
full_model = Model(inputs = [v_in, cat_in], outputs = [full_out])
full_model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 
                                                                'binary_crossentropy'])
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(all_data_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_data_df['source'])
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
def make_train_block(in_df, max_seq_len = 64):
    img_vec = np.stack(in_df['img'].values,0)/255.0
    src_vec = to_categorical(source_lab_enc.transform(in_df['source']))
    seq_vec = pad_sequences(w_token.texts_to_sequences(in_df['code']), maxlen = max_seq_len)
    return [[seq_vec, src_vec], img_vec]
X_train = make_train_block(train_df)
X_test = make_train_block(test_df)
print('train', X_train[0][0].shape, X_train[0][1].shape, X_train[1].shape, 'test', X_test[1].shape)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('code_to_gui')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', 
                             save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                   factor=0.8, 
                                   patience=3,
                                   verbose=1, 
                                   mode='auto', 
                                   epsilon=0.0001, 
                                   cooldown=5, min_lr=1e-5)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)
callbacks_list = [checkpoint, early]
full_model.fit(X_train[0], X_train[1], 
               batch_size = 128, 
               shuffle = True, 
               validation_data = X_test,
              epochs = 1, 
              callbacks = callbacks_list)
def preview_results():
    preview_rows = test_df.groupby(['source']).apply(lambda x: x.sample(2)).reset_index(drop = True)
    preview_rows['pred'] = [np.clip(x*255, 0, 255).astype(np.uint8)
                            for x in full_model.predict(make_train_block(preview_rows)[0])]
    fig, m_axs = plt.subplots(3, preview_rows.shape[0], figsize = (16, 18))

    for (_, c_row), (gui_ax, im_ax, im_lr) in zip(preview_rows.iterrows(), m_axs.T):

        im_ax.imshow(imread(c_row['png']))
        im_ax.axis('off')
        im_ax.set_title('Real Image')

        im_lr.imshow(c_row['pred'], interpolation = 'none')
        im_lr.set_title('Prediction')
        im_lr.axis('off')

        gui_ax.text(0, 0, read_text_file(c_row['gui']), 
                style='italic',
                bbox={'facecolor': 'yellow', 
                      'alpha': 0.1, 
                      'pad': 5},
                   fontsize = 4)
        gui_ax.axis('off')
        gui_ax.set_title(c_row['source'])
    return fig
preview_results().savefig('initial_results.png', dpi = 300)
full_model.fit(X_train[0], X_train[1], 
               batch_size = 256, 
               shuffle = True, 
               validation_data = X_test,
              epochs = 30, 
              callbacks = callbacks_list)
full_model.load_weights(weight_path)
preview_results().savefig('trained_results.png', dpi = 300)
