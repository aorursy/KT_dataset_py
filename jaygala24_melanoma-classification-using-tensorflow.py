import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import Xception
import tensorflow.keras.backend as K
from PIL import Image
input_path = '/kaggle/input/siim-isic-melanoma-classification'

train_images_path = os.path.join(input_path, 'jpeg', 'train')
test_images_path = os.path.join(input_path, 'jpeg', 'test')
train_df_path = os.path.join(input_path, 'train.csv')
test_df_path = os.path.join(input_path, 'test.csv')
train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)
train_df.head()
# is the data balanced?
train_df['target'].value_counts()
# % of benign and malign samples
print('% benign: {:.4f}'.format(sum(train_df['target'] == 0) / len(train_df)))
print('% malign: {:.4f}'.format(sum(train_df['target'] == 1) / len(train_df)))
def plot_images(data, nrows=5, ncols=5, target=0):
    data = data[data['target'] == target].sample(nrows * ncols)['image_name']
    plt.figure(figsize=(nrows * 2.5, ncols * 2.5))
    for idx, image_name in enumerate(data):
        image = Image.open(os.path.join(train_images_path, image_name + '.jpg'))
        plt.subplot(nrows, ncols, idx + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show();
# plot the benign images
print('Benign Samples')
plot_images(train_df)
# plot the malign images
print('Malign Samples')
plot_images(train_df, target=1)
# prepare the data: (training_images, labels)
train_images = train_df['image_name'].apply(lambda img_path: os.path.join(train_images_path, img_path + '.jpg')).values
test_images = test_df['image_name'].apply(lambda img_path: os.path.join(test_images_path, img_path + '.jpg')).values

train_labels = train_df['target'].values
# convert to dataframe for flow_from_dataframe
train_data = pd.DataFrame({'image': train_images, 'target': train_labels})
test_data = pd.DataFrame({'image': test_images})

train_data.head()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180.0,
    shear_range=2.0,
    zoom_range=8.0,
    width_shift_range=8.0,
    height_shift_range=8.0,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
# reference :~ https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66146
def focal_loss(alpha=0.5, gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        # compute binary cross entropy loss
        bce_loss = binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        alpha_factor = y_true * alpha + (1 - y_true) * (1- alpha)
        modulating_factor = K.pow((1 - p_t), gamma)
        
        # compute and return final loss
        return K.mean(alpha_factor * modulating_factor * bce_loss, axis=-1)
    return focal_crossentropy
def create_model():
    xception = Xception(weights='imagenet', include_top=False, input_shape=(229, 229, 3))
    output =  GlobalAveragePooling2D()(xception.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=xception.input, outputs=output)
    return model
# Training Block
skf = StratifiedKFold(n_splits=5)
n_epochs = 50
train_bs = 32
valid_bs = 16

for idx, (train_idx, val_idx) in enumerate(skf.split(train_data['image'], train_data['target'])):

    print('Fold: {:02d}'.format(idx))
    print('='*16)
    
    # train and validation generator for image preprocessing
    train_generator = train_datagen.flow_from_dataframe(train_data.iloc[train_idx], x_col='image', y_col='target',
                                                        target_size=(224, 224), batch_size=train_bs, shuffle=True,
                                                        class_mode='raw')
    val_generator = val_datagen.flow_from_dataframe(train_data.iloc[val_idx], x_col='image', y_col='target',
                                                    target_size=(224, 224), batch_size=valid_bs, shuffle=False,
                                                    class_mode='raw')
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    model = create_model()
    
    checkpoint_dir = 'PATH_TO_CHECKPOINT_DIR'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_filename = 'model_kfolds_{:02d}.hdf5'.format(idx)
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    # callbacks for model
    early_cb = EarlyStopping(patience=5, mode='max')
    reduce_lr_cb = ReduceLROnPlateau(patience=3, min_lr=0.001, mode='max')
    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, mode='max')
    
    # compile model
    model.compile(loss=focal_loss(),
                  optimizer=keras.optimizers.Adam(3e-4),
                  metrics=[keras.metrics.AUC()],
    )
    
    model.fit(train_generator, 
              steps_per_epoch=steps_per_epoch,
              epochs=n_epochs, 
              validation_data=val_generator,
              validation_steps=validation_steps,
              callbacks=[checkpoint_cb, reduce_lr_cb, early_cb],
    )

    print()

# Prediction Block
checkpoint_dir = '/kaggle/input/melanoma-contest-model-weights'

test_generator = val_datagen.flow_from_directory(os.path.join(test_images_path, '..'), classes=['test'], class_mode=None,
                                                         target_size=(224, 224), shuffle=False)

filenames = test_generator.filenames

final_predictions = np.zeros((test_data.shape[0], 1))

for filename in os.listdir(checkpoint_dir):
    if filename.endswith('.hdf5'):
        print('Loading {} for prediction'.format(filename))
        
        # define model
        model = create_model()

        # load the weights from checkpoint dir
        model.load_weights(os.path.join(checkpoint_dir, filename))

        # model predicts
        predictions = model.predict(test_generator)

        final_predictions += predictions

        print()

final_predictions = final_predictions / 5
sample = pd.DataFrame({'image_name': filenames, 'target': final_predictions.ravel()})
sample['image_name'] = sample['image_name'].apply(lambda img_path: os.path.basename(img_path).split('.')[0])
sample.to_csv('submission.csv', index=False)
sample.head()
