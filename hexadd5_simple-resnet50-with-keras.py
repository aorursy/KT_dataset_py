# LB 0.486

# simple cnn



from keras.preprocessing.image import ImageDataGenerator

from keras.applications import resnet50

from keras import models

from keras import layers

from keras import optimizers

from keras.engine.topology import Input

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard



import os

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



import argparse



# settings

base_dir='../input'

model_base_dir = 'models'

test_dir = os.path.join(base_dir, 'test')

train_dir=os.path.join(base_dir, 'train')

validation_dir=os.path.join(base_dir, 'validation')

# paramater

batch_size=32

SEED=1470

epochs=50

input_size = 224



def split_with_class_count(df, validation_split=0.1, class_count=1):

    df = pd.read_csv(os.path.join(base_dir, 'train.csv'))

    df = df[df.Id != 'new_whale']

    classes = df['Id'].unique()

    df['count'] = df.groupby('Id')['Id'].transform('count')

    fdf = df[df['count'] >= class_count]

    val_classes = fdf['Id'].unique()

    train_df = pd.DataFrame(columns=fdf.columns)

    validation_df = pd.DataFrame(columns=fdf.columns)

    for val_class in val_classes:

      class_df = fdf[fdf.Id == val_class]

      validation = class_df.sample(frac=validation_split, random_state=SEED)

      validation_df = pd.concat([validation_df, validation]) 

      train = class_df.drop(validation.index)

      train_df = pd.concat([train_df, train]) 

    train_df = train_df.drop('count', axis=1)

    train_df = train_df.reset_index()

    validation_df = validation_df.drop('count', axis=1)

    validation_df = validation_df.reset_index()

    return train_df, validation_df, classes.tolist()



def load_data():

    df = pd.read_csv(os.path.join(base_dir, 'train.csv'))

    df = df[df.Id != 'new_whale'] # without new_whale

    train_df, validation_df, classes = split_with_class_count(df, validation_split=0.01, class_count=50)

    datagen = ImageDataGenerator(

        rescale=1./255,

        horizontal_flip=True,

        rotation_range=30,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        brightness_range=[0.7, 1.0],

    )

    train_generator = datagen.flow_from_dataframe(

        dataframe=df,

        directory=train_dir,

        x_col='Image',

        y_col='Id',

        target_size=(input_size, input_size),

        batch_size=batch_size,

        classes=classes,

        seed=SEED,

    )

    datagen = ImageDataGenerator(

        rescale=1./255,

    )

    val_generator = datagen.flow_from_dataframe(

        dataframe=validation_df,

        directory=train_dir,

        x_col='Image',

        y_col='Id', 

        target_size=(input_size, input_size),

        batch_size=20,

        classes=classes,

        seed=SEED,

    )

    return train_generator, val_generator

from keras import regularizers

class ModelV7():

    def __init__(self):

        self.name = 'v7'

    def get_model(self):

        conv_base = resnet50.ResNet50(weights='imagenet',

                                      include_top=False,

                                      input_shape=(input_size, input_size, 3))

        for layer in conv_base.layers[:]:

            if 'BatchNormalization' in str(layer):

                layer.trainable = True

            else:

                layer.trainable = False

        main_input = conv_base.input

        embedding = conv_base.output

        x = layers.GlobalMaxPooling2D()(embedding)

        x = layers.Dropout(0.5)(x)

        x = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

        x = layers.BatchNormalization()(x)

        x = layers.Dropout(0.5)(x)

        x = layers.Dense(5004, activation='softmax')(x)

        model = models.Model(inputs=[main_input], outputs=[x])

        model.compile(

            loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model



def load_classes():

    train_generator, val_generator = load_data()

    return np.array([c for c, v in train_generator.class_indices.items()])



def load_test_data():

    datagen = ImageDataGenerator(

        rescale=1./255,

    )

    test_generator = datagen.flow_from_dataframe(

        pd.DataFrame(os.listdir(test_dir),columns=['filename']),

        test_dir,

        target_size=(input_size, input_size),

        batch_size=1,

        class_mode=None,

        shuffle=False,

        seed=SEED,

    )



    if len(test_generator) == 0:

        print('Train data not found')

        exit()

    return test_generator



def predict(model, generator):

    kth = 5

    generator.reset()

    pred = model.predict_generator(generator, steps=len(generator), verbose=1)

    classes = load_classes()

    classify_index = np.argpartition(-pred, kth)[:, :kth]

    classify_value = pred[np.arange(pred.shape[0])[:, None], classify_index]

    best_5_pred = np.zeros((len(classify_index), 5))

    best_5_class = np.zeros((len(classify_index), 5), dtype='int32')

    for i, p in enumerate(classify_value):

        sort_index = np.argsort(p)[::-1]

        best_5_pred[i] = (p[sort_index])

        best_5_class[i] = (classify_index[i][sort_index])

    # create output

    submit = pd.DataFrame(columns=['Image', 'Id'])

    for i, p in enumerate(best_5_pred):

        submit_classes = []

        if p[0] < 0.55:

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][0:4])

        elif p[1] < 0.4 :

            submit_classes.extend(classes[best_5_class[i]][0:1])

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][1:4])

        elif p[2] < 0.1 :

            submit_classes.extend(classes[best_5_class[i]][0:2])

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][2:4])

        elif p[3] < 0.05 :

            submit_classes.extend(classes[best_5_class[i]][0:3])

            submit_classes.append('new_whale')

            submit_classes.extend(classes[best_5_class[i]][3:4])

        else:

            submit_classes.extend(classes[best_5_class[i]])

        classes_text = ' '.join(submit_classes)

        submit = submit.append(pd.Series(np.array([generator.filenames[i], classes_text]), index=submit.columns), ignore_index=True)

    return submit





if __name__ == '__main__':

        epochs = 25

        train_generator, val_generator = load_data()

        steps_per_epochs = len(train_generator)

        model_wrapper = ModelV7()

        model = model_wrapper.get_model()

        model.summary()



        model_dir = os.path.join(model_base_dir, 'without_whale' + model_wrapper.name)



        os.makedirs(model_dir, exist_ok=True)



        model_checkpoint_path = os.path.join(model_dir, '{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5')

        model_checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)

        tensor_board = TensorBoard(log_dir=os.path.join(model_dir))

        history = model.fit_generator(

            train_generator,

            steps_per_epoch=steps_per_epochs,

            epochs=epochs,

            validation_data=val_generator,

            validation_steps=50,

        )



        # load test data

        test_generator = load_test_data()

        # predict

        submit = predict(model, test_generator)

        submit.to_csv('submit4.csv', index=False)



        