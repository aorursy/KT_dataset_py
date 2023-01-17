import numpy as np

import os

import tensorflow as tf

import tensorflow.keras as K

import matplotlib.pyplot as plt
seed = 24

batch_size = 32

num_classes = 16
classes = os.listdir('../input/miniimagenet-itjim-internship-2020-task6/dataset/train')
shed = K.optimizers.schedules.ExponentialDecay(

    0.01, 100000, decay_rate=0.95)

opt = K.optimizers.Adam(learning_rate=shed)
def build_fc_model(channels=3, img_size=84):

    

    inputs = K.layers.Input((img_size, img_size, channels))

    flat = K.layers.Flatten()(inputs)

    first_dense = K.layers.Dense(

        channels*img_size**2)(flat)

    first_dense = K.layers.LeakyReLU()(first_dense)

    drop = K.layers.Dropout(0.3)(first_dense)

    

    second_dense = K.layers.Dense(img_size*5)(drop)

    second_dense = K.layers.LeakyReLU()(second_dense)

    last_dense = K.layers.Dense(img_size*2)(second_dense)

    last_dense = K.layers.LeakyReLU()(last_dense)

    outputs = K.layers.Dense(num_classes, activation='softmax')(last_dense)



    model = K.models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=opt,

                    loss='categorical_crossentropy',

                    metrics=['categorical_accuracy', 

                             K.metrics.Recall(), 

                             K.metrics.Precision()])



    return model
def residual_block(x):

    filters = x.shape[-1]

    y = K.layers.Conv2D(filters, (3,3), padding="same")(x)

    l_relu = K.layers.LeakyReLU()(y)

    bn = K.layers.BatchNormalization()(l_relu)

    y = K.layers.Conv2D(filters, (3, 3), padding="same")(bn)

    

    out = K.layers.Add()([x, y])

    out = K.layers.BatchNormalization()(out)

    out = K.layers.Conv2D(filters, (3, 3), padding="same")(out)

    return out



def build_cnn_model(channels=3, img_size=84):

    

    inputs = K.layers.Input((img_size, img_size, channels))

    # inputs = K.layers.Lambda(lambda x: x / 255)(inputs)



    conv1 = K.layers.Conv2D(channels, (3, 3), padding='same')(inputs)

    conv1 = K.layers.LeakyReLU()(conv1)

    conv1 = K.layers.Dropout(0.3)(conv1)

    conv2 = K.layers.Conv2D(channels*5, (3, 3), padding='same')(conv1)

    conv2 = K.layers.LeakyReLU()(conv2)

    pool1 = K.layers.MaxPooling2D((2, 2))(conv2)

    

    bottleneck = K.layers.Conv2D(channels, (3, 3), padding='same')(pool1)

    bottleneck = K.layers.LeakyReLU()(bottleneck)

    bottleneck = K.layers.Dropout(0.3)(bottleneck)

    conv_last = K.layers.Conv2D(1, (3, 3), padding='same')(bottleneck)

    conv_last = K.layers.LeakyReLU()(conv_last)

    pool2 = K.layers.MaxPooling2D((2, 2))(conv_last)

    pool2 = K.layers.Flatten()(pool2)

    outputs = K.layers.Dense(num_classes, activation='softmax')(pool2)

    

    model = K.models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=opt,

                    loss='categorical_crossentropy',

                    metrics=['categorical_accuracy', K.metrics.Recall(), K.metrics.Precision()])

                  # metrics=['accuracy'])



    return model



def build_resnet_model(channels=3, img_size=84):

    

    inputs = K.layers.Input((img_size, img_size, channels))

    # inputs = K.layers.Lambda(lambda x: x / 255)(inputs)



    conv1 = K.layers.Conv2D(channels, (3, 3), padding='same')(inputs)

    conv1 = K.layers.LeakyReLU()(conv1)

    conv1 = K.layers.Dropout(0.3)(conv1)

    res = residual_block(conv1)

    conv2 = K.layers.Conv2D(channels, (3, 3), padding='same')(res)

    conv2 = K.layers.LeakyReLU()(conv2)

    pool = K.layers.MaxPooling2D((2, 2))(conv2)

    pool = K.layers.Flatten()(pool)

    outputs = K.layers.Dense(num_classes, activation='softmax')(pool)

    

    model = K.models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=opt,

                    loss='categorical_crossentropy',

                    metrics=['categorical_accuracy', K.metrics.Recall(), K.metrics.Precision()])

                  # metrics=['accuracy'])



    return model
train_set = K.preprocessing.image_dataset_from_directory(

    '../input/miniimagenet-itjim-internship-2020-task6/dataset/train',

    color_mode="rgb", batch_size=batch_size, image_size=(84, 84),

    shuffle=True, seed=seed, label_mode='categorical')



val_set = K.preprocessing.image_dataset_from_directory(

    '../input/miniimagenet-itjim-internship-2020-task6/dataset/val',

    color_mode="rgb", batch_size=batch_size, image_size=(84, 84),

    shuffle=True, seed=seed, label_mode='categorical')



test_set = K.preprocessing.image_dataset_from_directory(

    '../input/miniimagenet-itjim-internship-2020-task6/dataset/test',

    color_mode="rgb", batch_size=batch_size, image_size=(84, 84),

    shuffle=True, seed=seed, label_mode='categorical')
def show_sample(dataset, grid=(4, 4)):

    fig, ax = plt.subplots(grid[0], grid[1])

    i = j = 0

    for batch in dataset.take(1):



        for x in batch:

            for c, img in enumerate(x):

                #print(img.shape)

                ax[i][j].imshow(np.uint8(img))

                ax[i][j].set_title('{}'.format(

                    classes[c]), fontsize=11)

                ax[i][j].axis('off')

                j += 1

                if j == grid[0]:

                    j = 0

                    i += 1

                if i == grid[1]:

                    break

            plt.tight_layout()

            plt.show()

            break
train_gen = K.preprocessing.image.ImageDataGenerator(

    horizontal_flip=True, rescale=1/255)

train_gen = train_gen.flow_from_directory(

        '../input/miniimagenet-itjim-internship-2020-task6/dataset/train',

        target_size=(84, 84),

        batch_size=batch_size,

        class_mode='categorical')



val_gen = K.preprocessing.image.ImageDataGenerator(

    rescale=1/255)

val_gen = val_gen.flow_from_directory(

        '../input/miniimagenet-itjim-internship-2020-task6/dataset/val',

        target_size=(84, 84),

        batch_size=batch_size,

        class_mode='categorical')



test_gen = K.preprocessing.image.ImageDataGenerator(

    rescale=1/255)

test_gen = test_gen.flow_from_directory(

        '../input/miniimagenet-itjim-internship-2020-task6/dataset/test',

        target_size=(84, 84),

        batch_size=batch_size,

        class_mode='categorical')
epochs = 10
best_chpt = K.callbacks.ModelCheckpoint(filepath='{epoch}_{model}_best.ckpt',

                                        monitor='val_loss',

                                        mode='max',

                                        save_best_only=True, 

                                        save_weights_only=True,

                                       verbose=1)

runtime_chpt = K.callbacks.ModelCheckpoint(filepath='{epoch}_{model}_last.ckpt',

                                           monitor="val_loss",

                                           save_weights_only=True,

                                           save_freq=int(batch_size*5))
fc_model = build_fc_model()

"""

fc_history = fc_model.fit(

    train_set, steps_per_epoch=len(train_set)//batch_size, 

    epochs=epochs, validation_data=val_set)

    # callbacks=[best_chpt])

"""

fc_history = fc_model.fit_generator(

    train_gen, epochs=epochs, validation_data=val_gen,

    steps_per_epoch=len(train_gen)//batch_size)

# I might get back to checkpointing but for now it seems as a waste of time

preds = fc_model.predict(test_set)

#preds = np.argmax(probs, axis=1)

classes = train_set.class_names

#classes = [classes[x] for x in preds]

show_sample(test_set)

fc_model.save_weights('fc_model_weights.h5')
cnn_model = build_cnn_model()

"""

fc_history = fc_model.fit(

    train_set, steps_per_epoch=len(train_set)//batch_size, 

    epochs=epochs, validation_data=val_set)

    # callbacks=[best_chpt])

"""

cnn_history = cnn_model.fit_generator(

    train_gen, epochs=epochs, validation_data=val_gen,

    steps_per_epoch=len(train_gen)//batch_size)

# I might get back to checkpointing but for now it seems as a waste of time

probs = cnn_model.predict(test_set)

#preds = np.argmax(probs, axis=1)

classes = train_set.class_names

#classes = [classes[x] for x in preds]

show_sample(test_set)

cnn_model.save_weights('cnn_model_weights.h5')
mini_resnet = build_resnet_model()

"""

fc_history = fc_model.fit(

    train_set, steps_per_epoch=len(train_set)//batch_size, 

    epochs=epochs, validation_data=val_set)

    # callbacks=[best_chpt])

"""

res_history = mini_resnet.fit_generator(

    train_gen, epochs=epochs, validation_data=val_gen,

    steps_per_epoch=len(train_gen)//batch_size)

# I might get back to checkpointing but for now it seems as a waste of time

preds = mini_resnet.predict(test_set)

#preds = np.argmax(probs, axis=1)

classes = train_set.class_names

#classes = [classes[x] for x in preds]

show_sample(test_set)

mini_resnet.save_weights('res_model_weights.h5')