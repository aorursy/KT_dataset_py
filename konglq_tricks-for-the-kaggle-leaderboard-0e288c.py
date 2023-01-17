import pandas as pd

import os



clip = 0.90

classes = 8



def clip_csv(csv_file, clip, classes):

    # Read the submission file

    df = pd.read_csv(csv_file, index_col=0)



    # Clip the values

    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)

    

    # Normalize the values to 1

    df = df.div(df.sum(axis=1), axis=0)



    # Save the new clipped values

    df.to_csv('clip.csv')

    print(df.head(10))

    

# Of course you are going to use your own submission here

clip_csv('../input/sample_submission_stg1.csv', clip, classes)
import numpy as np

import pandas as pd



def blend_csv(csv_paths):

    if len(csv_paths) < 2:

        print("Blending takes two or more csv files!")

        return

    

    # Read the first file

    df_blend = pd.read_csv(csv_paths[0], index_col=0)

    

    # Loop over all files and add them

    for csv_file in csv_paths[1:]:

        df = pd.read_csv(csv_file, index_col=0)

        df_blend = df_blend.add(df)

        

    # Divide by the number of files

    df_blend = df_blend.div(len(csv_paths))



    # Save the blend file

    df_blend.to_csv('blend.csv')

    print(df_blend.head(10))



# Obviously replace this with two or more of your files

blend_csv(['../input/sample_submission_stg1.csv', '../input/sample_submission_stg1.csv'])
# Create a MixIterator object

# This class is a simple method to create batches from several other batch generators

class MixIterator(object):

    def __init__(self, iters):

        self.iters = iters

        self.multi = type(iters) is list

        if self.multi:

            self.N = sum([it[0].N for it in self.iters])

        else:

            self.N = sum([it.N for it in self.iters])



    def reset(self):

        for it in self.iters: it.reset()



    def __iter__(self):

        return self



    def next(self, *args, **kwargs):

        if self.multi:

            nexts = [[next(it) for it in o] for o in self.iters]

            n0 = np.concatenate([n[0] for n in nexts])

            n1 = np.concatenate([n[1] for n in nexts])

            return (n0, n1)

        else:

            nexts = [next(it) for it in self.iters]

            n0 = np.concatenate([n[0] for n in nexts])

            n1 = np.concatenate([n[1] for n in nexts])

        return (n0, n1)
# Example usage in Keras

# [replace by your own code]

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

validation_data=(x_test, y_test))



batch_size = 8

# [/replace by your own code]



predictions = model.predict(x_test, batch_size=batch_size)



gen = ImageDataGenerator()



train_batches = gen.flow(x_train, y_train, batch_size=44)

val_batches = gen.flow(x_val, y_val, batch_size=4)

test_batches = gen.flow(x_test, predictions, batch_size=16)



mi = MixIterator([train_batches, test_batches, val_batches])

model.fit_generator(mi, mi.N, nb_epoch=8, validation_data=(x_val, y_val))
