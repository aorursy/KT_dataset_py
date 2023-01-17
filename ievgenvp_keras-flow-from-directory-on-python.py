import pandas as pd

import numpy as np



'''

labels_df = pd.read_csv('train_labels.csv')

print (labels_df['invasive'].value_counts())

classes = [0,1]



for i in classes:

    df_class = labels_df.loc[labels_df['invasive'] == i]

    df_class = df_class['name'].astype(str)+'.jpg'

    n = lambda x: 100 if i == 1 else 37

    validation_pack = np.random.choice(df_class.values, n(i), replace = False)

    np.savetxt('class_{0}_val.txt'.format(i), validation_pack, fmt = '%s')



    training_pack = np.setdiff1d(df_class.values, validation_pack)

    np.savetxt('class_{0}_tr.txt'.format(i), training_pack, fmt = '%s')

'''
import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_auc_score



# place to put some parameters

batch_size = 20

num_val_samples = 137

steps = 2295*1.4/batch_size
'''

# data generator for training set

train_datagen = ImageDataGenerator(

    rescale = 1./255,

    shear_range = 0.2, # random application of shearing

    zoom_range = 0.2,

    horizontal_flip = True) # randomly flipping half of the images horizontally



# data generator for test set

test_datagen = ImageDataGenerator(rescale = 1./255)

'''
'''

# generator for reading train data from folder

train_generator = train_datagen.flow_from_directory(

    'data/train_tr',

    target_size = (256, 256),

    color_mode = 'rgb',

    batch_size = batch_size,

    class_mode = 'binary')



# generator for reading validation data from folder

validation_generator = test_datagen.flow_from_directory(

    'data/train_val',

    target_size = (256, 256),

    color_mode = 'rgb',

    batch_size = batch_size,

    class_mode = 'binary')



# generator for reading test data from folder

test_generator = test_datagen.flow_from_directory(

    'data/test',

    target_size = (256, 256),

    color_mode = 'rgb',

    batch_size = 1,

    class_mode = 'binary',

    shuffle = False)

'''
'''

# neural network model

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(64,activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy',

              optimizer = 'rmsprop',

              metrics = ['accuracy'])



model.fit_generator(train_generator,

                    steps_per_epoch = steps,

                    epochs = 10,

                    validation_data = validation_generator,

                    validation_steps = num_val_samples/batch_size)



#model.save_weights('nn_weights.h5')

#model.load_weights('nn_weights.h5')

'''
'''

# AUC for prediction on validation sample

X_val_sample, val_labels = next(validation_generator)

val_pred = model.predict_proba(X_val_sample)

val_pred = np.reshape(val_pred, val_labels.shape)

val_score_auc = roc_auc_score(val_labels, val_pred)

print ("AUC validation score")

print (val_score_auc)

print ('\n')

'''
'''

# test predictions with generator

test_files_names = test_generator.filenames

predictions = model.predict_generator(test_generator,

                                      steps = 1531)

predictions_df = pd.DataFrame(predictions, columns = ['invasive'])

predictions_df.insert(0, "name", test_files_names)

predictions_df['name'] = predictions_df['name'].map(lambda x: x.lstrip('test\\').rstrip('.jpg'))

predictions_df['name'] = pd.to_numeric(predictions_df['name'], errors = 'coerce')

predictions_df.sort_values('name', inplace = True)

predictions_df.to_csv('predictions_df.csv', index = False)

'''