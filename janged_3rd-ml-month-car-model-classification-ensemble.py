# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def get_model(application, size):

    base_model = application(weights='imagenet', input_shape=(size,size,3), include_top=False)

    #base_model.trainable = False

    model = models.Sequential()

    model.add(base_model)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(196, activation='softmax'))

    model.summary()



    optimizer = optimizers.RMSprop(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])



    return model
model_xception = get_model(Xception, 299)

model_xception.load_weights('./Kaggle/car_class/model/best_model_xception.hdf5')

xception_predict = model_xception.predict_generator(

    generator=test_generator_299,

    steps = get_steps(nb_test_sample, batch_size),

    verbose=1

)
model_DenseNet201 = get_model(DenseNet201, 224)

model_DenseNet201.load_weights('./Kaggle/car_class/model/best_model_DenseNet201.hdf5')

DenseNet201_predict = model_DenseNet201.predict_generator(

    generator=test_generator_224,

    steps = get_steps(nb_test_sample, batch_size),

    verbose=1

)
model_nasnetlarge = get_model(NASNetLarge, 331)

model_nasnetlarge.load_weights('./Kaggle/car_class/model/best_model_MobileNetV2.hdf5')

nasnetlarge_predict = model_nasnetlarge.predict_generator(

    generator=test_generator_331,

    steps = get_steps(nb_test_sample, batch_size),

    verbose=1

)
preds = DenseNet201_predict*0.34 + xception_predict*0.34 + nasnetlarge_predict*0.33



preds_class_indices=np.argmax(preds, axis=1)

preds_labels = (train_generator_299.class_indices)

labels = dict((v,k) for k,v in preds_labels.items())

final_predictions = [labels[k] for k in preds_class_indices]