import os

import cv2

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import json

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras import backend as K 
train_df = pd.read_csv("../input/imet-2019-fgvc6/train.csv")

train_df["attribute_ids"]=train_df["attribute_ids"].apply(lambda x:x.split(" "))

train_df["id"]=train_df["id"].apply(lambda x:x+".png")

train_df.head()
label_df = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")

print(label_df.shape)

label_df.head()
# Example of images with tags



i = 1

plt.figure(figsize=[30,30])

for img_name in os.listdir("../input/imet-2019-fgvc6/train/")[5:10]:   

    img = cv2.imread("../input/imet-2019-fgvc6/train/%s" % img_name)[...,[2, 1, 0]]

    plt.subplot(5, 1, i)

    plt.imshow(img)

    ids = train_df[train_df["id"] == img_name]["attribute_ids"]

    title_val = []

    for tag_id in ids.values[0]:

        att_name = label_df[label_df['attribute_id'].astype(str) == tag_id]['attribute_name'].values[0]

        title_val.append(att_name)

    plt.title(title_val)

    i += 1

    

plt.show()
nb_classes = 1103

batch_size = 300

img_size = 80

nb_epochs = 30
lbls = list(map(str, range(nb_classes)))
%%time



train_datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.25,

    horizontal_flip = True,    

    zoom_range = 0.3,

    width_shift_range = 0.3,

    height_shift_range=0.3

    )



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/imet-2019-fgvc6/train",

    x_col="id",

    y_col="attribute_ids",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    classes=lbls,

    target_size=(img_size,img_size),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/imet-2019-fgvc6/train",

    x_col="id",

    y_col="attribute_ids",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",    

    classes=lbls,

    target_size=(img_size,img_size),

    subset='validation')
# Loss



gamma = 2.0

epsilon = K.epsilon()

def focal_loss(y_true, y_pred):

    pt = y_pred * y_true + (1-y_pred) * (1-y_true)

    pt = K.clip(pt, epsilon, 1-epsilon)

    CE = -K.log(pt)

    FL = K.pow(1-pt, gamma) * CE

    loss = K.sum(FL, axis=1)

    return loss
# Metric



def f2_score(y_true, y_pred):

    beta = 2

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)

    

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    

    return K.mean(((1+beta**2)*precision*recall) / ((beta**2)*precision+recall+K.epsilon()))
model = applications.Xception(weights=None, 

                          include_top=False, 

                          input_shape=(img_size, img_size, 3))

model.load_weights('../input/xception-weight/xception_weights_tf_dim_ordering_tf_kernels_notop (1).h5')
for layer in model.layers[:-5]:

    layer.trainable = False
# Freeze some layers

# for layer in model.layers[:-4]:

#     layer.trainable = False
#Adding custom layers 

x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(nb_classes, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)



model_final.compile(optimizers.rmsprop(lr=0.001, decay=1e-6),loss=focal_loss,metrics=[f2_score])
# model_final.summary()
# Callbacks



checkpoint = ModelCheckpoint("model_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
%%time

history = model_final.fit_generator(generator=train_generator,                   

                                    steps_per_epoch=500,

                                    validation_data=valid_generator,                    

                                    validation_steps=200,

                                    epochs=nb_epochs,

                                    callbacks = [checkpoint, early],

                                    max_queue_size=16,

                                    workers=2,

                                    use_multiprocessing=True,

                                    verbose=1)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['f2_score', 'val_f2_score']].plot()
sam_sub_df = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

sam_sub_df["id"]=sam_sub_df["id"].apply(lambda x:x+".png")

print(sam_sub_df.shape)

sam_sub_df.head()
%%time

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  

        dataframe=sam_sub_df,

        directory = "../input/imet-2019-fgvc6/test",    

        x_col="id",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
%%time

test_generator.reset()

predict = model_final.predict_generator(test_generator, steps = len(test_generator.filenames))
len(predict)
%%time

import operator

predicted_class_indices_3=[]

for i in range(len(predict)):         

    d = {}

    for index, value in enumerate(predict[i]):               

        if value > 0.03:            

            d[index] = value 

    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

    

    # Take only first 10 items

    predicted_class_indices_3.append([i[0] for i in sorted_d[:10]])
%%time

predictions_3=[]



for i in range(len(predicted_class_indices_3)):

    labels = (train_generator.class_indices)

    labels = dict((v,k) for k,v in labels.items())

    predictions = [labels[k] for k in predicted_class_indices_3[i]]

    predictions_3.append(predictions)
predict_3 = []

for i in range(len(predictions_3)):

    str3 = " ".join(predictions_3[i])

    predict_3.append(str3)
filenames=test_generator.filenames

results=pd.DataFrame({"id":filenames,

                      "attribute_ids":predict_3})

results['id'] = results['id'].map(lambda x: str(x)[:-4])

results.to_csv("submission.csv",index=False)
results.head()