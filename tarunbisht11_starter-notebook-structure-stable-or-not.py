import numpy as np

import pandas as pd

import tensorflow as tf

import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith(".csv"):

            print(os.path.join(dirname, filename))
base_path="/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data"

submission_file="/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data/submit.csv"

test_path="/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data/test.csv"

train_path="/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data/train.csv"
train_data=pd.read_csv(train_path)

test_data=pd.read_csv(test_path)
train_data.head()
test_data.head()
submission=pd.read_csv(submission_file)
submission.head()
trainX,evalX,trainY,evalY = train_test_split(train_data.iloc[:,0].values,

                                             train_data.iloc[:,1].values,

                                             random_state=11,

                                             test_size=0.2

                                            )
testX = test_data.iloc[:,0].values
num_train_images = len(trainX)

num_eval_images=len(evalX)
print("Number of train images: ",num_train_images)

print("Number of eval images: ",num_eval_images)

print("Number of test images: ",len(testX))
EPOCHS=5

BATCH_SIZE=32

IMAGE_DIM=(256,256)
def get_path_of_image(image_id):

    return os.path.join(base_path,f"{image_id}.png")
def check_and_remove_defected_images(ids,labels):

    defected = []

    for i,img_id in enumerate(ids):

        try:

            image = tf.io.read_file(get_path_of_image(img_id))

            image = tf.image.decode_png(image,channels=3)

        except:

            defected.append(img_id)

            ids = np.delete(ids,i)

            labels = np.delete(labels,i)

    return defected,ids,labels
def load_tf_image(image_path,dim):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(image,channels=3)

    image = tf.image.resize(image,dim)

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = image/255.0

    return image
def generate_tf_dataset(X,Y,image_size):

    X = [get_path_of_image(str(x)) for x in X]

    datasetX = tf.data.Dataset.from_tensor_slices(X).map(

            lambda path: load_tf_image(path,image_size),

            num_parallel_calls=tf.data.experimental.AUTOTUNE

    )

    datasetY = tf.data.Dataset.from_tensor_slices(Y)

    dataset = tf.data.Dataset.zip((datasetX,datasetY))

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
def plot_images_grid(data,num_rows=1,labels=None,class_names=None):

    images, labels = data

    n=len(images)

    if n > 1:

        num_cols=np.ceil(n/num_rows)

        fig,axes=plt.subplots(ncols=int(num_cols),nrows=int(num_rows))

        axes=axes.flatten()

        fig.set_size_inches((15,15))

        for i,image in enumerate(images):

            axes[i].imshow(image.numpy())

            label = labels[i].numpy()

            axes[i].set_title(class_names[label])
defected,trainX,trainY = check_and_remove_defected_images(trainX,trainY)

print("Train Elements Defected: ",defected)
defected,trainX,trainY = check_and_remove_defected_images(evalX,evalY)

print("Eval Elements Defected: ",defected)
train_dataset=generate_tf_dataset(trainX,trainY,IMAGE_DIM)

print(train_dataset.element_spec)
eval_dataset=generate_tf_dataset(evalX,evalY,IMAGE_DIM)

print(eval_dataset.element_spec)
class_names=["not stable","stable"]
plot_images_grid(next(iter(train_dataset.take(1))),class_names=class_names,num_rows=4)
plot_images_grid(next(iter(eval_dataset.take(1))),class_names=class_names,num_rows=4)
densenet= tf.keras.applications.DenseNet121(

                include_top=False, weights='imagenet',input_shape=(*IMAGE_DIM,3)

            )

densenet.summary()
densenet.trainable = False
model = tf.keras.Sequential([

    densenet,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256,activation="relu"),

    tf.keras.layers.Dense(1,activation="sigmoid")

])
model.summary()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
checkpoint_path="best_checkpoint"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,monitor="val_accuracy",

                                                    save_best_only=True,mode="max",

                                                    save_weights_only=True,

                                                    verbose=1)

early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=10,

                                            mode="max")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode="max",

                                                 factor=0.2,patience=5, 

                                                 min_lr=0.001)
callbacks=[model_checkpoint,early_stop,reduce_lr]
history=model.fit(train_dataset,

                  epochs=EPOCHS,

                  steps_per_epoch=num_train_images//BATCH_SIZE,

                  validation_data=eval_dataset,

                  validation_steps=num_eval_images//BATCH_SIZE,

                  callbacks=callbacks)
if os.path.isfile(checkpoint_path):

    model.load_weights(checkpoint_path)
test_images_path = [get_path_of_image(str(x)) for x in testX]

test_dataset = tf.data.Dataset.from_tensor_slices(test_images_path).map(

        lambda path: load_tf_image(path,IMAGE_DIM),

        num_parallel_calls=tf.data.experimental.AUTOTUNE

)

test_dataset=test_dataset.batch(BATCH_SIZE)

test_dataset=test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
predictions = model.predict(test_dataset,verbose=1)
predictions= np.squeeze(predictions,axis=1)
predictions.shape
submission.stable = predictions

submission.to_csv('submission.csv', index=False)

submission.head()