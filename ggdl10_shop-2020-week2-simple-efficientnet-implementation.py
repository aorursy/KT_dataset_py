import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from efficientnet.keras import EfficientNetB4
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
# index csv
train_dex = pd.read_csv("train.csv")
test_dex = pd.read_csv("test.csv")

#paths
train_dir = os.path.join(os.path.abspath(''), 'copyDataset', 'train', 'train')
valid_dir = os.path.join(os.path.abspath(''), 'copyDataset', 'train', 'validation')
test_dir = os.path.join(os.path.abspath(''), 'dataset', 'test', 'test')
#model attributes
batch_size = 4
epochs = 20
IMG_HEIGHT = 300
IMG_WIDTH = 300
#dataset length
total_train = 0
for i in os.listdir(train_dir):
    total_train += len(os.listdir(os.path.join(train_dir, i)))
total_val = 0
for i in os.listdir(valid_dir):
    total_val += len(os.listdir(os.path.join(valid_dir, i)))
#data augmentation for training to prevent overfitting
from tensorflow.keras.applications.efficientnet import preprocess_input
train_image_generator = ImageDataGenerator(rescale=1./255, 
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           rotation_range=40,
                                           preprocessing_function=preprocess_input
                                           )

test_image_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
#flow from directory
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

valid_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=valid_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
#generating a callback to save training progress
checkpoint_path = "ic_chkpt/eff_netb3.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=10000)
#transfer learning with imagenet
eff_net = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
#freeze bottom layers
eff_net.trainable = False
#train these layers
x = eff_net.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(42, activation="softmax")(x)
model = tf.keras.Model(inputs = eff_net.input, outputs = predictions)

model.compile(tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
# Train model
history = model.fit(train_data_gen,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=valid_data_gen,
              callbacks=[cp_callback],
              initial_epoch=11
              )
#PREDICTIONS
#use validation image generator since its the same as test
test_generator = test_image_generator.flow_from_directory(directory=test_dir,
                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                        batch_size=1,
                                        class_mode=None,  # this means our generator will only yield batches of data, no labels
                                        shuffle=False)
test_step_size = test_generator.n//1
test_generator.reset()
pred = model.predict(test_generator,
                            steps = test_step_size,
                            verbose=1)
#padding zeroes for single digit class labels ("1" -> "01")
def zero_inserter(x):
    if len(str(x)) < 2:
        return ("0" + str(x))
    else:
        return (str(x))
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

result_df=pd.DataFrame({"filename":filenames,
                      "category":predictions})

result_df['filename'] = result_df['filename'].apply(lambda x: x.split("\\")[1])

actual_test = pd.read_csv("test.csv")
actual_test.head()
actual_test_list = list(actual_test['filename'])

result_df["category"] = result_df.category.apply(zero_inserter)

result_df[result_df['filename'].isin(actual_test_list)].to_csv("results_effnetb4.csv", index=False)