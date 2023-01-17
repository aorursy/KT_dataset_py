import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pandas as pd
%matplotlib inline
num_classes = 5
image_size = 128
batch_size = 32
epochs = 20
tfk = tf.keras
tfkl = tf.keras.layers
tfka = tf.keras.applications
base_model = tfka.resnet50.ResNet50(weights="imagenet", input_shape=(image_size, image_size, 3), include_top=False)
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False
    
x = base_model.layers[-1].output
x = tfkl.Flatten()(x)
x = tfkl.Dropout(0.2)(x)
x = tfkl.Dense(256, activation="relu")(x)
x = tfkl.Dropout(0.2)(x)
x = tfkl.Dense(512, activation="relu")(x)
x = tfkl.Dense(num_classes, activation="softmax")(x)
model = tfk.Model(inputs=base_model.inputs, outputs=x)
model.summary()
data_path = "../input/ammi-2020-convnets/train/train"
test_path = "../input/ammi-2020-convnets/test/test/0"
submission_file_path = "../input/ammi-2020-convnets/sample_submission_file.csv"
output_path = "../output/kaggle/working/sample_submission_file.csv"
extraimage_path = "../input/ammi-2020-convnets/extraimages/extraimages"


# Transformations for both the training and testing data
train_datagen = ImageDataGenerator(
    preprocessing_function=tfka.resnet50.preprocess_input,
    rotation_range=30, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    validation_split=0.2
)



train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit_generator(train_generator, steps_per_epoch=4527//batch_size+1, validation_data=validation_generator, validation_steps=1129//batch_size+1, epochs=epochs)
model.save("model.h5")
def get_single_batch(base_dir, filenames, image_size):
    output = np.zeros((len(filenames), image_size, image_size, 3))
    for i in range(len(filenames)):
        image = load_img(os.path.join(base_dir, filenames[i]), target_size=(image_size, image_size))
        
        image = img_to_array(image)
        image = tfka.vgg19.preprocess_input(image)
        output[i] = image
    return output

def get_batches(base_dir, filenames, image_size, batch_size=32):
    i = 0
    while i < len(filenames):
        image_files = filenames[i:i+batch_size]
        yield get_single_batch(base_dir, image_files, image_size)
        i+= batch_size
def create_submission(model, directory, sample_submission_file, class2index, batch_size=32):
    
    ss_df = pd.read_csv(sample_submission_file)
    
    index2class = {value:key for key, value in class2index.items()}
    labels = []
    
    filenames = ss_df["Id"].values.tolist()
    for batch in get_batches(directory, filenames, image_size, batch_size):
        prediction = model.predict(batch)
        for p in prediction:
            labels.append(index2class[np.argmax(p)])
    ss_df["Category"] = labels
    return ss_df
train_generator.class_indices
submission = create_submission(model, test_path, submission_file_path, train_generator.class_indices)
submission.head()
submission.to_csv('submission.csv', index=False)
# !kaggle competitions submit -c ammi-2020-convnets -f ../output/kaggle/working/submission.csv -m "tf-starter sumbission"
## Make Submission
%ls ../working

