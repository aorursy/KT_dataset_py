import pandas as pd

import matplotlib.pyplot as plt

from tensorflow import keras

from tensorflow.keras import layers

import glob
df = pd.read_csv("../input/aerial-cactus-identification/train.csv")

df.has_cactus = df.has_cactus.astype(str) # Convert the column to string to be used by keras generator later on

df.head()
all_paths = glob.glob("../input/aerial-cactus-identification/train/train/*.jpg")

plt.figure(figsize=(18, 4))



for i in range(1, 15):

    ax = plt.subplot(1, 15, i)

    ax.imshow(plt.imread(all_paths[i]))

    ax.grid(False)

    ax.axis('off')



plt.show()
vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(150, 150, 3))



for layer in vgg.layers:

    layer.trainable = False
x = vgg.output

x = layers.Flatten()(x)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dense(512, activation='relu')(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)



model = keras.Model(inputs=vgg.inputs, outputs=outputs)

model.summary()
datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



train_generator = datagen.flow_from_dataframe(df[:15000], 

                                              x_col="id", y_col="has_cactus", 

                                              directory="../input/aerial-cactus-identification/train/train/", 

                                              class_mode="binary", 

                                              batch_size=128, target_size=(150, 150))



valid_generator = datagen.flow_from_dataframe(df[15000:], 

                                              x_col="id", y_col="has_cactus", 

                                              directory="../input/aerial-cactus-identification/train/train/", 

                                              class_mode="binary", 

                                              batch_size=128, target_size=(150, 150))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit_generator(train_generator, validation_data=valid_generator, epochs=10)
test_generator = datagen.flow_from_directory("../input/aerial-cactus-identification/test", classes=None, target_size=(150, 150))
def load_and_scale_img(path):

    img = keras.preprocessing.image.load_img(path, target_size=(150, 150))

    img = keras.preprocessing.image.img_to_array(img)

    img /= 255.

    return img
test_paths = glob.glob("../input/aerial-cactus-identification/test/test/*.jpg")

print("Found %d images." % len(test_paths))
test_set = [load_and_scale_img(path) for path in test_paths]
predictions = model.predict([test_set])
submission_df = pd.DataFrame(data={

    "id": [path.split("/")[-1] for path in test_paths], 

    "has_cactus": predictions[:, 0].astype(int)

})

submission_df.head()
submission_df.to_csv('sample_submission.csv', index=False)