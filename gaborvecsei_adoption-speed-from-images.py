import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from keras.applications import resnet50
from keras import layers, models, callbacks

import matplotlib.pyplot as plt
%matplotlib inline
!ls ../input
!ls ../input/petfinder-adoption-prediction
!ls ../input/keras-pretrained-models/
df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv", header=0)
image_paths = list(Path("../input/petfinder-adoption-prediction/train_images").glob("*.jpg"))
image_ids = [x.stem.split("-")[0] for x in image_paths]
df = df.merge(pd.DataFrame({"PetID":image_ids, "ImagePath":image_paths}), on="PetID", how="outer")
has_images_mask = ~df["ImagePath"].isnull()
print("Number of entries without images: {0}".format((~has_images_mask).sum()))
fig = plt.figure(figsize=(10, 4))

ax0 = fig.add_subplot(1, 2, 1)
ax0.set_title("Has Images")
ax0.hist(df[has_images_mask]["AdoptionSpeed"].values)
ax0.set_xlabel("Adoption Speed")
ax0.set_ylabel("Count")

ax1 = fig.add_subplot(1, 2, 2, sharex=ax0)
ax1.set_title("Doesn't have images")
ax1.hist(df[~has_images_mask]["AdoptionSpeed"].values);
train_df = df.dropna(subset=["ImagePath"])
# This is only for testing
# train_df = train_df.sample(100)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
def read_image(p, image_height, image_width):
    image = cv2.imread(str(p))
    image = cv2.resize(image, (image_width, image_height))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = np.expand_dims(image, 2)
    image = image.astype(np.float32) / 255.0
    return image
images = []

for p in tqdm(train_df["ImagePath"].values):
    image = read_image(p, IMAGE_HEIGHT, IMAGE_WIDTH)
    images.append(image)
images = np.stack(images)
images.shape
adoption_speed = train_df["AdoptionSpeed"].values
ohe = OneHotEncoder()
ohe.fit(adoption_speed.reshape(-1, 1));
adoption_speed_ohe = ohe.transform(adoption_speed.reshape(-1, 1))
adoption_speed_ohe.shape
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
NUM_CLASSES = len(ohe.categories_[0])
base_model = resnet50.ResNet50(include_top=False, weights=None, input_shape=INPUT_SHAPE, pooling="avg")
x = base_model.output
# x = layers.Dense(1024, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(model.get_layer("res2c_branch2c").get_weights()[0].flatten(), bins=50)
axs[1].hist(model.get_layer("res5c_branch2c").get_weights()[0].flatten(), bins=50);
imagenet_weights_path = "../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
model.load_weights(imagenet_weights_path, by_name=True)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(model.get_layer("res2c_branch2c").get_weights()[0].flatten(), bins=50)
axs[1].hist(model.get_layer("res5c_branch2c").get_weights()[0].flatten(), bins=50);
layer_regex = {
    "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
    "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)",
    "5+": r"(res5.*)|(bn5.*)",
    "all": ".*",
}
trainable_layers = layer_regex["3+"]
for layer in model.layers:
    if not layer.weights:
        continue
    
    if re.fullmatch(trainable_layers, layer.name):
        layer.trainable = True
    else:
        layer.trainable = False
print("Number of trainable layers: {0}".format(len([x for x in model.layers if x.trainable])))
class KernelRunTimeCallback(callbacks.Callback):
    def __init__(self, init_timestamp, max_runtime_in_secs):
        super(callbacks.Callback, self).__init__()
        self.init_timestamp = init_timestamp
        self.max_runtime_in_secs = max_runtime_in_secs
    
    def on_batch_end(self, batch, logs=None):
        elapsed_time = time.time() - self.init_timestamp
        if elapsed_time > self.max_runtime_in_secs:
            self.model.stop_training = True
            print("Training stopped due to maximum kernel runtime restriction")
BATCH_SIZE = 32
EPOCHS = 100
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1)
checkpoint_file_path = "model_checkpoint.h5"
model_checkpoint = callbacks.ModelCheckpoint(checkpoint_file_path,
                                             monitor="val_loss",
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=True)
kernel_runtime_callback = KernelRunTimeCallback(time.time(), 110*60)
callbacks = [early_stopping, model_checkpoint, kernel_runtime_callback]
hist = model.fit(images,
                 adoption_speed_ohe,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 validation_split=0.2,
                 shuffle=True,
                 callbacks=callbacks)
plt.plot(hist.history["val_loss"], label="validation loss")
plt.plot(hist.history["loss"], label="train loss")
plt.legend();
plt.plot(hist.history["val_acc"], label="validation accuracy")
plt.plot(hist.history["acc"], label="train accuracy")
plt.legend();
model.load_weights(checkpoint_file_path)
forest = RandomForestClassifier(n_estimators=100)
imageless_df = df[~has_images_mask]
feature_mtx = imageless_df[df.columns[2:17]].values
adoption_speed = imageless_df["AdoptionSpeed"].values
forest.fit(feature_mtx, adoption_speed);
test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv", header=0)
image_test_paths = list(Path("../input/petfinder-adoption-prediction/test_images").glob("*.jpg"))
image_test_ids = [x.stem.split("-")[0] for x in image_test_paths]
tmp_df = pd.DataFrame({"PetID":image_test_ids, "ImagePath":image_test_paths})
test_df = test_df.merge(tmp_df, on="PetID", how="outer")
# Just for testing
# test_df = test_df.sample(100)
print("Number of entries without images: {0}".format(test_df["ImagePath"].isnull().sum()))
pet_ids = test_df["PetID"].values
image_paths = test_df["ImagePath"].values
feature_vector = test_df[test_df.columns[2:17]].values

pred_adoption_speed = []

for pet_id, image_path, feature_vector in tqdm(zip(pet_ids, image_paths, feature_vector)):
    if str(image_path) == "nan":
        pred = forest.predict(np.expand_dims(feature_vector, 0))[0]
        pred_adoption_speed.append(pred)
    else:
        image = read_image(image_path, IMAGE_HEIGHT, IMAGE_WIDTH)
        pred = model.predict(np.expand_dims(image, 0))[0]
        pred = np.argmax(pred)
        pred_adoption_speed.append(pred)
submission_df = pd.DataFrame()
submission_df["PetID"] = pet_ids
submission_df["AdoptionSpeed"] = pred_adoption_speed
submission_df = submission_df.groupby("PetID").mean().reset_index()
# TODO: rounding
submission_df["AdoptionSpeed"] = submission_df["AdoptionSpeed"].astype(int)
submission_df.shape
submission_df.to_csv("submission.csv", index=False)