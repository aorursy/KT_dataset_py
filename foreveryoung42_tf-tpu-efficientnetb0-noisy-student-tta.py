!pip install -U -q efficientnet
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn
from tqdm import tqdm
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("tf version " + tf.__version__)
# there are some issues with dataset - unable to load it to GCS now
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
GCS_DS_PATH
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU", tpu.master())
except ValueError:
    strategy = tf.distribute.get_strategy()
    print("Running on GPU")

print("REPLICAS: ", strategy.num_replicas_in_sync)
BASE_PATH = Path("..") / "input" / "siim-isic-melanoma-classification"
SHAPE = (224, 224, 3)
BATCH_SIZE = 64
train_df = pd.read_csv(BASE_PATH / "train.csv")
test_df = pd.read_csv(BASE_PATH / "test.csv")

train_df["image_path"] = train_df.apply(lambda x: str(
    BASE_PATH / "jpeg" / "train" / f"{x.image_name}.jpg"), axis=1)
test_df["image_path"] = test_df.apply(lambda x: str(
    BASE_PATH / "jpeg" / "test" / f"{x.image_name}.jpg"), axis=1)


train_df.head()
base_data_gen = dict(
    rescale=1/255.,
    horizontal_flip=True,
    vertical_flip=True,
)

train_datagen = ImageDataGenerator(
    **base_data_gen,
    zoom_range=0.1,
    validation_split=0.1,
)
# for TTA
test_datagen = ImageDataGenerator(
    **base_data_gen
)

gen_args = dict(
    dataframe=train_df,
    x_col="image_path",
    y_col="benign_malignant",
    batch_size=BATCH_SIZE,
    seed=42,
    target_size=SHAPE[:2],
    color_mode="grayscale" if SHAPE[-1] == 1 else "rgb",
    class_mode="binary",
    shuffle=True,
)

train_gen = train_datagen.flow_from_dataframe(
    **gen_args,
    subset="training"
)

valid_gen = train_datagen.flow_from_dataframe(
    **gen_args,
    subset="validation"
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col=None,
    batch_size=BATCH_SIZE,
    target_size=SHAPE[:2],
    color_mode="grayscale" if SHAPE[-1] == 1 else "rgb",
    shuffle=False,
    class_mode=None
)
def plot_examples(images_arr):
  fig, axes = plt.subplots(1, 5, figsize=(15,15))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img.reshape(SHAPE), cmap="gray")
    ax.axis("off")
  plt.tight_layout()
  plt.show()


sample_training_images, _ = next(train_gen)
plot_examples(sample_training_images)
def create_model(input_shape):
    base_model = efn.EfficientNetB0(
        weights="noisy-student", include_top=False, input_shape=input_shape
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model
with strategy.scope():
    model = create_model(input_shape=SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
%%time

callbacks = [
    ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=5),
]
steps_per_epoch = len(train_gen.filenames) // BATCH_SIZE
validation_steps = len(valid_gen.filenames) // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    verbose=1,
)
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()
tta = []
tta_steps = 10
test_steps = len(test_gen.filenames) // BATCH_SIZE

for _ in tqdm(range(tta_steps)):
    test_gen.reset()
    preds = model.predict_generator(
        generator=test_gen,
        steps=test_steps
    )
    tta.append(preds)

tta_mean = np.mean(tta, axis=0)
labels = train_gen.class_indices

sub_df =  pd.read_csv(BASE_PATH / "sample_submission.csv")
sub_df.target = tta_mean
sub_df.to_csv("submission.csv", index=False)
sub_df.head()