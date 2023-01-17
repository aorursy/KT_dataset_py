import os 
import path
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tensorflow.keras import layers


warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
sns.set_style("dark")
data_dir = path.Path('../input/jovian-pytorch-z2g/Human protein atlas')
test_dir = data_dir/'test'
train_dir = data_dir/'train'
train_csv_path = data_dir/'train.csv'
sub_csv_path = '../input/jovian-pytorch-z2g/submission.csv'
df = pd.read_csv(train_csv_path)
df.head()
labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}

indexes = {str(v):k for k,v in labels.items()}
label_freq = df['Label'].apply(lambda x: str(x).split(' ')).explode().value_counts().sort_values(ascending=False)

# Bar plot
plt.figure(figsize=(12,10))
sns.barplot(y=[labels[i] for i in list(map(int, label_freq.index.values))], 
            x=label_freq)
plt.title("Label frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
df['Label'] = df['Label'].apply(lambda x: x.split(' '))
df.head()
X_train, X_val, y_train, y_val = train_test_split(df['Image'], df['Label'], test_size=0.2, random_state=44)
print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))
X_train = [os.path.join(train_dir, str(f)+'.png') for f in X_train]
X_val = [os.path.join(train_dir, str(f)+'.png') for f in X_val]
X_train[:3]
y_train = list(y_train)
y_val = list(y_val)
y_train[:3]
nobs = 8 # Maximum number of images to display
ncols = 4 # Number of columns in display
nrows = nobs//ncols # Number of rows in display

plt.figure(figsize=(12,4*nrows))
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    plt.imshow(Image.open(X_train[i]))
    plt.title(y_train[i], size=10)
    plt.axis('off')
# img = tf.io.read_file(X_train[1])
# img = tf.image.decode_png(img)
# plt.imshow(img)
# std_img = tf.image.per_image_standardization(img)
# plt.imshow(std_img)
# img1 = tf.image.convert_image_dtype(std_img, tf.float32)
# plt.imshow(img1, cmap='hsv')
# img1
# type(img1)
mlb = MultiLabelBinarizer()
mlb.fit(y_train)
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)
for i in range(3):
    print(X_train[i], y_train_bin[i])
IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model
def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_png(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
#     image_normalized = image_resized/255.
    return image_resized, label
BATCH_SIZE = 32 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 128 # Shuffle the training data by a chunck of 1024 observations
def augment(image,label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
#     image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    image = tf.image.rot90(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta=0.5)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
    return image,label
def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    daraset = dataset.map(augment, num_parallel_calls = AUTOTUNE)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
#         dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)
for f, l in train_ds.take(1):
    print("Shape of features array:", f.numpy().shape)
    print("Shape of labels array:", l.numpy().shape)
plt.imshow(f[6])
# feature_extractor_url = "https://tfhub.dev/google/bit/m-r50x1/1"
# feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
#                                          input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
base_model = tf.keras.applications.DenseNet121(input_shape=[224,224,3], include_top=False,
                                              weights='imagenet')
len(base_model.layers)
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = False
    else:
        layer.trainable = True
base_model.summary()
model = tf.keras.Sequential([
    base_model,
    layers.GlobalMaxPooling2D(),
    layers.Dense(1024, activation='relu', name='hidden_layer',
                 kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu', name='hidden_layer2',
                kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
#     layers.Dense(256, activation='relu', name='hidden_layer3'),
#     layers.Dropout(0.3),
#     layers.Dense(128, activation='relu', name='hidden_layer4'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='sigmoid', name='output')
])

model.summary()
for batch in train_ds:
    print(model.predict(batch)[:1])
    break
LR = 1e-3 # Keep it small when transfer learning
EPOCHS = 20
import tensorflow_addons as tfa
fbeta=tfa.metrics.FBetaScore(num_classes=10, average="micro", threshold = 0.5)
from tensorflow.keras.callbacks import Callback
class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())
        
    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss= 'binary_crossentropy',
  metrics=[fbeta])
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.000001)
checkpointer = tf.keras.callbacks.ModelCheckpoint("classfication.h5"
                                        ,monitor='val_loss'
                                        ,verbose=1
                                        ,save_best_only=True
                                        ,save_weights_only=True)
csvlogger = tf.keras.callbacks.CSVLogger('log.csv')
lr = 5e-3
steps = np.ceil(len(X_train) / 32) * 20
lr_schedule = OneCycleScheduler(lr, steps)
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    steps_per_epoch=481,
                    validation_data=val_ds, 
                    validation_steps=120,
                    callbacks=[lr_schedule, checkpointer, csvlogger]
                   )
final_model = model.load_weights("classfication.h5")
def plot_history(training):
        """
        Plot training history
        """
        ## Trained model analysis and evaluation
        f, ax = plt.subplots(1,2, figsize=(12,3))
        ax[0].plot(training.history['loss'], label="Loss")
        ax[0].plot(training.history['val_loss'], label="Validation loss")
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Accuracy
        ax[1].plot(training.history['fbeta_score'], label="F_score")
        ax[1].plot(training.history['val_fbeta_score'], label="Val F_score")
        ax[1].set_title('F_score')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('F_score')
        ax[1].legend()
        plt.tight_layout()
        plt.show()
        
plot_history(history)