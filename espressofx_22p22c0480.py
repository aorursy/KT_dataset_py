import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
!nvidia-smi
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_DIR="../input/super-ai-image-classification"
OUTPUT_DIR="../working"
train_csv = pd.read_csv(DATA_DIR+'/train/train/train.csv')
filenames = [DATA_DIR+'/train/train/images/' + fname for fname in train_csv['id'].tolist()]
labels = train_csv['category'].tolist()
!rm -rf ../working/temp_train
os.mkdir(OUTPUT_DIR+"/temp_train/",mode = 0o777,)
for label in [0,1]:
    try:
        os.mkdir(OUTPUT_DIR+"/temp_train/"+str(label),mode = 0o777,)
    except OSError as error: 
        print(error) 
for idx,row in train_csv.iterrows():
    label=row["category"]
    shutil.copy(DATA_DIR+"/train/train/images/"+row["id"],OUTPUT_DIR+"/temp_train/"+str(label))
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
IMAGE_SIZE = 96
BATCH_SIZE = 32
image_generator = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    #rotation_range=45,
    #width_shift_range=.15,
    #height_shift_range=.15,
    #horizontal_flip=True,
    #zoom_range=0.5,
    #brightness_range=(0.1,0.9)
)
data_generator = image_generator.flow_from_directory(
    OUTPUT_DIR+"/temp_train/", 
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="binary"
)
train_generator = image_generator.flow_from_directory(
    OUTPUT_DIR+"/temp_train/", 
    subset='training',
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="binary"
)
val_generator = image_generator.flow_from_directory(
    OUTPUT_DIR+"/temp_train/", 
    subset='validation',
    shuffle=True,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="binary"
)
augmented_images = [val_generator[0][0][0] for i in range(5)]
plot_images(augmented_images)
from collections import Counter

counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
print(class_weights)
{class_id : num_images for class_id, num_images in counter.items()}   
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
# Pre-trained model with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet',
    #pooling='max'
)
# Freeze the pre-trained model weights
base_model.trainable = False
'''
for layer in base_model.layers:
    if 'conv5' in layer.name:
        layer.trainable=True
    else:
        layer.trainable=False
'''
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
flatten = tf.keras.layers.Flatten()

dense_512 = tf.keras.layers.Dense(512, activation='sigmoid')
dense_256 = tf.keras.layers.Dense(256,activation='sigmoid')
dense_128 = tf.keras.layers.Dense(128,activation='sigmoid')
dense_64 = tf.keras.layers.Dense(64,activation='sigmoid')
dense_32 = tf.keras.layers.Dense(32,activation='sigmoid')
dense_16 = tf.keras.layers.Dense(16,activation='sigmoid')
dense_8 = tf.keras.layers.Dense(8,activation='sigmoid')

dropout_1 = tf.keras.layers.Dropout(0.5)
dropout_2 = tf.keras.layers.Dropout(0.5)
dropout_3 = tf.keras.layers.Dropout(0.5)
dropout_4 = tf.keras.layers.Dropout(0.5)
dropout_5 = tf.keras.layers.Dropout(0.5)

batchnorm_1 = tf.keras.layers.BatchNormalization()
batchnorm_2 = tf.keras.layers.BatchNormalization()
batchnorm_3 = tf.keras.layers.BatchNormalization()
batchnorm_4 = tf.keras.layers.BatchNormalization()
batchnorm_5 = tf.keras.layers.BatchNormalization()

prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
#prediction_layer = tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.L2(0.01), activation='sigmoid')
"""
    flatten,
    batchnorm_1,
    dense_1,
    dropout_1,
    batchnorm_2,
    dense_2,
    dropout_2,
    batchnorm_3,
"""
# Layer classification head with feature detector
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    dense_256,
    dropout_1,
    dense_8,
    dropout_2,
    dense_128,
    dropout_3,
    prediction_layer
])
learning_rate = 0.0001
# Compile the model
#tf.keras.optimizers.Adam(lr=learning_rate)
#tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
              loss='binary_crossentropy',
              metrics=['acc']
)
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# add a checkpoint to save the lowest validation loss
filepath = 'super_ai_model_20201008.h5'

callbacks   = [
      EarlyStopping(monitor='val_loss', patience=20, mode='min'),
      ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                             save_best_only=True, save_weights_only=False, \
                             mode='auto', save_frequency=1)
]
train_generator.samples, val_generator.samples
num_epochs = 100
history=model.fit_generator(
          train_generator,
          callbacks=[callbacks],
          epochs=num_epochs,
          validation_data=val_generator,
          steps_per_epoch=train_generator.samples // BATCH_SIZE,
          validation_steps=val_generator.samples // BATCH_SIZE,
          class_weight=class_weights,
          verbose=1)
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
super_ai_model = tf.keras.models.load_model(filepath)
TEST_DIR=DATA_DIR+'/val/val'
TEST_DIR
test_generator = image_generator.flow_from_directory(
    TEST_DIR, 
    shuffle=False,
    batch_size=1,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="binary"
)
rows_list = []
for file_img in test_generator.filenames:
    #print(file_img)
    image = tf.keras.preprocessing.image.load_img(TEST_DIR+"/"+file_img, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    pred=super_ai_model.predict_classes(input_arr)
    pred_proba=super_ai_model.predict(input_arr)
    print(os.path.basename(file_img), pred_proba[0][0], pred[0][0])
    rows_list.append({"id":os.path.basename(file_img),"category":pred[0][0]})
    
    #ai_super_model.predict_classes(input_arr)
    
val_1=pd.DataFrame(rows_list,columns=["id","category"])
val_1=val_1.set_index(["id"])
val_1.head()
len(val_1[val_1["category"]==0]), len(val_1[val_1["category"]==1])
val_1[val_1["category"]==0].head(10)
# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(val_1, filename = "prediction.csv")
!tar -zcvf outputname.tar.gz /kaggle/working