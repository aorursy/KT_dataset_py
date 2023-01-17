import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns

!pip install efficientnet
import efficientnet.tfkeras as efn
# Dataset parameters:
INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_CSV_PATH = os.path.join(DATASET_DIR, 'train.csv')

SUBMISSION_PATH = os.path.join(DATASET_DIR, 'sample_submission.csv')

TRAIN_DF = pd.read_csv(TRAIN_CSV_PATH)
print(f'TRAIN SIZE : {len(TRAIN_DF)}')
TRAIN_DF['landmark_id'].value_counts()
TRAIN_DF.head(10)
train_df2 = pd.DataFrame(TRAIN_DF['landmark_id'].value_counts())
train_df2.reset_index(inplace=True)
train_df2.columns = ['landmark_id','count']
train_df2
import matplotlib as mpl
import matplotlib.pylab as plt

plt.figure(figsize = (14, 10))
plt.title('Top 20 landmarks')
g = sns.barplot(x="landmark_id", y="count", data=train_df2[:20], palette="pastel")


plt.show()
TEST_DF = pd.read_csv(SUBMISSION_PATH)
print(f'TEST SIZE : {len(TEST_DF)}')
labels = []
data = []
for i in range(TRAIN_DF.shape[0]):
    data.append(TRAIN_IMAGE_DIR + '/' + TRAIN_DF['id'].iloc[i][0] + '/' + TRAIN_DF['id'].iloc[i][1] + '/' + TRAIN_DF['id'].iloc[i][2] + '/' + TRAIN_DF['id'].iloc[i]+'.jpg')
    labels.append(TRAIN_DF['landmark_id'].iloc[i])

df = pd.DataFrame(data)
df.columns = ['images']
df['target'] = labels 
    

test_data = []
for i in range(TEST_DF.shape[0]):
    test_data.append(TEST_IMAGE_DIR + '/' + TEST_DF['id'].iloc[i][0] + '/' + TEST_DF['id'].iloc[i][1] + '/' + TEST_DF['id'].iloc[i][2] + '/' + TEST_DF['id'].iloc[i]+'.jpg')

df_test = pd.DataFrame(test_data)
df_test.columns = ['images']    

X_train, X_val, y_train, y_val = train_test_split(df['images'], df['target'], test_size=0.1, random_state=1234)

train=pd.DataFrame(X_train)
train.columns=['images']
train['target']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['target']=y_val
df2 = df[df['target']==66]
df2
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(70,70))
rows = 1
cols = len(df2)


for i in range(len(df2)):

    img1 = cv2.imread(df2['images'].iloc[i])

    ax1 = fig.add_subplot(rows, len(df2), i+1)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title(df2['target'].iloc[i])
    ax1.axis("off")

 
plt.show()
print(f'size : {len(df2)}')
def create_model(num_classes=None, input_size=224):
    model = efn.EfficientNetB3(weights='noisy-student', include_top=False, input_shape=(input_size, input_size, 3))
    x = GlobalAveragePooling2D()(model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(model.input, output)

    return model


def lr_scheduler(epoch, lr):
    if epoch == 3 or epoch == 8:
        return lr * 0.1
    else:
        return lr
train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=[1.0, 1.2],
                                   brightness_range=[0.8, 1.1],
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                  )
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='target',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='raw')

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='target',
    target_size=(224, 224),
    shuffle=False,
    batch_size=BATCH_SIZE,
    class_mode='raw')
model = create_model(8226, 224)
#model.summary()

opt = optimizers.Adam(lr=1e-4)
model.compile(
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    optimizer=opt)

nb_train_steps = train.shape[0]//BATCH_SIZE
nb_val_steps = validation.shape[0]//BATCH_SIZE

learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[learning_rate_callback],
    validation_steps=nb_val_steps)
for path in tqdm(df_test['images']):
    img= cv2.imread(str(path))
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img=np.reshape(img,(1,224,224,3))
    prediction=model.predict(img)
    target.append(prediction[0][0])

submission['target']=target
submission.to_csv('submission.csv', index=False)