!pip install wandb -q
import pandas as pd
import os 
import json
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.densenet import preprocess_input
from datetime import datetime as dt

print("TF Version:", tf.__version__)

import wandb
from wandb.keras import WandbCallback

# Initilize a new wandb run
wandb.init(project="Plant_Pathology_2020", name='exp6', tags=['DenseNet201','finetune'])

train_csv = '/kaggle/input/plant-pathology-2020-fgvc7/train.csv'
test_csv = '/kaggle/input/plant-pathology-2020-fgvc7/test.csv'

DATA_PATH = '/kaggle/input/plant-pathology-2020-fgvc7/images/'
SAVE_PATH = '/kaggle/working/'

MODEL_NAME = 'Plant_Pathology_2020'

# Default values for hyper-parameters
config = wandb.config # Config is a variable that holds and saves hyperparameters and inputs

config.PRETRAINED_MODEL = 'densenet201'

config.IMG_HEIGHT = 224
config.IMG_WIDTH = 224
config.BATCH_SIZE = 64
config.lr = 0.001
config.EPOCHS = 100
config.LOSS = 'binary_crossentropy'
# Train data
img_df =  pd.read_csv(train_csv)
img_df['image_id'] = img_df.image_id.apply(lambda x: x+'.jpg')

TRAIN_df, VAL_df = train_test_split(img_df, test_size=0.2, random_state=42)
TRAIN_df = TRAIN_df.reset_index(drop=True)
print('Train Set:')
print(TRAIN_df)
VAL_df = VAL_df.reset_index(drop=True)
print('')
print('Val Set')
print(VAL_df)
def show_batch(df, source_dir):
    df = shuffle(df)
    df.reset_index(drop=True, inplace=True)
    plt.figure(figsize=(12, 12))
    for n in range(9):
        ax = plt.subplot(3,3,n+1)
        filename = df.iloc[n, 0]
        img = load_img(source_dir+ filename, target_size=(224, 224))
        labels = list(df.iloc[n,:][df.iloc[n,:]==1].to_dict().keys())
        plt.imshow(img)
        plt.xlabel(labels)
        plt.axis('on')
show_batch(TRAIN_df, DATA_PATH)
def value_count(df):

    index=[]
    counts=[]
    for i in list(df.columns[1:]):
        index.append(i)
        counts.append(df[i].value_counts().to_dict())
    index_re = [i for i in index]

    #key_map = dict(zip(index, index_re))
    df = pd.DataFrame(counts, index=index_re)
    #df = pd.DataFrame(counts)
    original_df = df.copy()
    original_df['Disease_Type']=index
    original_df = original_df.rename(columns={0:"Negative", 1:"Positive"})
    original_df = original_df.sort_values(by=['Positive'], ascending=False)
    original_df = original_df[['Disease_Type','Negative', 'Positive']]
    
    df = df.rename(columns={0:"Negative",1:"Positive"})
    df = df.sort_values(by=['Positive'],ascending=False)

    ax = df.plot.bar(rot=0,title ="Positive/Negative Count",figsize=(20,11),legend=True, fontsize=12)
    ax.set_yticks([i for i in range(0,df['Negative'].max()+1000,1000)])
    ax.set_yticklabels([i for i in range(0,df['Negative'].max()+1000,1000)], fontsize=8)
    ax.set_xlabel("Disease_Type",fontsize=14)
    ax.set_ylabel("Count",fontsize=14)
    ax.grid(True,linestyle='--',color='0.75')
    
    return original_df
value_count(TRAIN_df)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     rotation_range=45, 
                                     width_shift_range=0.1, 
                                     height_shift_range=0.1, 
                                     zoom_range=0.1, 
                                     vertical_flip=True,
                                     horizontal_flip=True, 
                                     fill_mode="nearest")

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_flow = train_datagen.flow_from_dataframe(dataframe=TRAIN_df,
                                          directory=DATA_PATH,
                                          x_col='image_id',
                                          y_col=list(TRAIN_df.columns[1:]), class_mode="multi_output",
                                          target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                                          batch_size=config.BATCH_SIZE, shuffle=True, validate_filenames=True, drop_duplicates=False)

val_flow = val_datagen.flow_from_dataframe(dataframe=VAL_df,
                                        directory=DATA_PATH,
                                        x_col='image_id',
                                        y_col=list(VAL_df.columns[1:]), class_mode="multi_output",
                                        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                                        batch_size=config.BATCH_SIZE, shuffle=False, validate_filenames=True, drop_duplicates=False)
labels = list(TRAIN_df.columns[1:])
labels
inputs = Input((config.IMG_HEIGHT, config.IMG_WIDTH, 3))

base_model = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max')
x = base_model(inputs)

outputs = [Dense(1, name='{}'.format(i), activation='sigmoid')(x) for i in list(TRAIN_df.columns[1:])]

model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(config.lr, epsilon=1e-08)

for layer in model.layers:
    layer.trainable = True
    
model.compile(optimizer=optimizer, loss=config.LOSS, metrics=['accuracy'])

model.summary()

print("Trainable layers...")
for layer in model.layers:
    print(layer.name, ':', layer.trainable)
print("")
# Model checkpoint
model_checkpoint = ModelCheckpoint(filepath=os.path.join(SAVE_PATH,'{}_{}_{}_{dt}.hdf5'.format(MODEL_NAME,
                                                                                                config.PRETRAINED_MODEL,
                                                                                                config.lr,
                                                                                                dt=dt.now().strftime("%Y-%m-%d--%H-%M"))),
                                                                                                monitor='val_loss',
                                                                                                verbose=1, save_best_only=True, save_weights_only=True)

# Learning Rate Reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,
                                            factor=0.1,
                                            min_lr=0.0000001)

# Early stopping
earlystop = EarlyStopping(monitor='val_loss', patience=4)

# Callbacks
callbacks_list = [model_checkpoint, learning_rate_reduction, earlystop, WandbCallback()]
model.fit(train_flow,
          epochs=config.EPOCHS,
          verbose=1,
          validation_data=val_flow,
          validation_steps=math.ceil(val_flow.samples / val_flow.batch_size),
          steps_per_epoch=math.ceil(train_flow.samples / train_flow.batch_size),
          callbacks=callbacks_list)
# Predicting Test data

TEST_df =  pd.read_csv(test_csv)
TEST_df['image_id'] = TEST_df.image_id.apply(lambda x: x+'.jpg')
TEST_df.head()
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_flow = datagen.flow_from_dataframe(dataframe=TEST_df,
                                          directory=DATA_PATH,
                                          x_col='image_id',
                                          y_col=None, class_mode=None,
                                          target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                                          batch_size=config.BATCH_SIZE, shuffle=False, validate_filenames=True, drop_duplicates=False)
model_predictions = model.predict(test_flow,
                                  verbose=1,
                                  steps=math.ceil(test_flow.samples/test_flow.batch_size))
submission_df = pd.DataFrame(columns=TRAIN_df.columns)
submission_df['image_id'] = TEST_df['image_id']
submission_df['image_id'] = submission_df.image_id.apply(lambda x: x.split('.')[0])

submission_df['healthy'] = model_predictions[0]
submission_df['multiple_diseases'] = model_predictions[1]
submission_df['rust'] = model_predictions[2]
submission_df['scab'] = model_predictions[3]
submission_df
submission_df.to_csv('submission2_26_4_2020.csv',index=False)
asm_df1 = pd.DataFrame(columns=sub1.columns)
asm_df1['image_id'] = sub1['image_id']
asm_df2 = asm_df1.copy()
asm_df3 = asm_df1.copy()
asm_df1.iloc[:, 1:] = sub1.iloc[:,1:]*0.5 + sub2.iloc[:,1:]*0.5
asm_df2.iloc[:, 1:] = sub1.iloc[:,1:]*0.75 + sub2.iloc[:,1:]*0.25
asm_df3.iloc[:, 1:] = sub1.iloc[:,1:]*0.25 + sub2.iloc[:,1:]*0.75

asm_df1.to_csv('submission_ensemble_1.csv', index=False)
asm_df2.to_csv('submission_ensemble_2.csv', index=False)
asm_df3.to_csv('submission_ensemble_3.csv', index=False)