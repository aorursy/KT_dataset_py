import os 
print(os.listdir("/kaggle/input/skin-cancer-mnist-ham10000"))
dataPath = "/kaggle/input/skin-cancer-mnist-ham10000/"
import numpy as np
import pandas as pd
df = pd.read_csv(dataPath+'HAM10000_metadata.csv')
df.head()
dx = df['dx'].value_counts().sort_index()
print(dx)
categories = dx.index.values
print(categories)

counts = dx.values
print(counts)
#labels = ['光化角化病', '基底細胞癌', '良性角化病', '皮膚纖維瘤', '惡性黑色素瘤', '黑素細胞痣', '血管病變']
labels = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Malignant Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

num_classes = len(labels) # = len(categories)
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style("whitegrid")

def plot_equilibre(categories, counts):

    plt.figure(figsize=(12, 8))

    sns_bar = sns.barplot(x=categories, y=counts)
    sns_bar.set_xticklabels(categories, rotation=45)
    plt.title('Equilibre of Training Dataset')
    plt.show()
plot_equilibre(categories, counts)
# create local data directory 
data_dir = 'data'
os.mkdir(data_dir)
train_dir = os.path.join(data_dir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(data_dir, 'val')
os.mkdir(val_dir)
test_dir = os.path.join(data_dir, 'test')
os.mkdir(test_dir)
# create directory for each category in train/val/test directory
for category in categories:
    os.mkdir(os.path.join(train_dir, category))
    os.mkdir(os.path.join(val_dir,   category))
    os.mkdir(os.path.join(test_dir,  category))
!ls data/train
!ls data/val
!ls data/test
from sklearn.model_selection import train_test_split
# Split to train and validation set
df_train, df_tmp = train_test_split(df, test_size=0.2, random_state=101, stratify=df['dx'])
df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=101)
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
# image_id as df index
df_train = df_train.set_index('image_id') 
df_val   = df_val.set_index('image_id') 
df_test  = df_test.set_index('image_id')
import shutil
folder_1 = os.listdir(dataPath +'ham10000_images_part_1')
folder_2 = os.listdir(dataPath +'ham10000_images_part_2')

def copy_files(df, data_dir):
    fileList = df.index.values
    
    for file in fileList:
        fname = file + '.jpg'
        label = df.loc[file, 'dx'] 

        if fname in folder_1:
            src = os.path.join(dataPath+'ham10000_images_part_1', fname)
            dst = os.path.join(data_dir, label, fname)
            shutil.copyfile(src, dst)
            
        if fname in folder_2:
            src = os.path.join(dataPath+'ham10000_images_part_2', fname)
            dst = os.path.join(data_dir, label, fname)
            shutil.copyfile(src, dst)
copy_files(df_train, train_dir)
copy_files(df_val, val_dir)
copy_files(df_test, test_dir)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

target_size = (224,224)
batch_size = 16

# Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical')
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    'data/val',
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=False,    
    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=False,    
    class_mode='categorical')
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
#base_model=keras.applications.MobileNetV2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.InceptionV3(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.ResNet50V2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.ResNet101V2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.ResNet152V2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.DenseNet121(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.DenseNet169(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.DenseNet201(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.NASNetMobile(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=keras.applications.NASNetLarge(input_shape=(331,331,3), weights='imagenet',include_top=False)
!pip install -q efficientnet
import efficientnet.tfkeras as efn
base_model = efn.EfficientNetB7(input_shape=(224,224,3), weights='imagenet', include_top=False)
## Add Extra Layers to Model
x=base_model.output
x=GlobalAveragePooling2D()(x)      
x=Dense(1024,activation='relu')(x) 
x=Dense(64,activation='relu')(x)
out=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=out)
## for transfer learning
base_model.trainable = False 

model.summary()
# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST =test_generator.n//test_generator.batch_size

num_epochs = 30
# Add weights to make the model more sensitive to melanoma due to data equilibre
class_weights={
    0: 1.0,  # akiec
    1: 1.0,  # bcc
    2: 1.0,  # bkl
    3: 1.0,  # df
    4: 3.0,  # mel
    5: 1.0,  # nv
    6: 1.0,  # vasc
}
model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=num_epochs, class_weight=class_weights, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID)
# Save Model
model.save('tl_skinlesion.h5')
# Evaluate Model
loss, acc = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
print("The accuracy of the model is {:.3f}\nThe Loss in the model is {:.3f}".format(acc,loss))
from sklearn.metrics import classification_report, confusion_matrix

predY=model.predict_generator(test_generator)
y_pred = np.argmax(predY,axis=1)
#y_label= [labels[k] for k in y_pred]
y_actual = test_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)
print(classification_report(y_actual, y_pred, target_names=labels))
fig, ax = plt.subplots()
ax.matshow(cm, cmap='Blues')

for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
plt.show()
