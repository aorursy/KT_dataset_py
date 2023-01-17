import numpy as np

import pandas as pd

import os

import pydicom

import matplotlib.pyplot as plt

import seaborn as sns

import json

import cv2

from keras import layers

from keras.applications import DenseNet121

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

from tqdm import tqdm

print ("Complete Loding in the Libraries.")
#Load in the train and sub data

train = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")

sub = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")

train_images = os.listdir("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/")

test_images = os.listdir("../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/")

print("Train_row:", train.shape[0])

print("Train_col:", train.shape[1])

print("Sub_row:", sub.shape[0])

print("Sub_col:", sub.shape[1])
train.head()
train["type"] = train["ID"].str.split("_", n = 3, expand = True)[2]

train["PatientID"] = train["ID"].str.split("_", n = 3, expand = True)[1]

train["filename"] = train["ID"].apply(lambda st: "ID_" + st.split('_')[1] + ".png")



sub["filename"] = sub["ID"].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

sub["type"] = sub["ID"].apply(lambda st: st.split('_')[2])
print(train.Label.value_counts())

sns.set_palette("winter_r", 7, 0.5)

sns.countplot(x='Label', data=train)
sns.countplot(x="Label", hue="type", data=train)
TRAIN_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"

TEST_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"

BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'stage_1_train_images/'

TEST_DIR = 'stage_1_test_images/'



def window_image(img, window_center,window_width, intercept, slope, rescale=True):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    

    if rescale:

        # Extra rescaling to 0-1, not in the original notebook

        img = (img - img_min) / (img_max - img_min)

    

    return img

    

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]



    

    

def view_images(images, title = '', aug = None):

    width = 5

    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    

    for im in range(0, height * width):

        data = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,'ID_'+images[im]+ '.dcm'))

        image = data.pixel_array

        window_center , window_width, intercept, slope = get_windowing(data)

        image_windowed = window_image(image, window_center, window_width, intercept, slope)





        i = im // width

        j = im % width

        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

    plt.show()



case = 5

data = pydicom.dcmread(TRAIN_IMG_PATH+train_images[case])



print(data)

window_center , window_width, intercept, slope = get_windowing(data)





#displaying the image

img = pydicom.read_file(TRAIN_IMG_PATH+train_images[case]).pixel_array



img = window_image(img, window_center, window_width, intercept, slope)

plt.imshow(img, cmap=plt.cm.bone)

plt.grid(False)
view_images(train[(train['type'] == 'epidural') & (train['Label'] == 1)][:10].PatientID.values, title = '1.epidural')
view_images(train[(train['type'] == 'intraparenchymal') & (train['Label'] == 1)][:10].PatientID.values, title = '2.intraparenchymal')
view_images(train[(train['type'] == 'intraventricular') & (train['Label'] == 1)][:10].PatientID.values, title = '3.intraventricular')
view_images(train[(train['type'] == 'subarachnoid') & (train['Label'] == 1)][:10].PatientID.values, title = '4.subarachnoid')
view_images(train[(train['type'] == 'subdural') & (train['Label'] == 1)][:10].PatientID.values, title = '5.subdural')
view_images(train[(train['type'] == 'any') & (train['Label'] == 1)][:10].PatientID.values, title = '6.any')
test = pd.DataFrame(sub.filename.unique(), columns=["filename"])

print("Test_row:", test.shape[0])

print("Test_col:", test.shape[1])
test.head()
sample_files = np.random.choice(os.listdir(TRAIN_IMG_PATH), 200000)

sample_df = train[train.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]



pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(

    index='filename', columns='type', values='Label').reset_index()

print(pivot_df.shape)
def save_and_resize(filenames, load_dir):    

    save_dir = '/kaggle/tmp/'

    if not os.path.exists(save_dir):

        os.makedirs(save_dir)



    for filename in tqdm(filenames):

        path = load_dir + filename

        new_path = save_dir + filename.replace('.dcm', '.png')

        

        dcm = pydicom.dcmread(path)

        window_center , window_width, intercept, slope = get_windowing(dcm)

        img = dcm.pixel_array

        img = window_image(img, window_center, window_width, intercept, slope)

        

        resized = cv2.resize(img, (224, 224))

        res = cv2.imwrite(new_path, resized)

        if not res:

            print('Failed')

            

save_and_resize(filenames=sample_files, load_dir=BASE_PATH + TRAIN_DIR)

save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)
densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(224,224,3)

)
model = build_model()

model.summary()
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



total_steps = sample_files.shape[0] / BATCH_SIZE



history = model.fit_generator(

    train_gen,

    steps_per_epoch=total_steps * 0.85,

    validation_data=val_gen,

    validation_steps=total_steps * 0.15,

    callbacks=[checkpoint],

    epochs=11

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
model.load_weights('model.h5')

y_test = model.predict_generator(

    test_gen,

    steps=len(test_gen),

    verbose=1

)
test_df[['ID', 'Label']].head(10)
