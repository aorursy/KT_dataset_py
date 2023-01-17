# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
main_dir = '/kaggle/input/rsna-pneumonia-detection-challenge'
training_image_dir = main_dir + '/' + 'stage_2_train_images'
test_image_dir = main_dir + '/' + 'stage_2_test_images'
os.getcwd()

num_of_training_samples = len([name for name in os.listdir(training_image_dir) if os.path.isfile(os.path.join(training_image_dir, name)) ])
num_of_test_samples = len([name for name in os.listdir(test_image_dir) if os.path.isfile(os.path.join(test_image_dir, name)) ])

print('Number of training samples : ', num_of_training_samples  )
print('Number of test samples     : ', num_of_test_samples )
import pandas as pd

train_class_info = pd.read_csv(main_dir + "/stage_2_detailed_class_info.csv")
display(train_class_info)
train_labels = pd.read_csv(main_dir +  "/stage_2_train_labels.csv")
display(train_labels)
train_labels.nunique()
print(train_labels.shape)
print(train_class_info.shape)

train_label_class_info_merged =  pd.merge(train_class_info, train_labels, how='inner',left_index = True, right_index = True )
train_label_class_info_merged.drop('patientId_y',axis = 1, inplace=True)
train_label_class_info_merged.rename(columns={"patientId_x": "patientId"},inplace = True)
train_label_class_info_merged.loc [:, ['Target','class'] ].value_counts()
train_label_class_info_merged.loc [ train_label_class_info_merged.Target == 1 , ['x','y','width','height'] ].isna().count()
train_label_class_info_merged.groupby(['patientId'], sort=False).count().max()
train_label_class_info_merged
train_label_class_info_merged.sort_values(by=['patientId'],inplace = True)
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def fn_image_info_plot_image( image_dir, image_file_name, bPlot = True ) :
    filename = image_dir + '/' + image_file_name
    dataset = pydicom.dcmread(filename)

    print()
    print("Filename.........:", filename)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)
    print("Patient Age .......:", dataset.PatientAge)
    print("Patient Sex .......:", dataset.PatientSex)
    print("Body Part Examined .......:", dataset.BodyPartExamined)
    print("View Position .......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(rows=rows, cols=cols, size=len(dataset.PixelData)))
    
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))
    
    if bPlot == True :
        f, ax = plt.subplots(1,1, figsize=(8,9))
        
        plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
        
        rows = train_label_class_info_merged[train_label_class_info_merged['patientId']==dataset.PatientID]
        box_data = list(rows.T.to_dict().values())
        for j, row in enumerate(box_data):
            ax.add_patch(Rectangle(xy=(row['x'], row['y']),
                        width=row['width'],height=row['height'], 
                        color="yellow",alpha = 0.1))   
        
        plt.show()
# fn_image_info_plot_image(training_image_dir,'000db696-cf54-4385-b10b-6b16fbb3f985.dcm', True )

fn_image_info_plot_image(training_image_dir,'c1ec14ff-f6d7-4b38-b0cb-fe07041cbdc8.dcm',True)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
ALPHA = 1.0
import cv2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def prepare_image_data(imagedata):
    img = cv2.resize(imagedata, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) != 3 or img.shape[2] != 3:
            img = np.stack((img,) * 3, -1)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis = 0)
    return preprocess_input(img_array_expanded_dims)
masks = np.zeros((num_of_training_samples, IMAGE_HEIGHT, IMAGE_WIDTH))
X_train = np.zeros((num_of_training_samples, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

training_image_meta_info = pd.DataFrame(columns = ['PatientID', 'Modality', 'PatientAge','PatientSex','BodyPartExamined','ViewPosition','Rows','Columns','PixelSpacing','SliceLocation','FileName'])

index = 0

for filename in os.listdir(training_image_dir) :
    dataset = pydicom.dcmread(os.path.join(training_image_dir, filename))
    training_image_meta_info = training_image_meta_info.append({'PatientID' : dataset.PatientID, 'Modality': dataset.Modality, 'PatientAge': dataset.PatientAge, 'PatientSex' : dataset.PatientSex, 'BodyPartExamined': dataset.BodyPartExamined, 'ViewPosition' : dataset.ViewPosition, 'Rows' : int(dataset.Rows), 'Columns' : int(dataset.Columns), 'PixelSpacing' : dataset.PixelSpacing, 'SliceLocation' : dataset.get('SliceLocation', "-1"), 'FileName' : filename},
                                                               ignore_index = True)
               
    #try:
    #    img = img[:, :, :3]
    #except:
    #    continue
            
    # X_train[index] = preprocess_input(np.array(img, dtype=np.float32))
    X_train[index] = prepare_image_data(dataset.pixel_array)
    
    mask_df = train_labels.loc[ ( train_labels.patientId == dataset.PatientID ) &  ( train_labels.Target == 1 ) ,['x','y','width','height'] ]
        
    for mask_df_index, mask_df_row in mask_df.iterrows():
        x1 = int(mask_df_row['x'] * IMAGE_WIDTH)
        x2 = int(( mask_df_row['x'] + mask_df_row['width'] )  * IMAGE_WIDTH)
        y1 = int(mask_df_row['y'] * IMAGE_HEIGHT)
        y2 = int(( mask_df_row['y'] + mask_df_row['height'] ) * IMAGE_HEIGHT)
        masks[index][y1:y2, x1:x2] = 1
        
    index += 1
test_image_meta_info = pd.DataFrame(columns = ['PatientID', 'Modality', 'PatientAge','PatientSex','BodyPartExamined','ViewPosition','Rows','Columns','PixelSpacing','SliceLocation','FileName'])

for filename in os.listdir(test_image_dir) :
    dataset = pydicom.dcmread(os.path.join(test_image_dir, filename))
    test_image_meta_info = test_image_meta_info.append({'PatientID' : dataset.PatientID, 'Modality': dataset.Modality, 'PatientAge': dataset.PatientAge, 'PatientSex' : dataset.PatientSex, 'BodyPartExamined': dataset.BodyPartExamined, 'ViewPosition' : dataset.ViewPosition, 'Rows' : int(dataset.Rows), 'Columns' : int(dataset.Columns), 'PixelSpacing' : dataset.PixelSpacing, 'SliceLocation' : dataset.get('SliceLocation', "-1"), 'FileName' : filename},
                           ignore_index = True)
training_image_meta_info
test_image_meta_info
patient_class = train_label_class_info_merged.groupby('patientId')['class','Target'].max()
patient_class
patient_image_meta_info_merged = pd.merge( training_image_meta_info, patient_class, how='inner', left_on = 'PatientID', right_on = 'patientId')
patient_info = patient_image_meta_info_merged.drop(['Modality','BodyPartExamined','Rows','Columns','PixelSpacing','SliceLocation','FileName'],axis = 1)

patient_info.loc[patient_info.PatientID == '000db696-cf54-4385-b10b-6b16fbb3f985',:]
patient_info.loc[patient_info.Target == 1, 'class'].value_counts()
patient_info.loc[patient_info.Target == 1, 'ViewPosition'].value_counts()
patient_info.loc[patient_info.Target == 1, 'PatientSex'].value_counts()
import seaborn as sns

sns.distplot(patient_info.loc[patient_info.Target == 0, 'PatientAge'], hist=False, rug=True)
sns.distplot(patient_info.loc[patient_info.Target == 1, 'PatientAge'], hist=False, rug=True)
sns.countplot(patient_info['PatientSex'], hue=patient_info[ 'Target']) 
sns.countplot(patient_info['ViewPosition'], hue=patient_info[ 'Target']) 
tmp = patient_info.groupby(['class', 'PatientSex'])['PatientID'].count()
df1 = pd.DataFrame(data={'Count': tmp.values}, index=tmp.index).reset_index()
tmp = df1.groupby(['Count','class', 'PatientSex']).count()
df3 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()
fig, (ax) = plt.subplots(nrows=1,figsize=(6,6))
sns.barplot(ax=ax, x = 'PatientSex', y='Count', hue='class',data=df3)
plt.title("Train set: Patient Sex and class")
plt.show()
from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet

def create_model(trainable=True):
    model = MobileNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, alpha=ALPHA, weights="imagenet")

    for layer in model.layers:
        layer.trainable = trainable

    # Add all the UNET layers here
    block  = model.get_layer("conv_pw_1_relu").output
    block1 = model.get_layer("conv_pw_3_relu").output
    block2 = model.get_layer("conv_pw_5_relu").output
    block3 = model.get_layer("conv_pw_11_relu").output
    block4 = model.get_layer("conv_pw_13_relu").output

    x = Concatenate()([UpSampling2D()(block4), block3])
    x = Concatenate()([UpSampling2D()(x), block2])
    x = Concatenate()([UpSampling2D()(x), block1])
    x = Concatenate()([UpSampling2D()(x), block])
    x = UpSampling2D()(x)
    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)

    x = Reshape((IMAGE_HEIGHT, IMAGE_HEIGHT))(x)

    return Model(inputs=model.input, outputs=x)
model = create_model(trainable=False)

model.summary()
from tensorflow import reduce_sum
from tensorflow.keras.backend import epsilon

def dice_coefficient(y_true, y_pred):
  numerator = 2 * reduce_sum(y_true * y_pred)
  denominator = reduce_sum(y_true + y_pred)

  return numerator / (denominator + epsilon())
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import log, epsilon
def loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())
model.compile(optimizer='adam',
              loss=loss,
              metrics=[dice_coefficient])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,
                             save_weights_only=True, mode="min",  save_freq = 1)
stop = EarlyStopping(monitor="loss", patience=5, mode="min")
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="min")
#batch_size=1
#epochs=1

#model.fit(
#  X_train, masks, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint,reduce_lr,stop]
#)
