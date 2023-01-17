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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dcm
import cv2
import os
train_df = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
print("DataFrame Shape: ", train_df.shape)

train_df.head()
def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def view_images(files, width , height, title = '', aug = None, windowing = True, rescale= False):
    fig, axs = plt.subplots(height, width, figsize=(15,15))
    
    for im in range(0, height * width):
        data = dcm.dcmread(files[im])
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        if windowing:
            output = window_image(image, window_center, window_width, intercept, slope, rescale = rescale)
        else:
            output = image
        i = im // width
        j = im % width
        if width == 1 and height == 1:
            axs.imshow(output, cmap=plt.cm.gray) 
            axs.axis('off')
        else:
            axs[i, j].imshow(output, cmap=plt.cm.gray) 
            axs[i, j].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    

root_path = "../input/rsna-str-pulmonary-embolism-detection/train/"
example_files = [root_path + "0003b3d648eb/d2b2960c2bbf/00ac73cfc372.dcm",
                 root_path + "0003b3d648eb/d2b2960c2bbf/03d7693b0405.dcm",
                root_path + "0003b3d648eb/d2b2960c2bbf/055eabedd904.dcm", 
                root_path + "0003b3d648eb/d2b2960c2bbf/084e7b3d3d6c.dcm"]
view_images(example_files,2, 2, 'Images with Windowing')
def sample(train_df, row, nosamples):
    t = train_df.values
    t = t[:, 3:]
    filterd_smaple = train_df.iloc[np.where(list(map(lambda x : all(x), t[:, :] == t[row, :])))]
    samples = filterd_smaple.groupby(['StudyInstanceUID', 'SeriesInstanceUID']).first().reset_index().values
    np.random.shuffle(samples)
    samples = samples[np.random.randint(0, samples.shape[0], (nosamples, 1))]
    return np.squeeze(samples)
samples = sample(train_df, 0, 10)
samples = list(map(lambda x : root_path + "/".join(x) +".dcm", samples[:, :3]))
view_images(samples[:4],2, 2, 'Images with Windowing')
for col in train_df.columns[3:]:
    plt.bar(['0','1'], train_df[col].value_counts().values)
    plt.title(col)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.animation as animation
from IPython.display import HTML
feature_extractor = tf.keras.applications.VGG16(include_top = False)
input_layer = feature_extractor.input

output_layer = [out.output for out in feature_extractor.layers[1:]]
featur_extraction_model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
def decode_dcm(file, rescale):
    data = dcm.dcmread(file)
    image = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    output = window_image(image, window_center, window_width, intercept, slope, rescale = rescale)
    return output
file = example_files[0]
image = decode_dcm(file, True)
image = np.repeat(image[..., np.newaxis], 3, -1)
image = np.expand_dims(image, axis = 0)
preprocessed_image = preprocess_input(image)

print("Image Shape              : ", image.shape)
print("PreProcessed Image Shape : ", preprocessed_image.shape)
%time
predictions = featur_extraction_model.predict(image)
def layer_features(predictions, layer_idx):
    fig = plt.figure(figsize = (8, 8))
    img_list = []
    for pred in range(predictions[layer_idx].shape[-1]):
        img = plt.imshow(predictions[layer_idx][0, :, :, pred], cmap='gray', animated=True)
        img_list.append([img])

    ani = animation.ArtistAnimation(fig, img_list, interval=predictions[layer_idx].shape[-1] * 10, blit=True,
                                    repeat_delay=1000)
    return ani

ani = layer_features(predictions, 0)
HTML(ani.to_html5_video())
ani = layer_features(predictions, 1)
HTML(ani.to_html5_video())
ani = layer_features(predictions, 2)
HTML(ani.to_html5_video())
ani = layer_features(predictions, 3)
HTML(ani.to_html5_video())
file = example_files[0]
image = decode_dcm(file, True)
image = cv2.resize(image, (224, 224))
image = np.repeat(image[..., np.newaxis], 3, -1)
image = np.expand_dims(image, axis = 0)
preprocessed_image = preprocess_input(image)

predictions2 = featur_extraction_model.predict(image)
f, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].imshow(predictions2[0][0, :, :, 2], cmap = 'gray')
ax[0,0].set_title('Feature of Image with resize')
ax[0, 1].imshow(predictions[0][0, :, :, 2], cmap = 'gray')
ax[0, 1].set_title('Feature of Image without resize')
ax[1, 0].imshow(image[0, :, :, 0], cmap = 'gray')
ax[1, 0].set_title('Original Image with resize')
ax[1, 1].imshow(decode_dcm(file, True), cmap = 'gray')
ax[1, 1].set_title('Original Image Without resize')
f, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].imshow(predictions2[5][0, :, :, 2], cmap = 'gray')
ax[0,0].set_title('Feature of Image with resize at layer_5')
ax[0, 1].imshow(predictions[5][0, :, :, 2], cmap = 'gray')
ax[0, 1].set_title('Feature of Image without resize at layer_5')
ax[1, 0].imshow(predictions[6][0, :, :, 2], cmap = 'gray')
ax[1, 0].set_title('Feature of Image without resize at layer_6')
ax[1, 1].imshow(predictions[7][0, :, :, 2], cmap = 'gray')
ax[1, 1].set_title('Feature of Image without resize at layer_7')
f, ax = plt.subplots(2, 2, figsize=(15, 15))
ax[0, 0].imshow(predictions2[5][0, :, :, 2], cmap = 'gray')
ax[0,0].set_title('Feature of Image with resize at layer_5')
ax[0, 1].imshow(predictions[8][0, :, :, 2], cmap = 'gray')
ax[0, 1].set_title('Feature of Image without resize at layer_8')
ax[1, 0].imshow(predictions[9][0, :, :, 2], cmap = 'gray')
ax[1, 0].set_title('Feature of Image without resize at layer_9')
ax[1, 1].imshow(predictions[10][0, :, :, 2], cmap = 'gray')
ax[1, 1].set_title('Feature of Image without resize at layer_10')
train_df.groupby(['StudyInstanceUID', 'SeriesInstanceUID']).count().reset_index().head()
train_df.loc[(train_df.StudyInstanceUID == '6897fa9de148') & (train_df.SeriesInstanceUID == '2bfbb7fd2e8b')][train_df.columns[3:]].sum()
train_df.loc[(train_df.StudyInstanceUID == '0003b3d648eb') & (train_df.SeriesInstanceUID == 'd2b2960c2bbf')][train_df.columns[3:]].sum()
train_df.loc[(train_df.StudyInstanceUID == '000f7f114264')][train_df.columns[3:]].sum()
negative_exam_for_pe_0 = train_df.loc[(train_df.StudyInstanceUID == '6897fa9de148') & (train_df.SeriesInstanceUID == '2bfbb7fd2e8b')][train_df.columns[:3]].values
negative_exam_for_pe_1 = train_df.loc[(train_df.StudyInstanceUID == '0003b3d648eb') & (train_df.SeriesInstanceUID == 'd2b2960c2bbf')][train_df.columns[:3]].values
negative_exam_for_pe_0_sample = os.path.join(root_path, "/".join(negative_exam_for_pe_0[5]) + ".dcm")
negative_exam_for_pe_1_sample = os.path.join(root_path, "/".join(negative_exam_for_pe_1[10]) + ".dcm")
negative_exam_for_pe_1_sample1 = os.path.join(root_path, "/".join(negative_exam_for_pe_1[100]) + ".dcm")
negative_exam_for_pe_1_sample2 = os.path.join(root_path, "/".join(negative_exam_for_pe_1[121]) + ".dcm")
view_images([negative_exam_for_pe_0_sample, negative_exam_for_pe_1_sample, negative_exam_for_pe_1_sample1, negative_exam_for_pe_1_sample2], 2, 2, 'Images with Windowing')
for i in range(len(predictions)):
    print(f"layer_{i} Shape ---> ", predictions[i].shape)