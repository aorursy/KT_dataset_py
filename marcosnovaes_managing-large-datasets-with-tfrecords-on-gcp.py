# After you have linked your project change this flag to True to use GCS and execute the next cell as well to enable access
# To link your GCP project see Menu "Add-ons->Google Cloud SDK"
GCP_SDK_ENABLED = False

## IMPORTANT
## YOU MUST MODIFY THE LINE BELOW WITH THE NAME OF THE BUCKET YOU ARE USING IN GCS
BUCKET_NAME = "gs://your-unique-bucket-name-here"
# Authorize Tensorflow to write directly to GCS
# This requires this Notebook to be linked to a GCP project, see Menu "Add-ons->Google Cloud SDK"
# After linking the project, import credentials and authorize Tensorflow as follows

from kaggle_secrets import UserSecretsClient

if( GCP_SDK_ENABLED):
    user_secrets = UserSecretsClient()
    user_credential = user_secrets.get_gcloud_credential()
    user_secrets.set_tensorflow_credential(user_credential)
!conda install -c conda-forge gdcm -y
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
import scipy.ndimage
import gdcm

from os import listdir, mkdir
import os
listdir("../input/")
basepath = "../input/rsna-str-pulmonary-embolism-detection/"
listdir(basepath)
train_df = pd.read_csv(basepath + "train.csv")
test_df = pd.read_csv(basepath + "test.csv")
train_df.head()
# create a list of unique Study Ids
list_of_studies = train_df.StudyInstanceUID.unique()
list_of_studies.shape
# create a list of file directories for each study 
train_df["dcm_path"] = basepath + "train/" + train_df.StudyInstanceUID + "/" + train_df.SeriesInstanceUID
list_of_directories = train_df.dcm_path.unique()
list_of_directories.shape
def load_dicom_array(dcm_path):
    #dicom_files = glob.glob(osp.join(f, '*.dcm'))
    #dicoms = [pydicom.dcmread(d) for d in dicom_files]
    dicom_files = listdir(dcm_path)
    dicoms = [pydicom.dcmread(dcm_path + "/" + file) for file in listdir(dcm_path)]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    # Assume all images are axial
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    dicoms = np.asarray([d.pixel_array for d in dicoms])
    dicoms = dicoms[np.argsort(z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    return dicoms, np.asarray(dicom_files)[np.argsort(z_pos)]
dicom_imgs, img_names = load_dicom_array(list_of_directories[0])
img_names
fig, ax = plt.subplots(1,2,figsize=(20,3))
ax[0].set_title("Original CT-scan")
ax[0].imshow(dicom_imgs[0], cmap="bone")
ax[1].set_title("Pixelarray distribution");
sns.distplot(dicom_imgs[0].flatten(), ax=ax[1]);
def CT_window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    #X = (X*255.0).astype('uint8')
    return X
windowed_ct = CT_window(dicom_imgs[0], 100, 700)
fig, ax = plt.subplots(1,2,figsize=(20,3))
ax[0].set_title("PE Specific CT-scan")
ax[0].imshow(windowed_ct, cmap="bone")
ax[1].set_title("Pixelarray distribution");
sns.distplot(windowed_ct.flatten(), ax=ax[1]);
# Define the TFExample Data type for training models
# Our TFRecord format will include the CT Image and metadata of the image, including the prediction label (is PE present)

import tensorflow as tf

# Utilities serialize data into a TFRecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image, study_id, image_name, pred_label):
    image_shape = image.shape
    image_bytes = image.tostring()
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'image_raw': _bytes_feature(image_bytes),
        'study_id': _bytes_feature(study_id.encode()),
        'img_name': _bytes_feature(image_name.encode()),
        'pred_label':  _int64_feature(pred_label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Define function to write the TFRecord
# First, process each image into `tf.Example` messages.
# Then,TFRecordWriter to write to a `.tfrecords` file.


PE_WINDOW_LEVEL = 100
PE_WINDOW_WIDTH = 700

def create_tfrecord( study_id, study_path, sdk_enabled=False):
    if(sdk_enabled):
        storage_file_path = BUCKET_NAME+"/RSNA_PE/PE_Window_512/train/"+study_id+".tfrecords"
    else:
        storage_file_path = '/kaggle/working/'+study_id+'.tfrecords'

    study_images, study_image_file_names = load_dicom_array(study_path)
    num_records = study_images.__len__()

    total_records = 0
    with tf.io.TFRecordWriter(storage_file_path) as writer:
        for index in range(num_records):
            img_file_name = study_image_file_names[index]
            img_name = img_file_name.split(".")[0]
            img_data = train_df.loc[train_df["SOPInstanceUID"] == img_name]
            pred_label = img_data["pe_present_on_image"].values[0]
            #print("pred_label ",pred_label)
            windowed_image = CT_window(study_images[index], PE_WINDOW_LEVEL, PE_WINDOW_WIDTH)
            tf_example = image_example(windowed_image, study_id, img_name, pred_label)
            writer.write(tf_example.SerializeToString())
            total_records = total_records + 1
            print("*",end='')
            #print("wrote {}".format(img_name))
        writer.close()
        
    print("wrote {} records".format(total_records))
    return total_records
# This will write to local storage, just as a test. You can then refresh the output diretory and see the record file
num_records = create_tfrecord( list_of_studies[0], list_of_directories[0], False)
# now write to Cloud Storage (if GCP_SDK_ENABLED)
num_records = create_tfrecord( list_of_studies[0], list_of_directories[0], GCP_SDK_ENABLED)
num_records
# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'study_id': tf.io.FixedLenFeature([], tf.string),
    'img_name': tf.io.FixedLenFeature([], tf.string),
    'pred_label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)
# and now we can reload the dataset again directly from GCS

sample_study_id = list_of_studies[0]
if(GCP_SDK_ENABLED):
    storage_file_path = BUCKET_NAME+"/RSNA_PE/PE_Window_512/train/"+sample_study_id+".tfrecords"
else:
    storage_file_path = '/kaggle/working/'+sample_study_id+'.tfrecords'

encoded_image_dataset = tf.data.TFRecordDataset(storage_file_path)
parsed_image_dataset = encoded_image_dataset.map(_parse_image_function)
parsed_image_dataset
# extract a record from the dataset and display the image
def load_dataset(dataset) :
    reloaded_images = []
    img_mtd = []
    i=0
    for image_features in dataset.as_numpy_iterator():
        i=i+1
        sample_image = np.frombuffer(image_features['image_raw'], dtype='float64')
        mtd = dict()
        mtd['width'] = image_features['width']
        mtd['height'] = image_features['height']
        mtd['study_id'] = image_features['study_id'].decode()
        mtd['img_name'] = image_features['img_name'].decode()
        mtd['pred_label'] = image_features['pred_label']                                  
        reloaded_images.append(sample_image.reshape(mtd['width'],mtd['height'])) 
        img_mtd.append(mtd)
    return reloaded_images, img_mtd
#print(len(sample_array))
reloaded_images, img_mtd = load_dataset(parsed_image_dataset)
print(reloaded_images[0].shape)  
print(img_mtd[0])
fig, ax = plt.subplots(1,2,figsize=(20,3))
ax[0].set_title("Reloaded CT-scan {}".format(img_mtd[0]['img_name']))
ax[0].imshow(reloaded_images[0], cmap="bone")
ax[1].set_title("Pixelarray distribution");
sns.distplot(reloaded_images[0].flatten(), ax=ax[1]);
num_studies = list_of_studies.shape[0]
lower_range = 0
upper_range = 3
# you can upload up to num_studies, which is more than 17,000 and will take days. 
for index in range(lower_range,upper_range):
    print("processing study {} out of {}".format(index,num_studies))
    print("writing tfrecords for {}".format(list_of_studies[index]))
    num_records = create_tfrecord(list_of_studies[index], list_of_directories[index], GCP_SDK_ENABLED)
    print("wrote {} records".format(num_records))