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

train_df["dcm_path"] = basepath + "train/" + train_df.StudyInstanceUID + "/" + train_df.SeriesInstanceUID + "/" + train_df.SOPInstanceUID + ".dcm"

list_of_directories = train_df.dcm_path.unique()

list_of_directories.shape
# make list of positive PE studies (negative_exam_for_pe = 0)

positive_studies = train_df.loc[train_df["negative_exam_for_pe"] == 0]

positive_studies = positive_studies.StudyInstanceUID.unique()

positive_studies.shape
# make list of negative PE studies (negative_exam_for_pe = 1)

negative_studies_imgs = train_df.loc[train_df["negative_exam_for_pe"] == 1]

negative_studies_list = negative_studies_imgs.StudyInstanceUID.unique()

negative_studies_list.shape
# count how many positive images there are

positive_images = train_df.loc[train_df["pe_present_on_image"] == 1]

positive_images.shape
negative_studies_imgs.shape
negative_studies_imgs.head()
positive_images.head()
negative_subset = negative_studies_imgs.iloc[0:positive_images.shape[0]]

combined_set = positive_images.append(negative_subset)

random_indexes = np.arange(0,combined_set.shape[0] )

for i in range(3):

    np.random.shuffle(random_indexes)

mixed_set = combined_set.sample(frac=1).reset_index(drop=True)

#mixed_set = combined_set.iloc[random_indexes[0]]

#for i in range(1,20000):

#    mixed_set = mixed_set.append(combined_set.iloc[random_indexes[i]])

#    if( i % 1000 == 0):

#        print("mixed {} records".format(i))

                              

mixed_set.head()

random_indexes.shape[0]
def load_dicom_array_and_sort(dcm_path):

    dicoms = [pydicom.dcmread(file) for file in dcm_path]

    M = float(dicoms[0].RescaleSlope)

    B = float(dicoms[0].RescaleIntercept)

    # Assume all images are axial

    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]

    dicoms = np.asarray([d.pixel_array for d in dicoms])

    dicoms = dicoms[np.argsort(z_pos)]

    dicoms = dicoms * M

    dicoms = dicoms + B

    return dicoms, np.asarray(dcm_path)[np.argsort(z_pos)]





def load_dicom_array(dcm_path):

    dicoms = [pydicom.dcmread(file) for file in dcm_path]

    M = float(dicoms[0].RescaleSlope)

    B = float(dicoms[0].RescaleIntercept)

    # Assume all images are axial

    #z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]

    dicoms = np.asarray([d.pixel_array for d in dicoms])

    #dicoms = dicoms[np.argsort(z_pos)]

    dicoms = dicoms * M

    dicoms = dicoms + B

    return dicoms, np.asarray(dcm_path)

def CT_window(img, WL=50, WW=350):

    upper, lower = WL+WW//2, WL-WW//2

    X = np.clip(img.copy(), lower, upper)

    X = X - np.min(X)

    X = X / np.max(X)

    #X = (X*255.0).astype('uint8')

    return X
# Define the TFExample Data type for training models

# Our TFRecord format will include the CT Image and metadata of the image, including the prediction label (is PE present)



import tensorflow as tf





PE_WINDOW_LEVEL = 100

PE_WINDOW_WIDTH = 700



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



def create_tfrecord( images_array, image_file_names, output_path):

    num_records = images_array.__len__()

    total_records = 0

    #opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    opts = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(output_path, opts) as writer:

        for index in range(num_records):

            img_file_name = image_file_names[index]

            img_file_name = img_file_name.split("/")[-1]

            img_name = img_file_name.split(".")[0]

            img_data = train_df.loc[train_df["SOPInstanceUID"] == img_name]

            pred_label = img_data["pe_present_on_image"].values[0]

            study_id = img_data["StudyInstanceUID"].values[0]

            # the line below write the original CT image

            #tf_example = image_example(images_array[index], study_id, img_name, pred_label)

            # the 2 lines below apply a PE Window function prior to writing the image

            windowed_image = CT_window(images_array[index], PE_WINDOW_LEVEL, PE_WINDOW_WIDTH)

            tf_example = image_example(windowed_image, study_id, img_name, pred_label)

            writer.write(tf_example.SerializeToString())

            total_records = total_records + 1

            print("*",end='')

            #print("wrote {}".format(img_name))

        writer.close()

        

    print("wrote {} records".format(total_records))

    return total_records



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





def read_tf_dataset(storage_file_path):

    encoded_image_dataset = tf.data.TFRecordDataset(storage_file_path, compression_type="GZIP")

    parsed_image_dataset = encoded_image_dataset.map(_parse_image_function)

    return parsed_image_dataset
import shutil



#mixed_set, random_indexes



def write_tfrecord_parts( image_data, output_path, file_prefix, number_dirs, records_per_dir, parts_per_record ):

    for dir_number in range(number_dirs):

        print('working on directory number {}'.format(dir_number))

        dir_path = output_path+'dir{}/'.format(dir_number)

        # create directory

        if os.path.exists(dir_path):

            shutil.rmtree(dir_path)

        os.mkdir(dir_path)

        for part_number in range(records_per_dir):

            print("working on part {}".format(part_number))

            dataset_file_path = dir_path+file_prefix+'dir{}_part{}.tfrecords'.format(dir_number,part_number)

            lower_range = part_number * parts_per_record

            upper_range = lower_range + parts_per_record

            image_set = mixed_set[lower_range:upper_range]

            dicom_images, dicom_image_file_paths = load_dicom_array(image_set.dcm_path)

            num_records = create_tfrecord( dicom_images, dicom_image_file_paths, dataset_file_path)

    

output_path = '/kaggle/working/'

file_prefix = '/pe_window_shuffled_'



write_tfrecord_parts( mixed_set, output_path, file_prefix, 5, 40, 50)
!ls -l '/kaggle/working/dir0'