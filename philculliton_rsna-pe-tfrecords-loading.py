# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import tensorflow as tf



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
UNLABELED_TFRECORD_FORMAT = {'SpecificCharacterSet': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ImageType': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SOPClassUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SOPInstanceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Modality': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SliceThickness': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'KVP': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'GantryDetectorTilt': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'TableHeight': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'RotationDirection': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'XRayTubeCurrent': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Exposure': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ConvolutionKernel': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PatientPosition': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'StudyInstanceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SeriesInstanceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SeriesNumber': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'InstanceNumber': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ImagePositionPatient': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ImageOrientationPatient': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'FrameOfReferenceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SamplesPerPixel': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PhotometricInterpretation': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Rows': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Columns': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PixelSpacing': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'BitsAllocated': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'BitsStored': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'HighBit': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PixelRepresentation': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'WindowCenter': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'WindowWidth': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'RescaleIntercept': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'RescaleSlope': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None)}
LABELED_TFRECORD_FORMAT = {'SpecificCharacterSet': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ImageType': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SOPClassUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SOPInstanceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Modality': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SliceThickness': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'KVP': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'GantryDetectorTilt': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'TableHeight': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'RotationDirection': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'XRayTubeCurrent': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Exposure': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ConvolutionKernel': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PatientPosition': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'StudyInstanceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SeriesInstanceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SeriesNumber': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'InstanceNumber': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ImagePositionPatient': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'ImageOrientationPatient': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'FrameOfReferenceUID': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'SamplesPerPixel': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PhotometricInterpretation': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Rows': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'Columns': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PixelSpacing': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'BitsAllocated': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'BitsStored': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'HighBit': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'PixelRepresentation': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'WindowCenter': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'WindowWidth': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'RescaleIntercept': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'RescaleSlope': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),

 'negative_exam_for_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'qa_motion': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'qa_contrast': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'flow_artifact': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'rv_lv_ratio_gte_1': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'rv_lv_ratio_lt_1': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'leftsided_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'chronic_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'true_filling_defect_not_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'rightsided_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'acute_and_chronic_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'central_pe': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'indeterminate': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'pe_present_on_image': tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

 'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=None)}
def read_labeled_tfrecord(example):

    return read_tfrecord(example, LABELED_TFRECORD_FORMAT)



def read_unlabeled_tfrecord(example):

    return read_tfrecord(example, UNLABELED_TFRECORD_FORMAT)



def read_tfrecord(example, record_format):

    try:

        example = tf.io.parse_single_example(example, record_format)

    except:

        print (example)

        raise

    

    data = {k:tf.cast(example[k], record_format[k].dtype) for k in example}

        

    return data
def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset
train_dataset = load_dataset(tf.io.gfile.glob('/kaggle/input/rsna-pe-tfrecords-v2/train/*.tfrec'), labeled=True)

test_dataset = load_dataset(tf.io.gfile.glob('/kaggle/input/rsna-pe-tfrecords-v2/test/*.tfrec'), labeled=False)
# from kaggle_datasets import KaggleDatasets



# GCS_DS_PATH = KaggleDatasets().get_gcs_path("rsna-pe-tfrecords-v2") # you can list the bucket with "!gsutil ls $GCS_DS_PATH"



# train_dataset = load_dataset(tf.io.gfile.glob(GCS_DATA_PATH + '/train/*.tfrec'), labeled=True)

# test_dataset = load_dataset(tf.io.gfile.glob(GCS_DATA_PATH + '/test/*.tfrec'), labeled=False)
def plot_img(im):

    plt.figure(figsize=(10,10))

    ax = plt.subplot(1,2,1)

    plt.imshow(im)
train_fnames = []



for index, image_features in enumerate(train_dataset.as_numpy_iterator()):

    ## decoding the byte string all DICOM fields are stored in:

    image_name = image_features["SOPInstanceUID"].decode("utf-8")

    train_fnames.append(image_name)

    

    ## decoding and checking our image data. Note that we're ONLY reading the first five entries.

    if index < 5:

        image = np.frombuffer(image_features["image"], dtype=np.int16).reshape((512,512))

        plot_img(image)

    else:

        break
test_fnames = []



for index, image_features in enumerate(test_dataset.as_numpy_iterator()):

    ## decoding the byte string all DICOM fields are stored in:    

    image_name = image_features["SOPInstanceUID"].decode("utf-8")

    test_fnames.append(image_name)

    

    ## decoding and checking our image data. Note that we're ONLY reading the first five entries.

    if index < 5:

        image = np.frombuffer(image_features["image"], dtype=np.int16).reshape((512,512))

        plot_img(image)

    else:

        break