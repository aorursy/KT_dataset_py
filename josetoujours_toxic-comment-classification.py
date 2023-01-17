!pip3 install fairness-indicators==0.1.2 --user
# %tensorflow_version 2.x
import sys, os
import warnings
warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore deprecation warnings
import tempfile
import apache_beam as beam
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators
from tensorflow_model_analysis.addons.fairness.view import widget_view
from fairness_indicators.examples import util

import warnings
warnings.filterwarnings("ignore")

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget


print(tf.version.VERSION) 
print(tf) # This statement shows us what version of Python we are currently running.
download_original_data = False #@param {type:"boolean"}

if download_original_data:
  train_tf_file = tf.keras.utils.get_file('train_tf.tfrecord',
                                          'https://storage.googleapis.com/civil_comments_dataset/train_tf.tfrecord')
  validate_tf_file = tf.keras.utils.get_file('validate_tf.tfrecord',
                                             'https://storage.googleapis.com/civil_comments_dataset/validate_tf.tfrecord')

  # The identity terms list will be grouped together by their categories
  # (see 'IDENTITY_COLUMNS') on threshould 0.5. Only the identity term column,
  # text column and label column will be kept after processing.
  train_tf_file = util.convert_comments_data(train_tf_file)
  validate_tf_file = util.convert_comments_data(validate_tf_file)


else:
  train_tf_file = tf.keras.utils.get_file('train_tf_processed.tfrecord',
                                          'https://storage.googleapis.com/civil_comments_dataset/train_tf_processed.tfrecord')
  validate_tf_file = tf.keras.utils.get_file('validate_tf_processed.tfrecord',
                                             'https://storage.googleapis.com/civil_comments_dataset/validate_tf_processed.tfrecord')
# The computation of statistics using TFDV.  The returned value is a DatasetFeatureStatisticsList protocol buffer. 
stats = tfdv.generate_statistics_from_tfrecord(data_location=train_tf_file)


# A visualization of the statistics using Facets Overview.
tfdv.visualize_statistics(stats)
#@title Calculate label distribution for gender-related examples
raw_dataset = tf.data.TFRecordDataset(train_tf_file)

toxic_gender_examples = 0
nontoxic_gender_examples = 0

# TODO 4

# There are 1,082,924 examples in the dataset
for raw_record in raw_dataset.take(1082924):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  if str(example.features.feature["gender"].bytes_list.value) != "[]":
    if str(example.features.feature["toxicity"].float_list.value) == "[1.0]":
      toxic_gender_examples += 1
    else:
      nontoxic_gender_examples += 1

# TODO 4a

print("Toxic Gender Examples: %s" % toxic_gender_examples)
print("Nontoxic Gender Examples: %s" % nontoxic_gender_examples)