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
!pip install --upgrade fairness-indicators
import os
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

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget
# This code is to find the Path of the File in Kaggle
#import os
#for dirname in os.walk('/kaggle/input'):
 #   print(dirname)

#@title Options for Downloading data

#@markdown You can choose to download the original and process the data in
#@markdown the colab, which may take minutes. By default, we will download the
#@markdown data that we have already prepocessed for you. In the original
#@markdown dataset, for each indentity annotation columns, the value represents
#@markdown the percent of raters who thought the comment references the identity.
#@markdown When processing the raw data, the threshold 0.5 is chosen and the
#@markdown identities are grouped together by their categories. For example
#@markdown if one comment has { male: 0.3, female: 1.0, transgender: 0.0,
#@markdown heterosexual: 0.8, homosexual_gay_or_lesbian: 1.0 }, after the
#@markdown processing, the data will be { gender: [female], 
#@markdown sexual_orientation: [heterosexual, homosexual_gay_or_lesbian] }.
download_original_data = True #@param {type:"boolean"}

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
stats = tfdv.generate_statistics_from_tfrecord(data_location=train_tf_file)
tfdv.visualize_statistics(stats)
BASE_DIR = tempfile.gettempdir()

TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'
FEATURE_MAP = {
    # Label:
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    # Text:
    TEXT_FEATURE:  tf.io.FixedLenFeature([], tf.string),

    # Identities:
    'sexual_orientation':tf.io.VarLenFeature(tf.string),
    'gender':tf.io.VarLenFeature(tf.string),
    'religion':tf.io.VarLenFeature(tf.string),
    'race':tf.io.VarLenFeature(tf.string),
    'disability':tf.io.VarLenFeature(tf.string),
}