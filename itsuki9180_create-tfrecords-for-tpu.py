# LOAD LIBRARIES

import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf, re, math

import glob
# If you want to make full dataset, set TEST = 0.

# Full dataset exceeds Kaggle notebook capacity.

# So you should use local environment.

TEST = 1
IMG_DIR = '../input/rsna-str-pe-detection-jpeg-256/train-jpegs'
df = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/train.csv')

df_test = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv')
df.head()
df_test.head()
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
def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12):

  feature = {

      'image': _bytes_feature(feature0),

      'StudyInstanceUID': _bytes_feature(feature1),

      'SeriesInstanceUID':  _bytes_feature(feature2),

      'SOPInstanceUID': _bytes_feature(feature3),

      'negative_exam_for_pe':_float_feature(feature4),

      'rv_lv_ratio_gte_1':_float_feature(feature5),

      'rv_lv_ratio_lt_1':_float_feature(feature6),

      'leftsided_pe':_float_feature(feature7),

      'chronic_pe':_float_feature(feature8),

      'rightsided_pe':_float_feature(feature9),

      'acute_and_chronic_pe':_float_feature(feature10),

      'central_pe':_float_feature(feature11),

      'indeterminate':_float_feature(feature12),

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
IMGS  =  glob.glob(IMG_DIR+os.sep+"/*")



stID = df['StudyInstanceUID'].unique()

p_end=0

p_start= 0

start = 29

end = 29

flg = False

for p in range(251):

    if p==5 and TEST:

        break

    start=p*29

    end = (p+1)*29

    print(start,"to",end)

    length = 0

    for i in range(start, end, 1):

        pat = df.query('StudyInstanceUID == "{}"'.format(stID[i]))

        length += len(pat)

        

    with tf.io.TFRecordWriter('train-{}-{}.tfrec'.format(p,length)) as writer:

        for i in range(start, end, 1):

            pat = df.query('StudyInstanceUID == "{}"'.format(stID[i]))

            p_end += len(pat)

            #print((stID[i]),p_start,p_end)

            for j in range(p_start,p_end):

                path = glob.glob(f'{IMG_DIR}/{df["StudyInstanceUID"][j]}/{df["SeriesInstanceUID"][j]}/*_{df["SOPInstanceUID"][j]}.jpg')

                img = cv2.imread(path[0])

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Fix incorrect colors

                if flg==False:

                    plt.imshow(img/255)

                    plt.show()

                    flg=True

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

                

                example = serialize_example(

                    img,

                    str.encode(df['StudyInstanceUID'][j]),

                    str.encode(df['SeriesInstanceUID'][j]),

                    str.encode(df['SOPInstanceUID'][j]),

                    df['negative_exam_for_pe'][j],

                    df['rv_lv_ratio_gte_1'][j],

                    df['rv_lv_ratio_lt_1'][j],

                    df['leftsided_pe'][j],

                    df['chronic_pe'][j],

                    df['rightsided_pe'][j],

                    df['acute_and_chronic_pe'][j],

                    df['central_pe'][j],

                    df['indeterminate'][j],

                )

                writer.write(example)

            p_start=p_end

    
# All outputs are assembled in "../input/rsnav4".

# You can use this dataset.