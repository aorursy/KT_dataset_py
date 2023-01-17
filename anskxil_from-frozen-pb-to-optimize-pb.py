# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
input_node_names = ['input_1']
output_node_name = 'predictions/Softmax'
input_graph_def = tf.GraphDef()
with tf.gfile.Open('../input/tf_cashclassifier_2018_08_10.pb', "rb") as f:
    input_graph_def.ParseFromString(f.read())
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)
with tf.gfile.FastGFile('optimize.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())