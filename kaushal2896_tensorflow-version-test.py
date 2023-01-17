!pip uninstall tensorflow -y
!pip install tensorflow-gpu==1.15
import tensorflow

print(tensorflow.__version__)

import keras

print(keras.__version__)
tensorflow.test.is_gpu_available()