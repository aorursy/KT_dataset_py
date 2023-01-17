!ls ../input/tensorflow1131-offline-bert
def setup_tensorflow_1_13():

    

    # Install `tensorflow-gpu==1.13.1` from pre-downloaded wheels

    PATH_TO_TF_WHEELS = '/kaggle/input/tensorflow1131-offline-bert/tensorflow_gpu_1_13_1_with_deps_whl/tensorflow_gpu_1_13_1_with_deps_whl'

    # yes, mixing up Python code and bash is ugly. But it's handy 

    !pip install --no-deps $PATH_TO_TF_WHEELS/*.whl
%%capture

setup_tensorflow_1_13()
import tensorflow as tf

print(tf.__version__)
!ls /kaggle/input/tensorflow1131-offline-bert/bert-tensorflow-1.0.1/bert-tensorflow-1.0.1/
import sys

sys.path.append('/kaggle/input/tensorflow1131-offline-bert/bert-tensorflow-1.0.1/bert-tensorflow-1.0.1/')
from bert import modeling