!pip install /kaggle/input/tf-nightly-220-dev20200311/h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl

!pip uninstall tensorflow -y

!pip uninstall tensorboard -y

!pip install /kaggle/input/tf-nightly-220-dev20200311/gast-0.3.3-py2.py3-none-any.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/tf_estimator_nightly-2.1.0.dev2020031101-py2.py3-none-any.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/tensorboard_plugin_wit-1.6.0.post2-py3-none-any.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/gviz_api-1.9.0-py2.py3-none-any.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/tb_nightly-2.2.0a20200310-py3-none-any.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/scipy-1.4.1-cp36-cp36m-manylinux1_x86_64.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/astunparse-1.6.3-py2.py3-none-any.whl

!pip install /kaggle/input/tf-nightly-220-dev20200311/tf_nightly-2.2.0.dev20200311-cp36-cp36m-manylinux2010_x86_64.whl
import tensorflow as tf

print(tf.version.VERSION)