!pip install -q tensorflow-datasets
# adapted from https://www.tensorflow.org/datasets/add_dataset



import tensorflow_datasets.public_api as tfds



class MyDataset(tfds.core.GeneratorBasedBuilder):

  """Short description of my dataset."""



  VERSION = tfds.core.Version('0.1.0')



  def _info(self):

    # metadata goes here

    pass # TODO



  def _split_generators(self, dl_manager):

    # specifies what data goes into train, validation, test (or whatever)

    pass  # TODO



  def _generate_examples(self):

    # Yields examples from the dataset

    yield 'key', {}
from dataset import MyDataset

from kaggle_datasets import KaggleDatasets



if tpu:

    DATA_DIR = KaggleDatasets().get_gcs_path('my-dataset')

else:

    DATA_DIR = '/kaggle/input/my-dataset'
ds = tfds.load('my_dataset', split=['train', 'test'], data_dir=DATA_DIR)