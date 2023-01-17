import fastai
fastai.__package__
from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *
%load_ext autoreload

%autoreload 2

%matplotlib inline
path = 'data/mnist/'
import os

os.makedirs(path, exist_ok=True)
URL='http://deeplearning.net/data/mnist/'

FILENAME='mnist.pkl.gz'



def load_mnist(filename):

    return pickle.load(gzip.open(filename, 'rb'), encoding='latin-1')
get_data(URL + FILENAME, path + FILENAME)

((x, y), (x_valid, y_valid), _) = load_mnist(path + FILENAME)