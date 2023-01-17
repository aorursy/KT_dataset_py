import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *

import matplotlib.pyplot as plt

from IPython.display import Image



import os

print(os.listdir('../input'))



PATH = '../input/leapgestrecog/leapGestRecog/'

print(os.listdir(PATH))
os.listdir(f'{PATH}/00/01_palm/frame_00_01_0045.png')
f'{PATH}/00/01_palm/frame_00_01_0045.png'
Image(filename=f'{PATH}/00/01_palm/frame_00_01_0045.png')