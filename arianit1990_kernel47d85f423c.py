%reset -sf

import random

import collections

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from tqdm import tqdm



random.seed(42)

!ls /kaggle/input/hashcode-photo-slideshow/
!head -3 "/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt"
!head "/kaggle/input/d-output/d_example_output_11.txt"
f = open("/kaggle/input/d-output/d_example_output_11.txt", "r")

content = f.read()

f2 = open("submission.txt", "w")

f2.write(content)

f2.close()