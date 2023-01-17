#import warnings
#warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from collections import Counter
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

train=os.listdir("../input/test/challenge2018_test")
sample_submission=pd.read_csv("../input/sample_submission.csv")
#The Total Number of images in the dataset 
Count=Counter(train)
print(sum(Count.values()))
print(train)
# This is needed to display the images.
%matplotlib inline
from utils import label_map_util

from utils import visualization_utils as vis_util
