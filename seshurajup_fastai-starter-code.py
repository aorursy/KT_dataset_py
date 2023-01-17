%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

PATH="../input/iwildcam-2020-fgvc7"

os.listdir(PATH)
import pandas as pd

import numpy as np

from fastai.vision import *

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
! ls '../input/iwildcam-2020-fgvc7/train' | wc
! du -sh '../input/iwildcam-2020-fgvc7/train'
! ls '../input/iwildcam-2020-fgvc7/test' | wc
! du -sh '../input/iwildcam-2020-fgvc7/test'
import numpy as np

from fastai.vision import *

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
path = Path('../input/iwildcam-2020-fgvc7/train')