# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.2
import pandas as pd

sample_submission = pd.read_csv("../input/sample_submission.csv")
# SKLearn Implemention

from sklearn.metrics import log_loss

log_loss(["REAL", "FAKE", "FAKE", "REAL"],

         [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import cv2

plt.style.use('ggplot')

from IPython.display import Video

from IPython.display import HTML