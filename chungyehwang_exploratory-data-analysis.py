# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!ls -lah ../input
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pydicom
train_labels = pd.read_csv('../input/stage_1_train_labels.csv')
train_labels.info()
train_labels.head(10)
dcm_data = pydicom.read_file('../input/stage_1_train_images/' + train_labels['patientId'][4]+ '.dcm')
print(dcm_data)
fig, m_axs = plt.subplots(1, 1)
m_axs.imshow(dcm_data.pixel_array, cmap='bone')
m_axs.add_patch(Rectangle(xy=(train_labels['x'][4], train_labels['y'][4]),
                                width=train_labels['width'][4],
                                height=train_labels['height'][4], 
                                 alpha = 0.5))
