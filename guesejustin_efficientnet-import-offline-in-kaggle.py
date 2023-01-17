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



# Any results you write to the current directory are saved as output.
import sys

package_path = '/kaggle/input/efficientnet100minimal/'

sys.path.append(package_path)
import efficientnet.keras as efn 

model = efn.EfficientNetB2(include_top=False, input_shape=(128,128,3), weights=None, pooling='avg')

# now this would usually download the weights, but because this is offline we will import 

# the weights from another datasource

# just be sure to add the matching b0 to b7 number, depending on which model you started above

model.load_weights('../input/efficientnetb0b7-keras-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')