# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install efficientnet

import tensorflow as tf 

import efficientnet.tfkeras as efn

import tensorflow.keras.backend as K



# !pip install -q efficientnet
x = efn.EfficientNetB0(include_top=False, weights='noisy-student', pooling=max)

x.save('EfficientNetB0.h5')
x1 = efn.EfficientNetB1(include_top=False, weights='noisy-student', pooling=max)

x1.save('EfficientNetB1.h5')
x2 = efn.EfficientNetB2(include_top=False, weights='noisy-student', pooling=max)

x2.save('EfficientNetB2.h5')
x3 = efn.EfficientNetB3(include_top=False, weights='noisy-student', pooling=max)

x3.save('EfficientNetB3.h5')
x4 = efn.EfficientNetB4(include_top=False, weights='noisy-student', pooling=max)

x4.save('EfficientNetB4.h5')
x5 = efn.EfficientNetB5(include_top=False, weights='noisy-student', pooling=max)

x5.save('EfficientNetB5.h5')
x6 = efn.EfficientNetB6(include_top=False, weights='noisy-student', pooling=max)

x6.save('EfficientNetB6.h5')
x7 = efn.EfficientNetB7(include_top=False, weights='noisy-student', pooling=max)

x7.save('EfficientNetB7.h5')