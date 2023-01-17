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
!pip install --upgrade tensorflow==1.13.1
import tensorflow as tf
tf.__version__
!python /kaggle/input/facenet/facenet/src/train_tripletloss.py --max_nrof_epochs=3 --epoch_size=300 --people_per_batch=4 --images_per_person=30 --models_base_dir=/kaggle/working/model --data_dir=/kaggle/input/facestar/face/train-format-160