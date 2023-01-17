from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%cd /kaggle/working
!git clone https://github.com/harissetiyono/Mask_RCNN.git
%cd /kaggle/working/Mask_RCNN
!unzip logs.zip -d /kaggle/working/Mask_RCNN
!wget http://harisset.com/files/logs.zip
!ls /kaggle/working/Mask_RCNN/samples/custom/dataset
!pip3 install -r requirements.txt
%cd /kaggle/working/Mask_RCNN/samples/custom
!python palm.py train --dataset=/kaggle/working/Mask_RCNN/samples/custom/dataset --weight=/kaggle/working/Mask_RCNN/logs/palm20200905T0425/mask_rcnn_palm_0002.h5