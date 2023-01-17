# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

%matplotlib inline

import  glob, cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
submission = pd.read_csv('/kaggle/input/recunoasterea-scris-de-mana/sampleSubmission.csv')#

submission.head()
train = pd.read_csv('/kaggle/input/recunoasterea-scris-de-mana/train.csv')

train.head()
train['label'].hist()
test_img = glob.glob('/kaggle/input/recunoasterea-scris-de-mana/test/test/*')

train_img = glob.glob('/kaggle/input/recunoasterea-scris-de-mana/train/train/*')

len(test_img), len(train_img)
#build here the model, train and validate on training images,  then predict on test images
#save the prediction

submission['Expected'] = 0

submission.to_csv('submission.csv', index=False)