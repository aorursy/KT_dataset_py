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
import os

import cv2

import numpy as np

from sklearn.utils import shuffle



Images = []

Labels = []

directory = 'C:/CNN/datasets/alien_vs_predator_thumbnails/data/train/' # change your directory, # don't forget last '/'



for label, names in enumerate(os.listdir(directory)):

    try:

        for image_file in os.listdir(directory+names):

            image = cv2.imread(directory+names+r'/'+image_file)

            image = cv2.resize(image,(150,150))

            Images.append(image)

            Labels.append(label)



    except Exception as e:

        print(str(e))



shuffle(Images,Labels, random_state=5) # your choice



Images = np.array(Images)

Labels = np.array(Labels)



file_names = 'alien_predator_train' # change name you want

Save = np.savez(directory+file_names, x=Images, y=Labels)