# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in sorted(filenames):

        if filename.startswith("test_image_data") or filename.startswith("sample_submission"):

            fullpath = os.path.join(dirname, filename)

            print('{}:{} MB'.format(fullpath, round(os.path.getsize(fullpath) / (1024.0 ** 2), 4)))



# Any results you write to the current directory are saved as output.
submission_data = pd.read_csv("/kaggle/input/bengaliai-cv19/sample_submission.csv")

submission_data
files = [

    "test_image_data_0.parquet",

    "test_image_data_1.parquet",

    "test_image_data_2.parquet",

    "test_image_data_3.parquet",

]





test_image_data_set = []

for file in files:

    test_image_data_set.append(pd.read_parquet('/kaggle/input/bengaliai-cv19/{}'.format(file)))

test_image_data_set[0].head()
test_image_data_set[1].head()
test_image_data_set[2].head()
test_image_data_set[3].head()
IMAGE_ROW = 137 

IMAGE_COLUMN = 236



from matplotlib import pylab as plt



image_set = []

index = -1

row_count = len(test_image_data_set)

plt.figure(figsize=(15,10))



for test_image_data in test_image_data_set:



    index = index + 1

    drop_data = test_image_data.drop('image_id', axis=1)

    column_count = len(test_image_data_set)

    

    for row, item in drop_data.iterrows():

    

        image = item.values.reshape([IMAGE_ROW, IMAGE_COLUMN])

        no = index * column_count + row + 1

        plt.subplot(row_count, column_count, no)

        plt.imshow(image)
