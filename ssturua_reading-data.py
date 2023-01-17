# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



from PIL import Image

import matplotlib.pyplot as plt



img_dir = '../input/myautoge-cars-dataset/images/images'

df_dir = '../input/myautoge-cars-dataset/characteristics.csv'

# Any results you write to the current directory are saved as output.
df = pd.read_csv(df_dir)
df.head()
def get_images(auto_id):

    auto_dir = os.path.join(img_dir, auto_id)

    if os.path.exists(auto_dir):

        path, dirs, files = next(os.walk(auto_dir))

        plt.figure(figsize=(15,15))

        for i, pic in enumerate(files):

            img_path = os.path.join(auto_dir, pic)

            img_array = np.array(Image.open(img_path))

            plt.subplot(5, 3, i+1)

            plt.xticks([])

            plt.yticks([])

            plt.imshow(img_array)

        plt.show()

    else:

        print('No pictures found')
get_images('1723')
get_images('123')