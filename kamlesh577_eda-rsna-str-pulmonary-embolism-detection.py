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
import numpy as np

import pydicom

import os

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.transform import resize

from sklearn.cluster import KMeans

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.figure_factory as ff

from plotly.graph_objs import *

init_notebook_mode(connected=True)

import pandas as pd

from tqdm import tqdm

import seaborn as sns
df_train = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv')

df_test = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/test.csv')



PATH = "../input/rsna-str-pulmonary-embolism-detection/"

TRAIN_PATH = PATH + "train/"

TEST_PATH = PATH + "test/"

sub = pd.read_csv(PATH + "sample_submission.csv")

train_image_file_paths = glob(TRAIN_PATH + '/*/*/*.dcm')

test_image_file_paths = glob(TEST_PATH + '/*/*/*.dcm')
df_train.head(5)
df_test.head()
df_train.shape

df_test.shape
sample_submission = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")

sample_submission.head()
x = df_train.pe_present_on_image.value_counts()



x.plot(kind='barh')

#x.label('pe_present_on_image')
# Draw a pie chart about pe_present_on_image.

plt.pie(df_train["pe_present_on_image"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of pe_present_on_image")

plt.show()
# Draw a pie chart about negative_exam_for_pe.

plt.pie(df_train["negative_exam_for_pe"].value_counts(),labels=["0","1"],autopct="%.1f%%")

plt.title("Ratio of negative_exam_for_pe")

plt.show()
x = df_train.negative_exam_for_pe.value_counts()

print(x)

x.plot(kind='barh')
dcm_file
fig, ax = plt.subplots(2,1,figsize=(20,10))

for file in train_image_file_paths[0:10]:

    dataset = pydicom.read_file(file)

    image = dataset.pixel_array.flatten()

    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept

    sns.distplot(image.flatten(), ax=ax[0]);

    sns.distplot(rescaled_image.flatten(), ax=ax[1])

ax[0].set_title("Raw pixel array distributions for 10 examples");
# View the correlation heat map

corr_mat = df_train.corr(method='pearson')

sns.heatmap(corr_mat,

            vmin=-1.0,

            vmax=1.0,

            center=0,

            annot=True, # True:Displays values in a grid

            fmt='.1f',

            xticklabels=corr_mat.columns.values,

            yticklabels=corr_mat.columns.values

           )

plt.show()