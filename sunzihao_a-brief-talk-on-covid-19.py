import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json



import matplotlib.pyplot as plt

plt.style.use('ggplot')



import os



# Any results you write to the current directory are saved as output.
!ls /kaggle/input/CORD-19-research-challenge/
import pandas as pd

# code from Dear MaksimEkin!

root_path = '/kaggle/input/CORD-19-research-challenge'

metadata_path = f'{root_path}/metadata.csv' 

meta_df = pd.read_csv(metadata_path, dtype={ 

    'pubmed_id': str, 

    'Microsoft Academic Paper ID': str, 

    'doi': str 

}) 

meta_df.head()
from IPython.display import Image

Image(filename='/kaggle/input/image1/Pic1.jpg', width=1600) 
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



root_path = '/kaggle/input/CORD-19-research-challenge'

metadata_path = f'{root_path}/metadata.csv' 

dataframe_all=pd.read_csv(metadata_path)

num_rows=dataframe_all.shape[0]



counter_nan=dataframe_all.isnull().sum()

counter_without_nan=counter_nan[counter_nan==0]

dataframe_all=dataframe_all[counter_without_nan.keys()]

dataframe_all=dataframe_all.ix[:,10:]
from IPython.display import Image

Image(filename='/kaggle/input/image2/Pic2.jpg', width=1600) 
from IPython.display import Image

Image(filename='/kaggle/input/image3/Pic3.jpg', width=1600) 
from IPython.display import Image

Image(filename='/kaggle/input/image3/Pic4.png', width=1600) 
from IPython.display import Image

Image(filename='/kaggle/input/image3/Pic5.jpg', width=1600) 
from IPython.display import Image

Image(filename='/kaggle/input/imageimage/2020032221200544.png', width=1600) 