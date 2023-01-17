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
# libraries

import seaborn as sns



#color

from colorama import Fore, Back, Style



#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



#read the .dcm file

import pydicom
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
# preview the train dataframe

train_df
# preview the train dataframe

test_df
# Check the list of files or folders in the data source

list(os.listdir("../input/siim-isic-melanoma-classification/"))
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission
submission1 = sample_submission

submission1.to_csv('submission1.csv',index = False)
# check if there is missing data in the dataframe

# check the null part in the whole data set, red part is missing data, blue is non-null

sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')

train_df.isnull().sum()
# check the Missing data distribution in train_df

fig = px.scatter(train_df.isnull().sum())



fig.update_layout(

    title="Missing Data in train_df",

    xaxis_title="Columns",

    yaxis_title="Missing data count",

    showlegend=False,

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="RebeccaPurple"

    )

)



fig.show()
# check if there is missing data in the dataframe

# check the null part in the whole data set, red part is missing data, blue is non-null

sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')

test_df.isnull().sum()
# check the Missing data distribution in test_df

fig = px.scatter(test_df.isnull().sum())



fig.update_layout(

    title="Missing Data in test_df",

    xaxis_title="Columns",

    yaxis_title="Missing data count",

    showlegend=False,

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="RebeccaPurple"

    )

)



fig.show()
# Shape of train and test dataframe

print(Fore.RED + 'Training data shape: ',Style.RESET_ALL,train_df.shape)

print(Fore.BLUE + 'Test data shape: ',Style.RESET_ALL,test_df.shape)
# Show the list of columns

columns = train_df.keys()

columns = list(columns)

print(Fore.RED + "List of columns in the train_df",Fore.GREEN + "", columns)
# This dataset has some missing values, which we set to the median of the column for the purpose of this tutorial. 

cleaned_train_df = train_df.dropna()
# check if there is missing data in the dataframe

# check the null part in the whole data set, red part is missing data, blue is non-null

sns.heatmap(cleaned_train_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')

cleaned_train_df.isnull().sum()
# verify if the patient_id is unique for the train_df



print ('Rows in trains_df is', len(train_df))

print ('Number of unique patient id is', train_df['patient_id'].nunique())
# verify if the image_name is unique for the train_df



print ('Rows in trains_df is', len(train_df))

print ('Number of unique patient id is', train_df['image_name'].nunique())
# The Histogram of sex

train_df['sex'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'Sex',linecolor='black',opacity=0.7,color='green',theme='pearl',bargap=0.5,

                                       gridcolor='white',title='Distribution of the Sex column in the Unique Patient Set')
# The Histogram of benign_malignant

train_df['benign_malignant'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'Sex',linecolor='black',opacity=0.7,color='blue',theme='pearl',bargap=0.5,

                                       gridcolor='white',title='Distribution of the Sex column in the Unique Patient Set')

# the 'benign' corresponds to 0 in 'target'.

# The Histogram of target

train_df['target'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'Sex',linecolor='black',opacity=0.7,color='orange',theme='pearl',bargap=0.5,

                                       gridcolor='white',title='Distribution of the Sex column in the Unique Patient Set')
# The Histogram of tadiagnosisrget

train_df['diagnosis'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'Sex',linecolor='black',opacity=0.7,color='red',theme='pearl',bargap=0.5,

                                       gridcolor='white',title='Distribution of the Sex column in the Unique Patient Set')
# The Histogram of anatom_site_general_challenge

train_df['anatom_site_general_challenge'].value_counts().iplot(kind='bar',yTitle='Counts',xTitle = 'Sex',linecolor='black',opacity=0.7,color='purple',theme='pearl',bargap=0.5,

                                       gridcolor='white',title='Distribution of the Position column in the Unique Patient Set')
# https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed

def load_scan(path):

    """

    Loads scans from a folder and into a list.

    

    Parameters: path (Folder path)

    

    Returns: slices (List of slices)

    """

    slices = pydicom.dcmread(path)

    #slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    #slices.sort(key = lambda x: int(x.InstanceNumber))

        

    return slices
# https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed

def get_pixels_hu(scans):

    """

    Converts raw images to Hounsfield Units (HU).

    

    Parameters: scans (Raw images)

    

    Returns: image (NumPy array)

    """

    

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)



    # Since the scanning equipment is cylindrical in nature and image output is square,

    # we set the out-of-scan pixels to 0

    image[image == -2000] = 0

    

    

    # HU = m*P + b

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
INPUT_FOLDER = '/kaggle/input/siim-isic-melanoma-classification/train/'



pictures = os.listdir(INPUT_FOLDER)

pictures.sort()

pictures[0]
test_patient_scans = load_scan(INPUT_FOLDER + pictures[0])
test_patient_scans.dir()
test_patient_scans.PixelData
test_patient_images = get_pixels_hu(test_patient_scans)
plt.imshow(test_patient_scans.PixelData, cmap='gray')

plt.title("Original Slice")

plt.show()