import os

import numpy as np 

import pandas as pd 

# List files available

list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))



train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')# read csv 

train_df.head(5)#display first 5 entries
#let us look at the composition of our dataset, features and datatypes

train_df.info()



#we can see we have 7 columns and 1549 entries. in our training set
#lets also look at the test/ dataset

test_df.info()

#our test dataset is reletively small. 





#we have data about 1549 entries in our training data and, 5 e entries our test.
#look at how many people in our dataset smoke, smoked before or never smoked.

train_df.groupby(['SmokingStatus']).count()['Patient']

# we can see most of our patients ahve smoked before. 
train_df.isnull().sum(),test_df.isnull().sum()



# we do not have any null values in our data
len(train_df['Patient'].unique()) # in our training set we have 176 unique/individual patients, 



# this means we have ongoing or progress information about patients. 

# we can say that roughly, every patient has about 9 entries in our data, 

# patients who have been sufferent for longer are likely to have more entries in our data 
len(test_df['Patient'].unique())# all the entries in our test set are unique
#train data 

files = folders = 0

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)

files,folders
files = folders = 0

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)

files,folders
df=train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()
import plotly.express as px



fig = px.histogram(df, x="SmokingStatus",title="distribution of smoking status in our dataset")

fig.show()

df["Sex"].value_counts()

#in our unique dataset this is the gender distribution that we have



#we can see that the number adds up to 176, ie the number of unique patients



fig = px.histogram(train_df, x="Weeks",title="distribution of weeks",)

fig.show()



fig = px.histogram(train_df, x="Age",title="distribution of age",)

fig.show()

train_df['Age'].max(),train_df['Age'].min()


fig = px.scatter(train_df, x="Weeks", y="Age", color='Sex')

fig.show()



fig = px.scatter(train_df, x="Weeks", y="Age", color='SmokingStatus')

fig.show()

fig = px.histogram(train_df, x="FVC",title="distribution of FVC score",)

fig.show()
fig = px.scatter(train_df, x="FVC", y="Percent", color='Age')

fig.show()
fig = px.scatter(train_df, x="FVC", y="Age", color='Percent')

fig.show()

fig = px.scatter(train_df, x="FVC", y="Age", color='Sex')

fig.show()
patients= train_df.Patient.unique()

patients[10],




patient = train_df[train_df.Patient.isin([patients[10],patients[5],patients[2],patients[175]])  ]



fig = px.line(patient, x="Weeks", y="FVC", color='Patient')

fig.show()



fig = px.violin(train_df, y="Percent", color="SmokingStatus",

                violinmode='overlay',)

fig.show()
fig = px.violin(train_df, y="FVC", color="SmokingStatus",

                violinmode='overlay',)

fig.show()
train_df[train_df['SmokingStatus'] == 'Never smoked']

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="darkgrid")



plt.figure(figsize=(16, 6))

sns.kdeplot(train_df.loc[train_df['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(train_df.loc[train_df['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(train_df.loc[train_df['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');



import os

len(os.listdir('../input/osic-pulmonary-fibrosis-progression/train')),len(os.listdir('../input/osic-pulmonary-fibrosis-progression/test'))

#we have 176 folders containing each patients lung images and we have 5 folders in test dir

import matplotlib.pyplot as plt

%matplotlib inline

import pydicom # to view dicom images



imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"

print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))



#set grid

# view first (columns*rows) images in order

fig=plt.figure(figsize=(12, 12))

columns = 3

rows = 3

imglist = os.listdir(imdir)# list of files inside ID00123637202217151272140 directory





for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    #eg, train/ID00123637202217151272140/1.dcm

    

    #read the file

    ds = pydicom.dcmread(filename)

    

    fig.add_subplot(rows, columns, i)# add space for figure at correct location 

    plt.imshow(ds.pixel_array, cmap='RdBu')# add file to the specific spot in fig

plt.show()#show final figure




import imageio

from IPython.display import Image



import os

import pydicom as dicom

import glob



apply_resample = False



def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)



def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg

import numpy as np

from scipy.ndimage.interpolation import zoom





scans = load_scan('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')

scan_array = set_lungwin(get_pixels_hu(scans))





imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')
import matplotlib.pyplot as plt 

plt.imshow(scan_array[5], animated=True, cmap="gist_rainbow_r")



# scan_array is the array object that contains all the images in sequence 
import matplotlib.animation as animation



fig = plt.figure()



ims = [] # list to store imshow renders



for image in scan_array:

    im = plt.imshow(image, animated=True, cmap="gist_rainbow_r") # render immage from arrayas variable im

    plt.axis("off")

    ims.append([im])#add to list of images 



ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,



                                repeat_delay=1000)#create animation using matplotlib.animation
from IPython.display import HTML # library required to display this animation

HTML(ani.to_html5_video())
patients[15]


scans = load_scan('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00035637202182204917484/')

scan_array = set_lungwin(get_pixels_hu(scans))



fig = plt.figure()



ims = [] # list to store imshow renders



for image in scan_array:

    im = plt.imshow(image, animated=True, cmap="mako") # render immage from arrayas variable im

    plt.axis("off")

    ims.append([im])#add to list of images 



ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,



                                repeat_delay=1000)#create animation using matplotlib.animation



HTML(ani.to_html5_video())#display as html5 video
imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')