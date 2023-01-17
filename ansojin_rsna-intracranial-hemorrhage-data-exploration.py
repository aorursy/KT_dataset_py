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



# Any results you write to the current directory are saved as output.
import glob, pylab, pandas as pd

import pydicom, numpy as np

from os import listdir

from os.path import isfile, join

import matplotlib.pylab as plt

import os

import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
train_df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

train_df.head()
print("Training Dataset's shape:", train_df.shape)
train_df['Label'].value_counts()
sns.countplot(train_df.Label)


train_df['Sub_type'] = train_df['ID'].str.split("_", n = 3, expand = True)[2]

train_df['Img_file_name'] = train_df['ID'].str.rsplit("_", n =1, expand = True)[0]

train_df.head()

sub_type_summary = train_df.groupby('Sub_type').sum()

sub_type_summary
sns.set(rc={'figure.figsize':(8,8)})

plot = sns.barplot(x=sub_type_summary.index, y= sub_type_summary.Label)



plt.xticks(rotation=45)



fig=plt.figure(figsize=(8, 8))



sns.countplot(x="Sub_type", hue="Label", data=train_df, palette="deep")

plt.xticks(rotation=45)

plt.title("Total Images by Subtype")
#train_df.to_csv('subtype_train.csv')

#from IPython.display import FileLink, FileLinks

# FileLink('subtype_train.csv')
# Train/Test Image Files Overview



train_imgs = sorted(glob.glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm"))

test_imgs = sorted(glob.glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/*.dcm"))

print("total Train DICOM images: ", len(train_imgs))

print("Total Test DICOM images: ", len(test_imgs))
plot_dicom_all=plt.figure(figsize=(15, 15))

plt.title('DICOM Images Overview')

plt.rcParams["axes.grid"] = False

columns = 4; rows = 4

for i in range(1, columns*rows +1):

    dicom_img = pydicom.dcmread(train_imgs[i])

    plot_dicom_all.add_subplot(rows, columns,i)

    plt.imshow(dicom_img.pixel_array, cmap=plt.cm.bone)

    plot_dicom_all.add_subplot

    

    

# Meta data structure

pydicom.dcmread(train_imgs[0])
def show_dicom_metadata(filename):

    """

    show_dicom_metadata function is to get all important DICOM metadata, such as windowing parameters and other information and also plot it

    input parameter:

    filename: string, DICOM filename

    """



    dataset = pydicom.dcmread(filename)

    # Normal mode:

    print()

    print("Filename.........:", filename)

    print("Storage type.....:", dataset['SOPInstanceUID'])

    print("Patient id.......:", dataset.PatientID)

    print("Modality.........:", dataset.Modality)

    print()

    print("Window Center.........:", dataset.WindowCenter)

    print('Window Width.........:',dataset.WindowWidth)

    print('Rescale Intercept.........:',dataset.RescaleIntercept)

    print('Rescale Slope.........:',dataset.RescaleSlope)





    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

                        rows=rows, cols=cols, size=len(dataset.PixelData)))

    if 'PixelSpacing' in dataset:

        print("Pixel spacing....:", dataset.PixelSpacing)



    # use .get() if not sure the item exists, and want a default value if missing

    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))



    # plot the image using matplotlib

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()
show_dicom_metadata(train_imgs[0])
def window_image(img, window_center,window_width, intercept, slope):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img 



def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

    
base_path = '../input/rsna-intracranial-hemorrhage-detection/'

stage_1_train_images_path  = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'







def display_dicom_image(df, sub_type, column_number,row_number):

    

    """

    display_dicom_image function shows the DICOM imgae from the training dataset dataframe.

    df: data frame that includes the images and subtype information

    sub_type: string, what sub_type want to show

    column_number: int, how many images in a row

    row_number: int, how many rows want to show



    """

    # print(sub_type)

    if sub_type not in ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']:

        print('No this Type:',sub_type)

        return   

    

    images = df[(df['Sub_type'] == sub_type) & (train_df['Label'] == 1)][:(column_number*row_number)].Img_file_name.values

    

    fig, axs = plt.subplots(row_number, column_number, figsize=(15,15))

    

    

    for im in range(0, column_number*row_number):

        # print(images[im])

        # print(os.path.join(stage_1_train_images_path,images[im]+ '.dcm'))

        data = pydicom.read_file(os.path.join(stage_1_train_images_path,images[im]+ '.dcm'))

        

        image = data.pixel_array

        window_center , window_width, intercept, slope = get_windowing(data)

        image_windowed = window_image(image, window_center, window_width, intercept, slope)





        i = im // column_number

        j = im % column_number

        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

       

    plt.suptitle('Images of Hemorrhage Sub-type:' + sub_type )

    plt.show()

display_dicom_image(train_df, 'any', 4,4)
display_dicom_image(train_df, 'intraventricular', 4,4)
display_dicom_image(train_df, 'epidural', 4,4)
display_dicom_image(train_df, 'subarachnoid', 4,4)
display_dicom_image(train_df, 'intraventricular', 4,4)