%matplotlib inline

import os, shutil # directory operations
import numpy as np # linear algebra 
import pandas as pd # I/O of data
from zipfile import ZipFile # woking with zip archives

# packages for visualization
import ipywidgets as iw
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
zp_filenames = ['Testing.zip','Training.zip','Validation.zip']
# using list comprehension to build a list for zipped file with their path names.
zp_filenamesWithpath = [os.path.join('../input/kaggle-for-deep-learning-p-1-getting-data-ready',k) for k in zp_filenames] 
for k in zp_filenamesWithpath: # looping over the files that need to be unzipped
    
    # extracting the zipfiles to the current directory
    with ZipFile(k,'r') as zp: 
        zp.extractall()
f_names = [k for k in os.listdir('working/Training')] # list of fruit names

dp = iw.Dropdown(options = f_names, 
                 description = 'Select fruit') # creating a dropdown widgets for fruit names

path4 = '../input/fruits/fruits-360_dataset/fruits-360/Training' # path to training images

# displaying the dropdown widget
display(dp)
out = iw.Output()
display(out)

# defining a function which is activated when the drop down value is changed
def on_index_change(change):
    
    out.clear_output() # first let's clear the output if there is any
    
    f_dir = f_names[change.new] # assigning the new fruit name to f_dir
    
    num_imgs = len(os.listdir(os.path.join(path4, f_dir))) - 1 # calculating the number of images in the new fruit directory 
    path2image = os.path.join(path4, f_dir) # getting the path to the new fruit directory
    
    # defining a function which will be used for the slider widget. This function shows an image based on scroll value.
    def showfruit(vchange):
        
        fig = plt.figure(figsize = (5, 5)) # initializing the figure
        
        im2show = os.listdir(path2image)[vchange] # storing the name of image to be shown
        path5 = os.path.join(path2image,im2show) # storing the complete path to the image to be shown
        
        img = plt.imread(path5) # reading the image to be shown
        
        plt.imshow(img) # displaying the image
        plt.axis('off') # turning the axis off
    
    # making a slider widget which uses the function showfruit to plot fruit images
    with out:
        k = iw.interactive(showfruit, vchange =  iw.IntSlider(description = 'Scroll images',
                                                              min = 0,
                                                              max = num_imgs,
                                                              readout = False))
        
        # displaying the slider widget
        display(k)

# observing if there is a change to the dropdown widget index
dp.observe(on_index_change, names='index')
# initializing an array to store resized images
imgs = np.empty([75,32,32,3])

# looping over folders in the training directory
for i, folder in enumerate(os.listdir(path4)):
    
    # selecting first image in the folder
    first_image = (os.listdir(os.path.join(path4,folder)))[0]
    
    # reading the image
    img = plt.imread(os.path.join(path4,folder + '/' + first_image))
    
    # resizing the image
    imgs[i,:,:,:] = imresize(img, (32, 32, 3), mode = 'constant')

# creating 75 subplots with 13 rows and 5 columns
figure, ax = plt.subplots(13,5,figsize = (20, 20))

# initializing count variable i

i = 0

# looping over subplot rows
for k in range(13):
    
    # looping over subplot columns
    for j in range(5):
        # plotting image
        ax[k,j].imshow(imgs[i])
        
        #turning axis off
        ax[k,j].axis('off')
        
        # updating count variable
        i += 1
path1 = 'working/Training' # path to the training folder where folders with fruit names are stored
path2 = 'working/Validation' # path to the validation folder where folders with fruit names are stored
path3 = 'working/Test' # path to the test folder where folders with fruit names are stored
f_names = [k for k in os.listdir(path1)] # list of fruit names
# initializing empty dictionaries to contain number of fruits
count_train = {}
count_test = {}
count_validation = {}
count_total = {}

# looping over the list of fruit names
for k in f_names:
    
    # storing the number of images in each class to different folders
    count_train[k] = len(os.listdir(os.path.join(path1,k)))
    count_test[k] = len(os.listdir(os.path.join('working/Test',k)))
    count_validation[k] = len(os.listdir(os.path.join('working/Validation',k)))

# Assigning the number of fruits in different sets to one dictionary
count_total['Test'] = count_test
count_total['Training'] = count_train
count_total['Validation'] = count_validation

# Storing the dictionary to a data frame, df
df = pd.DataFrame.from_dict(count_total)
df.head()
# Calculating the number of images in each set and the number of classes. Storing these values in a dictionary
cnn_params = {}
cnn_params['No training images'] = np.sum(df.Training)
cnn_params['No validation images'] = np.sum(df.Validation)
cnn_params['No test images'] = np.sum(df.Test)
cnn_params['No classes'] = len(df.Test)

# Printing the number of images in each set and the number of classes.
print('Number of training images = {} \n'.format(cnn_params['No training images']))
print('Number of validation images = {} \n'.format(cnn_params['No validation images']))
print('Number of test images = {} \n'.format(cnn_params['No test images']))
print('Number of classes = {} \n'.format(cnn_params['No classes']))
# batch sizes to search from
potential_batch_sizes = [k for k in range(8,129)]

# dividing the total number of images by potential_batch_sizes are calculating the remainder
remaining_images_training = list(zip(potential_batch_sizes, cnn_params['No training images']%potential_batch_sizes))
remaining_images_validation = list(zip(potential_batch_sizes, cnn_params['No validation images']%potential_batch_sizes))

# printing the potential_batch_sizes with their respective remainders
print('Batch size and corresponding remaining training images :\n {}'.
      format(remaining_images_training)) 
print('Batch size and corresponding remaining validation images :\n {}'.
      format(remaining_images_validation)) 
cnn_params['Training batch size'] = 18
cnn_params['Validation batch size'] = 17
cnn_params['Training steps/epoch'] = int(cnn_params['No training images'] / cnn_params['Training batch size'])
cnn_params['Validation steps/epoch'] = int(cnn_params['No validation images'] / cnn_params['Validation batch size'])
# Storing the dictionary to a data frame, cnn_params
df_cnn_params = pd.DataFrame.from_dict(cnn_params, orient='index')
df_cnn_params.to_csv('CNNParameters.csv')
# making the bar plot
df['Training'].plot(kind='bar', figsize=(20, 10), legend=False)

# labeling the axis and setting a title
plt.ylabel('Number of images')
plt.title('Training set')
plt.tight_layout()
# making the bar plot
df['Validation'].plot(kind='bar', figsize=(20, 10), legend=False)

# labeling the axis and setting a title
plt.ylabel('Number of images')
plt.title('Validation set')
plt.tight_layout()
# making the bar plot
df['Test'].plot(kind='bar', figsize=(20, 10), legend=False)

# labeling the axis and setting a title
plt.ylabel('Number of images')
plt.title('Test set')
plt.tight_layout()
# Removing extracted images folders
shutil.rmtree('working')