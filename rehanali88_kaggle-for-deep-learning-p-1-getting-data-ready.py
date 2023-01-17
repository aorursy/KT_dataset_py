%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # I/O of data
import matplotlib.pyplot as plt # making plots

import os, random, shutil, zlib # directory operations
 # print the path of the current directory
print('Current directory is {}'.format(os.getcwd()))

# print the contents of the current directory
print('Current directory contains the follwoing sub-directories:\n {}'.format(os.listdir())) 
# print the current directory
print('Kaggle directory contains the following sub-directories:\n {}'.format(os.listdir('../'))) 
print('Input directory contains the following sub-directories:\n {}'.format(os.listdir('../input/fruits-360_dataset/fruits-360')))
print('Validation directory contains the following sub-directories:\n {}'. \
      format(os.listdir('../input/fruits-360_dataset/fruits-360/Test')))
print('Training directory contains the following sub-directories:\n {}'. \
      format(os.listdir('../input/fruits-360_dataset/fruits-360/Training')))
assert os.listdir('../input/fruits-360_dataset/fruits-360/Test') == os.listdir('../input/fruits-360_dataset/fruits-360/Training')
 # path to validation input directory
validationPathSource = '../input/fruits-360_dataset/fruits-360/Test'
# path to training input directory
trainPathSource = '../input/fruits-360_dataset/fruits-360/Training'

# path to the validation directory to which we will move validation images
validationPathDest = '../working/Validation' 
# path to the test directory to which we will move test images
testPathDest = '../working/Test' 
# path to the test directory to which we will move training images
trainPathDest = '../working/Training'
def get2working():
    
    """"" This function changes the current directory to the working directory
    regardless of the fact if the current directory is upstream or downstream the
    working directory """""
    
    while True:
        if os.getcwd() == '/kaggle/working': # if we are in the working directory then break
            break
        elif os.getcwd() == '/': # else if we are upstream change it to the working directory
            os.chdir('kaggle/working')
        else:
            os.chdir('..') # else if we are downstream move back a directory untill we are in the working directory
def createfolder(pathandname):
    
    """"" Given a path ending with the directory's name to be created as the input to 
    this function, a folder with that name is created """""
    
    get2working() # ensure that the current directory is the working directory
    
    try:
        os.mkdir(pathandname) # make the desired directory
        print('Folder created')
    except FileExistsError:
        print('Folder already exists so command ignored') # ignore if the directory already exits
createfolder(validationPathDest)
createfolder(testPathDest)
createfolder(trainPathDest)
test_dict = {} # empty dictionary to store validation data
train_dict = {} # empty dictionary to store training data
fruit_numbers = {} # empty dictionary to store the above defined 2 dictionaries
get2working() # ensure that the current directory is the working directory

for classes in os.listdir(validationPathSource): # looping over the subdirectories in the validationPath (this can be changed to trainPathSource too as it will make no difference to the result)
    # calculating number of fruits
    test_dict[classes] = len(os.listdir(os.path.join(validationPathSource,classes))) 
    train_dict[classes] = len(os.listdir(os.path.join(trainPathSource,classes)))

fruit_numbers['Test'] = test_dict # assigning val_dict to 'Validation' key
fruit_numbers['Training'] = train_dict # assigning test_dict to 'Training' key

df_fruit_numbers = pd.DataFrame.from_dict(fruit_numbers) # creating a dataframe from fruit_numbers
print(df_fruit_numbers) # visualizing the dataframe

# making sure that no values are null or zero. The following code should not print empty dataframes
print(df_fruit_numbers[(df_fruit_numbers.Training == 0)])
print(df_fruit_numbers[(df_fruit_numbers.Training == np.nan)])
print(df_fruit_numbers[(df_fruit_numbers.Test == 0)])
print(df_fruit_numbers[(df_fruit_numbers.Test == np.nan)])                  
df_fruit_numbers.to_csv('FruitNumbers.csv') 
get2working() # Changing path to working directory
os.chdir(validationPathSource) # Changing path to input training folder
fruitnames = [file for file in os.listdir()]; # Storing names of fruits (sub folders within the training folder) in a list

# Looping over the list of fruit names
for fruit in fruitnames:
    get2working() # Changing path to working directory
    validationpath = os.path.join(validationPathDest,fruit) # Creating path for a specifc fruit for the output validation folder
    testpath = os.path.join(testPathDest,fruit) # Creating path for a specifc fruit for the output test folder
    trainpath = os.path.join(trainPathDest,fruit) # Creating path for a specifc fruit for the output training folder
    
    sourcepath = os.path.join(validationPathSource,fruit) # Creating path for a specifc fruit for the source validation folder
    sourcepathtrain = os.path.join(trainPathSource,fruit) # Creating path for a specifc fruit for the source training folder
    
    os.mkdir(testpath) # Creating a folder for a specific fruit in the test directory
    os.mkdir(validationpath) # Creating a folder for a specific fruit in the validation directory
    os.mkdir(trainpath) # Creating a folder for a specific fruit in the training directory
    
    os.chdir(sourcepath) # Changing path to the source directory
    randomsample = random.sample(os.listdir(),len(os.listdir())) # Sampling random fruit images for a certain fruit
    
    get2working() # Changing path to the working directory
    # Copying the first 25% fruit images from the source folder (randomaly sampled already) and copying them to the test folder 
    for k in range(0,len(randomsample)//4):
        shutil.copy(os.path.join(sourcepath,randomsample[k]),testpath)
    
    # Copying the rest of fruit images from the source folder (randomaly sampled already) and copying them to the validation folder
    for k in range(len(randomsample)//4,len(randomsample)):
        shutil.copy(os.path.join(sourcepath,randomsample[k]),validationpath)
    
    # Copying all images from source training folder to the training folder in the working directory
    os.chdir(sourcepathtrain) # Changing path to the source training directory
    name_images = os.listdir()
    
    get2working() # Changing path to the working directory
    for k in range(0,len(name_images)):
        shutil.copy(os.path.join(sourcepathtrain,name_images[k]),trainpath)
# Compressing output folders to zip files
shutil.make_archive('Validation', 'zip',os.getcwd(), validationPathDest)
shutil.make_archive('Testing', 'zip',os.getcwd(), testPathDest)
shutil.make_archive('Training', 'zip',os.getcwd(), trainPathDest)

# Removing uncompressed output folders
shutil.rmtree(validationPathDest)
shutil.rmtree(testPathDest)
shutil.rmtree(trainPathDest);
# making sure that the data has been written
os.listdir()