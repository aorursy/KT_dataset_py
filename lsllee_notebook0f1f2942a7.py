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
import os.path

import random

import numpy as np

from skimage import io

import matplotlib.pyplot as plt

import cv2

import os

import math  

import pandas as pd

import sklearn.metrics as sm

import tensorflow as tf
# Check current directory

curr_wkdir = os.getcwd()          # get the current working directory

print(curr_wkdir)
# Load the model

import os.path

from tensorflow.keras.models  import  load_model



# model_path = curr_wkdir + '/model'           # *** parameter : need to update with correct folder name during submission

# model_path = model_path.replace("\\","/")

model_path = '../input/osic-model-ver3c'

model_savefile = model_path + '/osic_model_ver1.0.h5'



osic_model = load_model(model_savefile)

osic_model.summary()
# Check model weights

osic_model.get_weights()
# **************************************************************

# *** (B) Testing the model on data with known true values   ***

# **************************************************************

# Test with a known true values dataset, compute standard deviation, compute model's confidence, 

# and create submission.csv



# p = curr_wkdir + '/data'              # *** parameter : need to update with correct folder name during submission         

# test_dir = p.replace("\\","/")

test_dir ='../input/osic-pulmonary-fibrosis-progression'



# Changes on 5/Oct/2020  (train_data for reference)

maxPercent = 160

# need to refer to train data for data attributes profiling     5/Oct/2020

train_file = test_dir + '/train.csv'   # *** parameter : need to update with correct file name during submission

train_data = pd.read_csv(train_file)

train_data['FullFVC'] = (train_data['FVC'] * 100) / train_data['Percent'] 

train_data['PercentScaled'] = train_data['Percent'] / maxPercent

train_data['Gender'] = np.where(train_data['Sex'] == 'Male', 1, 0)

# end of changes for train_data profile  5/Oct/2020





# *** Specify the file to be tested below  5/Oct/2020   

# test_file = '../input/osic-pulmonary-fibrosis-progression/train.csv'   # *** parameter : need to update with correct file name during submission

test_file = '../input/osic-pulmonary-fibrosis-progression/test.csv'   # *** parameter : need to update with correct file name during submission





test_data = pd.read_csv(test_file)



# below codes is to remove those rows belonging to the compressed dicom images ID00011637202177653955184

if  'ID00011637202177653955184' in test_data.values:

    # Get names of indexes for which column Patient has specified value 

    indexNames = test_data[ test_data['Patient'] == "ID00011637202177653955184" ].index

    # Delete these row indexes from dataFrame

    test_data.drop(indexNames , inplace=True)



# Add computed columns to the dataframe - FullFVC, PercentScaled, Gender

test_data['FullFVC'] = (test_data['FVC'] * 100) / test_data['Percent'] 



# Changes for ver3c

# test_data['PercentScaled'] = test_data['Percent'] / 100

# maxPercent = 160                                                   # 4/Oct/2020  - move to the top

test_data['PercentScaled'] = test_data['Percent'] / maxPercent

# end of changes for ver3c



test_data['Gender'] = np.where(test_data['Sex'] == 'Male', 1, 0)

# Below statements are for testing and checking

print(test_data.columns.values)
# generate the test patient info and test input file

test_patient_info = test_data.drop(columns=['Weeks','FVC','Percent','Sex','PercentScaled'])

print(test_patient_info.columns.values)

print(len(test_patient_info))
# generate the test patient info and test input file

test_patient_info_np = np.array(test_patient_info)

patient_count = len(test_patient_info)



test_data_file = pd.DataFrame()



for i in range(patient_count):

    patient = test_patient_info_np[i,0]

    age = test_patient_info_np[i,1]

    smokingstatus = test_patient_info_np[i,2]

    fullfvc = test_patient_info_np[i,3]

    gender = test_patient_info_np[i,4]



    for wk in range(-12, 134):                     # need to generate from weeks -12 to 133

        test_data_file = (test_data_file.append({'Patient':patient, 'Weeks':wk,

                                                'Age':age, 'SmokingStatus':smokingstatus,

                                                'FullFVC':fullfvc, 'Gender':gender},

                                                ignore_index=True))

    

    

    
# The below codes are for testing

print(f'test_data_file.columns.values : {test_data_file.columns.values}')

print(f'len(test_data_file): {len(test_data_file)}')

# print(test_data_file)
# function for plotting images

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 10, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
# Backup - NOTE : this is for ver1 & Ver3

# Define function that returns montage of images

def create_montage(dir):

    inputImages = []

    outputImage = np.zeros((224,224), dtype="uint8")      

    files_list = []

    

    # obtain the list of image files in the folder

    files_list = os.listdir(dir)



    # Check number of files (i.e. images in the folder)

    onlyfiles = next(os.walk(dir))[2] #dir is your directory path as string

    imagecount = len(onlyfiles)



    # based on the number of images for the patient, determine the interval if 4 images are to be taken

    imageinterval = math.floor(imagecount/4) 



    # initialize array for selected images

    selectedimagesno= []           



    # Loop thru 0 to 3, and take images file no for the interval

    for i in range(4):

        imageslice = (i * imageinterval) + 1

        selectedimagesno.append(imageslice) 



    # print (str(selectedimagesno)) 



    # loop thru the selected image file numbers and append image to inputImages array

    for item in selectedimagesno:

#         print(f'{dir}/{files_list[item]}')         # for testing purpose

        image = io.imread(f'{dir}/{files_list[item]}')

#         print(f'{dir}/{item}.dcm')                 # for testing purpose

        # plt.imshow(image)

        image = cv2.resize(image,(112,112))

        inputImages.append(image)





    # # inputImages (below plot image statement works!)

    # plotImages(inputImages)



    outputImage[0:112, 0:112] = inputImages[0]

    outputImage[0:112, 112:224] = inputImages[1]

    outputImage[112:224, 112:224] = inputImages[2]

    outputImage[112:224, 0:112] = inputImages[3]  





# # Below plot images statement works!

#     plt.imshow(outputImage)

    

    return outputImage
# # Define function that returns montage of images (for ver2)

# def create_montage(dir):

#     inputImages = []

#     outputImage = np.zeros((312,312), dtype="uint8")      

#     files_list = []

    

#     # obtain the list of image files in the folder

#     files_list = os.listdir(dir)



#     # Check number of files (i.e. images in the folder)

#     onlyfiles = next(os.walk(dir))[2] #dir is your directory path as string

#     imagecount = len(onlyfiles)



#     # based on the number of images for the patient, determine the interval if 4 images are to be taken

#     imageinterval = math.floor(imagecount/4) 



#     # initialize array for selected images

#     selectedimagesno= []           



#     # Loop thru 0 to 3, and take images file no for the interval

#     for i in range(4):

#         imageslice = (i * imageinterval) + 1

#         selectedimagesno.append(imageslice) 



#     # print (str(selectedimagesno)) 



#     # loop thru the selected image file numbers and append image to inputImages array

#     for item in selectedimagesno:

# #         print(f'{dir}/{files_list[item]}')         # for testing purpose

#         image = io.imread(f'{dir}/{files_list[item]}')

# #         print(f'{dir}/{item}.dcm')                 # for testing purpose

#         # plt.imshow(image)

#         image = cv2.resize(image,(156,156))

#         inputImages.append(image)





#     # # inputImages (below plot image statement works!)

#     # plotImages(inputImages)



#     outputImage[0:156, 0:156] = inputImages[0]

#     outputImage[0:156, 156:312] = inputImages[1]

#     outputImage[156:312, 156:312] = inputImages[2]

#     outputImage[156:312, 0:156] = inputImages[3]  





# # # Below plot images statement works!

# #     plt.imshow(outputImage)

    

#     return outputImage

# Loop thru the image folder and create montage for every folder



# p = curr_wkdir + '/data/train'        # *** parameter : need to update with correct folder name during submission

# path = p.replace("\\","/")





# *** Specify the folder for images to be tested below   5/Oct/2020

# path = '../input/osic-pulmonary-fibrosis-progression/train'

path = '../input/osic-pulmonary-fibrosis-progression/test'

print(path)



images_list=[]

imagesID_list =[]

directory_list = []



directory_contents = os.listdir(path)

# print(directory_contents)       # for testing



# the case ID00011637202177653955184 with compressed dicom images will be removed 

# because the quality of the decompressed image may be compromised, hence affecting 

# the training, validation and testing

# remove the folder with compressed dicom from dataset

directory_list = directory_contents



if directory_list.count('ID00011637202177653955184') > 0:

    directory_list.remove('ID00011637202177653955184')



for item in directory_contents:

#     print(item)

#     print(f'{path}/{item}')

    folder = f'{path}/{item}'

#     print(folder)

#     montage = create_montage(f'{path}/{item}')

    montage = create_montage(folder)

    images_list.append(montage)

    imagesID_list.append(item)





# plt.imshow(images_list)

plotImages(images_list)            # for testing purpose
# # Backup before changes 5/Oct/2020 - point to generated test data file

# # prepare input images

# # Prepare image list for the test set



# input_images = []



# for i in range(len(test_data)) : 

#     imageloc = imagesID_list.index(test_data.iloc[i, 0])      

#     input_images.append(images_list[imageloc])



# input_images = np.array(input_images)    
# prepare input images

# Prepare image list for the test set



input_images = []



for i in range(len(test_data_file)) : 

    imageloc = imagesID_list.index(test_data_file.iloc[i, 3])     #patient column in test_data_file  

    input_images.append(images_list[imageloc])



input_images = np.array(input_images)    
# Process the attributes of the patients

# import the necessary packages



# a) Numeric/continuous values:    Age, Weeks, FullFVC

# b) Categorical values:           Gender and Smoking Status 



# data in datasets are as following:

# ['Weeks' 'Age' 'SmokingStatus' 'FullFVC' 'Gender']



from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MinMaxScaler



def process_input_attributes(df, inputdata):

    # initialize the column names of the continuous data

    continuous = ['Weeks','Age','FullFVC']

    

    # performin min-max scaling each continuous feature column to

    # the range [0, 1]

    cs = MinMaxScaler()

    inputdataContinuous = cs.fit_transform(inputdata[continuous])

    

    # one-hot encode the SmokingStatus and Gender categorical data (by definition of

    # one-hot encoding, all output features are now in the range [0, 1])

    zipBinarizer = LabelBinarizer().fit(df['SmokingStatus'])

    inputdataCategorical1 = zipBinarizer.transform(inputdata['SmokingStatus'])



    zipBinarizer = LabelBinarizer().fit(df['Gender'])

    inputdataCategorical2 = zipBinarizer.transform(inputdata['Gender'])

    

    # construct our inputdataing and testing data points by concatenating

    # the categorical features with the continuous features

    inputdataX = np.hstack([inputdataCategorical1, inputdataCategorical2, inputdataContinuous])

    

    # return the concatenated inputdataing and testing data

    return (inputdataX)

# # Backup before changes 5/Oct/2020 - point to generated test data file

# # prepare input data

# # ['Weeks' 'Age' 'SmokingStatus' 'FullFVC' 'Gender']



# input_data = test_data.drop(columns=['Patient','FVC','Percent','Sex','PercentScaled'])



# # *** Below changes was made on 4/Oct/2020

# # input_attributes = process_input_attributes(test_data, input_data)

# input_attributes = process_input_attributes(train_data, input_data)

# # *** end of change  4/Oct/2020



# print(f'input data cols : {input_data.columns.values}')   # for checking purpose
# prepare input data     -   point to test_data_file  5/Oct/2020

# test_data_file.columns.values : ['Age' 'FullFVC' 'Gender' 'Patient' 'SmokingStatus' 'Weeks']

# ['Weeks' 'Age' 'SmokingStatus' 'FullFVC' 'Gender']



input_data = test_data_file.drop(columns=['Patient'])

input_attributes = process_input_attributes(train_data, input_data)       # chg on 4/Oct/2020



print(f'input data cols : {input_data.columns.values}')   # for checking purpose
# ***  TESTING : The below paragrapgh are for testing

print(f'len(test_data) : {len(test_data)}')

print(f'len(test_data_file) : {len(test_data_file)}')

print(f'len(input_images) : {len(input_images)}')

print(f'len(input_attributes) : {len(input_attributes)}')
# make prediction with model

test_predictions = osic_model.predict([input_attributes, input_images])
# # The below para is not required anymore because the true value is not known   5/Oct/2020

# # Measure accuracy of prediction

# actual_percentscaled = test_data['PercentScaled']



# # print(f'{actual_percentscaled}')     # for testing

# # print(f"Explain variance score = {round(sm.explained_variance_score(y_valid_rs, predict_validations), 2)}") 



# print(f"Explain variance score = {round(sm.explained_variance_score(actual_percentscaled, test_predictions), 2)}") 

# # Backup before making changes to point to test_data_file 

# #      (generated based on weeks -12 to 133)  5/Oct/2020

# # Try 3a   ==> works but column in alphabetical order

# # # include predictions into the test data, compute predicted FVC, standard deviation, 

# # # and model's confidence for every record

# # # Generate output submission.csv file

# # # Data used are from test_data & test_predictions



# # column_names = ['Patient' 'Weeks' 'FVC' 'Percent' 'Age' 'Sex' 'SmokingStatus' 'FullFVC'

# #  'PercentScaled' 'Gender','PredictionPercentScaled']



# test_data_np = np.array(test_data)

# # results = pd.DataFrame(columns = column_names)

# results = pd.DataFrame()

# rec_count = len(test_data)



# for i in range(rec_count):

#     results = (results.append({'Patient':test_data_np[i,0], 'Weeks':test_data_np[i,1], 

#                                'FVC': test_data_np[i,2], 'Percent': test_data_np[i,3], 

#                                'Age': test_data_np[i,4], 'Sex': test_data_np[i,5], 

#                                'SmokingStatus': test_data_np[i,6], 'FullFVC': test_data_np[i,7], 

#                                'PercentScaled': test_data_np[i,8], 'Gender': test_data_np[i,9], 

#                                'PredictedPercentScaled': test_predictions[i]}, 

#                               ignore_index=True))    

    

# print('done')
# Point to test_data_file  5/Oct/2020

# Try 3a   ==> works but column in alphabetical order

# # include predictions into the test data, compute predicted FVC, standard deviation, 

# # and model's confidence for every record

# # Generate output submission.csv file

# # Data used are from test_data & test_predictions



# test_data_file.columns.values : ['Age' 'FullFVC' 'Gender' 'Patient' 'SmokingStatus' 'Weeks']

#  'PercentScaled' 'Gender','PredictionPercentScaled']



test_data_file_np = np.array(test_data_file)

# results = pd.DataFrame(columns = column_names)

results = pd.DataFrame()

rec_count = len(test_data_file)



for i in range(rec_count):

    results = (results.append({'Age':test_data_file_np[i,0], 'FullFVC':test_data_file_np[i,1], 

                               'Gender': test_data_file_np[i,2], 'Patient': test_data_file_np[i,3], 

                               'SmokingStatus': test_data_file_np[i,4], 'Weeks': test_data_file_np[i,5], 

                               'PredictedPercentScaled': test_predictions[i]}, 

                              ignore_index=True))    

    

print('done')
print(len(test_data_file))    # for testing
# # Backup before changes due to generated test data file

# #   ==> true fvc will not be known, hence unable to compute std deviation and confidence



# # compute predicted full fvc

# # compute mean square error and std dev



# # Changes for ver3c

# # results['PredictedFVC'] = results['PredictedPercentScaled'] * results['FullFVC']

# results['PredictedFVC'] = ((results['PredictedPercentScaled'] * maxPercent) / 100) * results['FullFVC']

# # end of changes for ver3c



# predictedfvc = results['PredictedFVC']

# truefvc = results['FVC']



# # Compute Mean square error

# print(f"Mean squared error = {round(sm.mean_squared_error(truefvc, predictedfvc), 2)}") 

# print(f"Standard deviation = {round(math.sqrt(sm.mean_squared_error(truefvc, predictedfvc)), 5)}") 



# print(f"Explain variance score = {round(sm.explained_variance_score(truefvc, predictedfvc), 5)}") 



# std_deviation = math.sqrt(sm.mean_squared_error(truefvc, predictedfvc))

# # print(f'std_deviation : {std_deviation}')                 

# Point to test_data_file (generated test data file)    5/Oct/2020

# compute predicted full fvc



# Changes for ver3c

# results['PredictedFVC'] = results['PredictedPercentScaled'] * results['FullFVC']

results['PredictedFVC'] = ((results['PredictedPercentScaled'] * maxPercent) / 100) * results['FullFVC']

# end of changes for ver3c



predictedfvc = results['PredictedFVC']
# # Compute confidence not required any more because true fvc is not known   5/Oct/2020

# # Compute Confidence of prediction for every row

# def compute_confidence(std_dev, true_fvc, predicted_fvc):

#     std_dev_clipped = max(std_dev, 70)

    

# #     *** Changes made 5/OCt/2020

#     upper_limit = np.array([1000])

# #     delta =  min(abs(true_fvc - predicted_fvc),1000)

#     delta =  min(abs(true_fvc - predicted_fvc),upper_limit) 

# #     *** end of changes made 5/Oct/2020

    

#     sqrt_2 = math.sqrt(2)

    

#     metric = -((sqrt_2*delta)/std_dev_clipped) - math.log(sqrt_2*std_dev_clipped)

#     return metric
# # Backup before making changes due to generated test data file   5/Oct/2020

# # # Compute Confidence in results sets

# # results['Confidence'] = compute_confidence(std_deviation,results['FVC'],results['PredictedFVC'])



# results_confidence =[]

# results_np = np.array(results)



# for i in range(len(results_np)):

#     truefvc       = results_np[i,1]     # i is the row number

#     predictedfvc  = results_np[i,11]  

#     confidence = compute_confidence(std_deviation, truefvc, predictedfvc)

#     results_confidence.append(confidence)

    

# # print(f'results_confidence : {results_confidence}')       # for testing



# # add the results_confidence as the 'Confidence' column in results

# results['Confidence'] = results_confidence



# # for checking / testing

# # print(results_confidence)
# # Compute Confidence in results sets

# ==> Confidence will be hardcoded to 100 because true fvc is not known



results_confidence =[]

results_np = np.array(results)



for i in range(len(results_np)):

    confidence = 100

    results_confidence.append(confidence)

    

# print(f'results_confidence : {results_confidence}')       # for testing



# add the results_confidence as the 'Confidence' column in results

results['Confidence'] = results_confidence



# for checking / testing

# print(results_confidence)
print(f'results columns : {results.columns.values}')
# Generate the column Patient_Week 

# convert week to int first to remove .0 before converting to string

results['Patient_Week'] = results['Patient'] + '_' + results['Weeks'].astype(int).astype(str)

print(f'results columns : {results.columns.values}')
# Below statement is for testing - for review of the results

# results_filename = curr_wkdir + '/results.csv'

# results_filename = results_filename.replace("\\","/")

results_filename = './' + 'results.csv'

results.to_csv(results_filename, index=True)
# Below statements are for testing

# print(results['Weeks'])

# print(results['Patient'])

# print(results['Patient_Week'])

print(f'results no of rows : {len(results)}')
# Generate submission file

submission = pd.DataFrame()

patientwk_s = results['Patient_Week']

fvc_s       = results['PredictedFVC']

confidence_s = results['Confidence']



submission = pd.concat([patientwk_s, fvc_s, confidence_s], axis=1) 



# rename submssion file column

submission_file = submission.rename(columns = {'PredictedFVC': 'FVC'}, inplace = False)



# remove square bracket from the fvc and confidence columns

# submission_file['FVC'] =  submission_file['FVC'].str.get(0)                          

submission_file['FVC'] =  (submission_file['FVC'].str.get(0)).round(0).astype(int)     # 5/Oct/2020

# submission_file['Confidence'] =  submission_file['Confidence'].str.get(0)            # chg due to generated data file 5/Oct/2020



print(submission_file)



# output submission file to csv, set index = False so that index will not be included in output

# output_filename = curr_wkdir + '/submission.csv'                   #parameter : need to update with correct folder name

# output_filename = output_filename.replace("\\","/")



output_filename = './' + 'submission.csv'

# output_filename = '../output' + 'submission.csv'



submission_file.to_csv(output_filename, index=False) 