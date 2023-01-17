# Import 
import cv2
import pandas as pd 
import numpy as np 
import matplotlib
from IPython.display import clear_output, Image, display
import PIL.Image
import io
import glob
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# Read the train data
df_train = pd.read_csv("../input/train.csv")
print ("Total Number of Images: " + str(df_train.shape[0]))
# check for null data 
data_check_images = df_train.loc[df_train['Image'].isnull()]
data_check_id = df_train.loc[df_train['Id'].isnull()]
print ('Number of null entry in Images : ' + str(data_check_images.shape[0]))
print ('Number of null entry in Id : ' + str (data_check_id.shape[0]))
df_train.head()
#Function to display image in jupyter notebook 
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
# Get all the images into the list
images_glob = glob.glob("../input/train/*.jpg")
print ("Number of Train images: " + str(len(images_glob)))
# Display a random image
random_image = images_glob[0]
image_data = cv2.imread(random_image)
showarray(image_data)
print ("Id of the image : ")
print (df_train['Id'].loc[df_train['Image'].apply(lambda image : image==random_image.split('/')[-1])])
#check for sizes of the images
for each_image in images_glob[5:10]:
    data_image = cv2.imread(each_image)
    print (each_image.split('/')[-1] +" shape is : " + str(data_image.shape))
# create a dictionary containing key as image and its value as its shape
image_data_id = {}
for each_image, each_id in zip(df_train['Image'].tolist()[:100], df_train['Id'].tolist()[:100]):
    image_data_id[each_image] = cv2.imread("../input/train/"+each_image).shape
# print out the minimum resolution of the image
print ("Minimum Length of an Image in whole subset of data : " + str(np.array(list(image_data_id.values()))[:,0].min())) 
print ("Minimum Width of an Image in Whole Subset of data: " + str(np.array(list(image_data_id.values()))[:,1].min())) 

# create a dictionary with image name as key and resized image data as its value
image_data = {}
for each_image, each_id in zip(df_train['Image'].tolist()[:100], df_train['Id'].tolist()[:100]):
    data_image = cv2.imread("../input/train/"+each_image)
    data_image = cv2.resize(data_image, (100,300))
    image_data[each_image] = data_image
    
# Create a dataframe with resized data of the images
df_train_resized = pd.DataFrame()
df_train_resized['Image'] = image_data.keys()
df_train_resized['resized_data']=list(image_data.values())
print (df_train_resized.head())
# Display the resized image
showarray(df_train_resized.iloc[7,1])
print ("Shape of the resized Image : " + str(df_train_resized.iloc[7,1].shape))
# resize the image to the lowest available resolution
df_train_resized['Labels'] = df_train["Id"].tolist()[:100]
# create a target data
target = pd.get_dummies(df_train_resized['Labels'])
labelled_whale = df_train_resized.loc[target.iloc[:,1:].any(axis=1)]
print (labelled_whale.head())
# Dataframe with only images labelled as "new_whale"
new_whale = df_train_resized.drop(labelled_whale.index)
# create a train data 
train_data = np.array(df_train_resized['resized_data'][:50].apply(lambda arr: arr.flatten()).tolist()).reshape((-1,300,100,3))
target_labeled_vs_new_whale = target.iloc[:,1:].any(axis=1)[:50]
# create a dataframe for the distribution plot
data_frame_train = pd.DataFrame()
data_frame_train['labels'] = target_labeled_vs_new_whale.tolist()
# Percentage Distribution of whales with labelling and unknown whales (new_whale)
print ("percentage of Lablled whales in train data : " + str((target_labeled_vs_new_whale.mean(axis=0)) * 100))
print ("Percentage of whales lablled as \'new_whale in train data' : " +  str(100 - ((target_labeled_vs_new_whale.mean(axis=0)) * 100)))
# Distribution under sample of training data 
plt.bar([0,1],data_frame_train['labels'].value_counts().tolist(), color=['r','b'], width=0.3)
plt.xticks([0,1], ['Labelled_with_Id', 'New_whale'])
plt.ylabel("Number of Images")
plt.title("Small Subset of data : 50 examples")

# creating the validation data
val_data = np.array(df_train_resized['resized_data'][50:].apply(lambda arr: arr.flatten()).tolist()).reshape((-1,300,100,3))
val_target = target.iloc[:,1:].any(axis=1)[50:]
print ("percentage of Lablled whales in test data : " + str ((val_target.mean(axis=0)) * 100))
print ("percentage of Lablled as \'new_whale' : " + str(100 - ((val_target.mean(axis=0)) * 100)))
