from matplotlib import pyplot as plt

%matplotlib inline

import numpy as np

import cv2 #will be used to count the lines

import os

import pandas as pd

import seaborn as sns
df = pd.read_csv('../input/traininglabels.csv')

print(f'{df.has_oilpalm.value_counts()[1]/len(df):.1%} of photos with oilpalm in training')
test_id = os.listdir('../input/leaderboard_test_data/leaderboard_test_data')

test_id = np.array(test_id)

holdout_id = os.listdir('../input/leaderboard_holdout_data/leaderboard_holdout_data')

holdout_id = np.array(holdout_id)

print(f'images in test {len(test_id)},  in holdout {len(holdout_id)}')



path = '../input/leaderboard_test_data/leaderboard_test_data/'

path2 = '../input/leaderboard_holdout_data/leaderboard_holdout_data/'



rho =  np.pi/360

lst = []

for i in range(0,len(test_id)): 

    img = cv2.imread("%s/%s" % (path, test_id[i]))

    edges = cv2.Canny(img,100,200)

    lines = cv2.HoughLinesP(

            edges, 1, rho, 

            20, 0, 20, 20)

    if lines is None:

        lst.append(0)

    else: 

        lst.append(len(lines))



lst2 = []

for i in range(0,len(holdout_id)): 

    img = cv2.imread("%s/%s" % (path2, holdout_id[i]))

    edges = cv2.Canny(img,100,200)

    lines = cv2.HoughLinesP(

            edges, 1, rho, 

            20, 0, 20, 20)

    if lines is None:

        lst2.append(0)

    else: 

        lst2.append(len(lines))
fig=plt.figure(figsize=(20, 100))

columns = 4

rows = 25

for i in range(1, 101):

    img = cv2.imread("%s/%s" % (path, test_id[i-1]))

    title = f'lines: {lst[i-1]}'

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.title(title)

plt.show()
sns.distplot(lst,  bins=40, kde=False)

plt.title('distribution of lines in test folder')
sns.distplot(lst2, bins=40, kde=False)

plt.title('Distribution of lines in holdout folder')
fig=plt.figure(figsize=(20, 100))

columns = 4

rows = 25

for i in range(1, 101):

    img = cv2.imread("%s/%s" % (path2, holdout_id[i-1]))

    title = f'lines : {lst2[i-1]}'

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.title(title)

plt.show()
Low_limit = 8 

High_limit = 50



sub_col1 = list(test_id) + list(holdout_id)

sub_col2 = lst + lst2

sub = pd.DataFrame.from_dict({'image_id': sub_col1 , 'lines' : sub_col2} )



sub['has_oilpalm'] = 0

sub.loc[(sub['lines']> Low_limit) & (sub['lines'] < High_limit ), 'has_oilpalm' ] = 1

sub = sub[['image_id', 'has_oilpalm']]



sns.distplot(sub['has_oilpalm'],  bins=20, kde=False)

plt.title(f'{sub.has_oilpalm.value_counts()[1]/len(sub):.1%} of photos considered with oilpalm in submission')
sub.to_csv('sub_init3.csv', header = True, index = False)
#end

#bonus : 

#some steps if you want to prepare data for deep learning with keras



#https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

#import keras

#from keras import *

#from keras_preprocessing.image import ImageDataGenerator



# batchsize = 32

# targetsize = (256,256)



# train_data = ImageDataGenerator(rescale=1./255, 

#                                validation_split=0.2

#                                )

# train_generator=train_data.flow_from_dataframe(

#     dataframe=df, directory= '../input/train_images/train_images', 

#     x_col="image_id", y_col="has_oilpalm",

#     color_mode = "grayscale",

#     class_mode="categorical", 

#     target_size=targetsize, batch_size=batchsize, 

#     subset = "training"

#     )

# validation_generator=train_data.flow_from_dataframe(

#     dataframe=df, directory= '../input/train_images/train_images', 

#     x_col="image_id", y_col="has_oilpalm", 

#     color_mode = "grayscale",

#     class_mode="categorical", 

#     target_size=targetsize, batch_size=batchsize, 

#     subset = "validation"

#     )

# info on the created object : 

# x,y = validation_generator[2]

# x.shape, x[0].shape , x[0].squeeze().shape, y.shape









#############Backup#####



# def find_lines(img, rho , threshold, minlinenlength) :

#     "the two parameters after img will serve to tune the returned array"

#     edges = cv2.Canny(img,100,200)

#     lines = cv2.HoughLinesP(

#         edges, 1, rho, 

#         threshold, 0, minlinenlength, 20)

#     return lines





# def lines_counts(x, rng, rho, fl1, fl2): 

#     'x is one of the batches, i is the batch size. '

#     lst = []

#     for i in range(0, rng):

#         img = x[i].squeeze()

#         img = np.array(img * 255, dtype = np.uint8)

#         lines = find_lines(img , rho, fl1, fl2)

#         if lines is None:

#             lst.append(0)

#         else: 

#             lst.append(len(lines))

#     return lst





#def print lines  : #do do

  #print lines

 # for line in lines[0]:

    #print line

  #  cv2.line(img, (line[0],line[1]), (line[2],line[3]), (0,255,0), 2)

  #cv2.imwrite("line_edges.jpg", edges)

  #cv2.imwrite("lines.jpg", img) 

  #return lines

    

#img_orig= cv2.imread("lines.jpg")

#plt.imshow(img_orig)

#img_lign = cv2.imread("line_edges.jpg")

#plt.imshow(img_lign)

#!rm line_edges.jpg

#!rm lines.jpg

#del(lines)



# fig=plt.figure(figsize=(20, 30))

# columns = 4

# rows = batchsize/columns

# for i in range(1, batchsize+1):

#     img = x[i-1].squeeze() #from 3 to 2 dims

#     title = f'{i} : oilp {int(test[i-1])}, lines : {lst[i-1]}'

#     fig.add_subplot(rows, columns, i)

#     plt.imshow(img)

#     plt.title(title)

# plt.show()