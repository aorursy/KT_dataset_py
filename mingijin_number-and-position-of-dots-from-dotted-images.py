%matplotlib inline

import numpy as np

import pandas as pd

from subprocess import check_output

from glob import glob

from scipy import stats

import matplotlib.pyplot as plt

import cv2

import os 
df = pd.read_csv("../input/Train/train.csv")

current_TD_dir="../input/TrainDotted/{}.jpg"

current_T_dir="../input/Train/{}.jpg"
'''

boundaries[0] : red / adult_males

boundaries[1] : mageneta / subadult_males

boundaries[2] : brown / adult_females

boundaries[3] : blue / juveniles

boundaries[4] : green / pups

'''

boundaries = [

    ('adult_males'   , [210, 0, 0],   [255, 40,  20]),

    ('subadult_males', [220, 0, 210], [255, 30, 255]),

    ('adult_females' , [70, 30, 0],   [100, 60,  30]),

    ('juveniles'     , [10, 30, 140], [40,  70, 190]),

    ('pups'          , [20,150,10],   [60, 190,  60]),

]
dotted_im = cv2.cvtColor(cv2.imread(current_TD_dir.format(3)),cv2.COLOR_BGR2RGB)

raw_im    = cv2.cvtColor(cv2.imread(current_TD_dir.format(3)),cv2.COLOR_BGR2RGB)



fig = plt.figure(figsize=(8,6))



ax = fig.add_subplot(2,2,1)

plt.imshow(dotted_im)

plt.title("TrainDotted image")



ax = fig.add_subplot(2,2,2)

plt.imshow(raw_im)

plt.title("Raw image")



ax = fig.add_subplot(2,2,3)

diff = cv2.absdiff(dotted_im,raw_im)

plt.imshow(diff)

plt.title("TrainDotted - Raw image")



ax = fig.add_subplot(2,2,4)

gray = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)

plt.imshow(gray)

plt.title("TraninDotted - Raw gray image")
def check_count(num,boundaries=boundaries,TrainDotted_dir=current_TD_dir,Train_dir=current_T_dir):

    '''

    This function is designed to count the number of points by color

    

    num        : image file number

    boundaries : color range

    TrainDotted_dir : TrainDotted image in directory

    Train_dir : Train image in directory

    '''

    dotted_file = TrainDotted_dir.format(num)

    raw_file    = Train_dir.format(num)

    

    dotted_im = cv2.cvtColor(cv2.imread(dotted_file),cv2.COLOR_BGR2RGB)

    raw_im    = cv2.cvtColor(cv2.imread(raw_file),cv2.COLOR_BGR2RGB)

    

    diff = cv2.absdiff(dotted_im,raw_im)

    

    lower_bound = np.array([3,   3,  3], dtype="uint8")

    upper_bound = np.array([255, 255, 255], dtype="uint8")

    

    mask = cv2.inRange(diff,lower_bound,upper_bound)    

    point_im  = cv2.bitwise_and(dotted_im,dotted_im,mask = mask)

    

    for boundary in boundaries :

        lower_color = np.array(boundary[1],dtype="uint8")

        upper_color = np.array(boundary[2],dtype="uint8")

        

        mask = cv2.inRange(point_im,lower_color,upper_color)

        res  = cv2.bitwise_and(point_im,dotted_im,mask = mask)

        

        gray = cv2.cvtColor(res,cv2.COLOR_RGB2GRAY)

        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        

        count = 0

        for cnt in cnts:

            ((_, _), r) = cv2.minEnclosingCircle(cnt)

            if r>=1.5 and r<4:

                count = count + 1

        

        print("{} Provided: {}, this function Found: {}".format(boundary[0],df.iloc[num][boundary[0]],count)) 
for i in range(0,10):

    check_count(i)

    print("\n")
def check_series(num,boundaries=boundaries,TrainDotted_dir=current_TD_dir,Train_dir=current_T_dir):

    '''

    This function is designed to return the number of point per color.

    '''

    dotted_file = TrainDotted_dir.format(num)

    raw_file    = Train_dir.format(num)

    

    dotted_im = cv2.cvtColor(cv2.imread(dotted_file),cv2.COLOR_BGR2RGB)

    raw_im    = cv2.cvtColor(cv2.imread(raw_file),cv2.COLOR_BGR2RGB)

    

    diff = cv2.absdiff(dotted_im,raw_im)

    

    lower_bound = np.array([3,   3,  3], dtype="uint8")

    upper_bound = np.array([255, 255, 255], dtype="uint8")

    

    mask = cv2.inRange(diff,lower_bound,upper_bound)    

    point_im  = cv2.bitwise_and(dotted_im,dotted_im,mask = mask)

    

    temp = {}

    for boundary in boundaries :

        lower_color = np.array(boundary[1],dtype="uint8")

        upper_color = np.array(boundary[2],dtype="uint8")

        

        mask = cv2.inRange(point_im,lower_color,upper_color)

        res  = cv2.bitwise_and(point_im,dotted_im,mask = mask)

        

        gray = cv2.cvtColor(res,cv2.COLOR_RGB2GRAY)

        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        

        count = 0

        for cnt in cnts:

            ((_, _), r) = cv2.minEnclosingCircle(cnt)

            if r>=1.5 and r<4:

                count = count + 1

        

        temp[boundary[0]] = count

    

    result = pd.Series(temp,dtype=np.int64)

    result.name = num

    

    return result
# Check accuracy of finding dotted function with DataFrame

result_dot = pd.DataFrame(columns = ['adult_males','subadult_males','adult_females','juveniles','pups'])

for i in range(0,10):

    result_dot = result_dot.append(check_series(i))



# Percentage of Error (number of provided dot) - (found dot) < = 1

(np.sum(result_dot - df[0:10] <= 1) / len(result_dot) * 100)[0:5]
def find_point(num,type_lion,boundaries=boundaries,TrainDotted_dir=current_TD_dir,Train_dir=current_T_dir):

    '''

    This function is designed to return the center of points by color

    

    num        : image file number

    

    type_lion  : 

        0 - red / adult_males

        1 - mageneta / subadult_males

        2 - brown / adult_females

        3 - blue / juveniles

        4 - green / pups

    

    boundaries : color range

    TrainDotted_dir : TrainDotted image in directory

    Train_dir : Train image in directory

    '''

    dotted_file = TrainDotted_dir.format(num)

    raw_file    = Train_dir.format(num)

    

    dotted_im = cv2.cvtColor(cv2.imread(dotted_file),cv2.COLOR_BGR2RGB)

    raw_im    = cv2.cvtColor(cv2.imread(raw_file),cv2.COLOR_BGR2RGB)

    

    diff = cv2.absdiff(dotted_im,raw_im)

    

    lower_bound = np.array([3,   3,  3], dtype="uint8")

    upper_bound = np.array([255, 255, 255], dtype="uint8")

    

    mask = cv2.inRange(diff,lower_bound,upper_bound)    

    point_im  = cv2.bitwise_and(dotted_im,dotted_im,mask = mask)

    

    boundary = boundaries[type_lion]

    

    lower_color = np.array(boundary[1],dtype="uint8")

    upper_color = np.array(boundary[2],dtype="uint8")



    mask = cv2.inRange(point_im,lower_color,upper_color)

    res  = cv2.bitwise_and(point_im,dotted_im,mask = mask)



    gray = cv2.cvtColor(res,cv2.COLOR_RGB2GRAY)

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    

    result = []

    for cnt in cnts:

        ( (cx, cy), r) = cv2.minEnclosingCircle(cnt)

        if r>=1.5 and r<4:

            result.append((int(cx),int(cy)))

    

    return np.array(result)
# Visualize the location of points (by classes) 

def check_img(img_num,type_num,current_TD_dir = current_TD_dir):



    example_im = cv2.cvtColor(cv2.imread(current_TD_dir.format(img_num)),cv2.COLOR_BGR2RGB)

    example_point = find_point(img_num,type_num)

    

    length_point = len(example_point)

    

    fig = plt.figure(figsize=(12,length_point*2))



    for i in range(1,length_point+1):

        ax = plt.subplot(length_point,3,i)

        example_target = example_im[example_point[i-1][1]-30:example_point[i-1][1]+30,example_point[i-1][0]-30:example_point[i-1][0]+30]

        plt.imshow(example_target,aspect='auto')

        plt.axis("off")
check_img(4,2)