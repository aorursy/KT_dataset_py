# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

"""

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

"""

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import scipy.io 

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
def dataTableOptimizer(mat_file):

    our_data = mat_file['d_skel']

    datas = []

    frame_size = len(our_data[0][0])-1

    for each in range(0,frame_size):

        datas.append(our_data[:,:,each])

    return datas,frame_size



def dataTableOptimizerUpdated(mat_file):

    our_data = mat_file['d_skel']

    datas = []

    frame_size = len(our_data[0][0])-1

    for each in range(0,frame_size):

        data_flatten = our_data[:,:,each].flatten()

        data_flatten = data_flatten[np.newaxis]

        datas.append(data_flatten)

    return datas,frame_size



def dataTableFirstFrame(data,joint_names,column_names):

    data = pd.DataFrame(columns=column_names,index=joint_names,data=data[0])

    return data
def dataLinePlotter(data):

    points = data.loc['head',:]

    points2 = data.loc['soulder',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['soulder',:]

    points2 = data.loc['left_shoulder',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['soulder',:]

    points2 = data.loc['right_shoulder',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['soulder',:]

    points2 = data.loc['spine',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['spine',:]

    points2 = data.loc['hip_center',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_hip',:]

    points2 = data.loc['hip_center',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_hip',:]

    points2 = data.loc['hip_center',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_hip',:]

    points2 = data.loc['left_knee',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_hip',:]

    points2 = data.loc['right_knee',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_ankle',:]

    points2 = data.loc['left_knee',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_ankle',:]

    points2 = data.loc['right_knee',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_foot',:]

    points2 = data.loc['left_ankle',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_foot',:]

    points2 = data.loc['right_ankle',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_shoulder',:]

    points2 = data.loc['left_elbow',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_shoulder',:]

    points2 = data.loc['right_elbow',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_wrist',:]

    points2 = data.loc['right_elbow',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_wrist',:]

    points2 = data.loc['left_elbow',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['right_wrist',:]

    points2 = data.loc['right_hand',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    points = data.loc['left_wrist',:]

    points2 = data.loc['left_hand',:]

    ax.plot3D([points.x,points2.x],[points.y,points2.y],[points.z,points2.z])

    for x,y,z in datas_new.values:

        ax.scatter(x,y, z, color='red', marker='o')

    ax.set_xlabel('X Label')

    ax.set_ylabel('Y Label')

    ax.set_zlabel('Z Label')

    plt.show()

    plt.pause(0.5)

    ax.cla()
def dataTableForCluster(data,joint_names,column_names,frame,chosen_class_number):

    datam = pd.DataFrame(columns=column_names,index=joint_names,data=data[0])

    for x in range(1,frame):

        datam = pd.concat([datam,pd.DataFrame(columns=column_names,index=joint_names,data=data[x])])

    datam['classs'] = np.full((1,len(datam)),chosen_class_number).T

    return datam



def combineDatas(data_1,data_2):

    data_1 = data_1.append(data_2,ignore_index=True) 

    return data_1



def combineMultipleDatas(data_names):

   datas = data_names[0]

   x = 0

   for data in data_names:

       if x == 0:

           result = datas.append(data,ignore_index=True)

       else:

           result = result.append(data,ignore_index=True)

       x = x+ 1

   return result

       

def dataTableForCluster2(data):

    datam = pd.DataFrame(data=data.iloc[0])

    for x in range(1,60):

        datam = pd.concat([datam,pd.DataFrame(data=data.iloc[x])])

    return datam
joint_names = ['head','soulder','spine','hip_center','left_shoulder','left_elbow','left_wrist','left_hand','right_shoulder','right_elbow','right_wrist','right_hand','left_hip','left_knee','left_ankle','left_foot','right_hip','right_knee','right_ankle','right_foot']

col_names = ['x','y','z']
mat = scipy.io.loadmat('/kaggle/input/human-action-recognition-dataset/a1/a1_s1_t1_skeleton.mat')

all_data_class_a1 ,frame_a1 = dataTableOptimizer(mat_file=mat)

data_zero_frame = dataTableFirstFrame(data=all_data_class_a1,joint_names=joint_names,column_names=col_names)

fig=plt.figure(figsize=(12,6))

ax = fig.add_subplot(111,projection='3d')



for i in range(0,frame_a1):

    datas_new = pd.DataFrame(columns=col_names,index=joint_names,data=all_data_class_a1[i])

    dataLinePlotter(datas_new)