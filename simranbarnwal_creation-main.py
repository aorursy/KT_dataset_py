
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import genfromtxt
import matplotlib as mpl
from matplotlib import pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
train_data_org=pd.read_csv("../input/Training_rss_21Aug17.csv",header=None)
train_target_org=pd.read_csv("../input/Training_coordinates_21Aug17.csv",header=None)
test_data_org=pd.read_csv("../input/Test_rss_21Aug17.csv",header=None)
test_target_org=pd.read_csv("../input/Test_coordinates_21Aug17.csv",header=None)
train_device = pd.read_csv("../input/Training_device_21Aug17.csv")
test_device = pd.read_csv("../input/Test_device_21Aug17.csv")


train_device.loc[-1] = [train_device.columns.values[0]]  # adding a row
train_device.index = train_device.index + 1  # shifting index
train_device.sort_index(inplace=True) 
train_device.columns = ['995']

test_device.loc[-1] = [test_device.columns.values[0]]  # adding a row
test_device.index = test_device.index + 1  # shifting index
test_device.sort_index(inplace=True) 
test_device.columns = ['995']
train_data_org.iloc[:10]


test_data = test_data_org.replace({100:0})
train_data = train_data_org.replace({100:0})

data = test_data.append(train_data,  ignore_index=True)
print(data.index)

target = test_target_org.append(train_target_org,  ignore_index=True)
print(target.index)

device = train_device.append(test_device,  ignore_index=True)
print(device.index)

target.columns = ['992','993','994']
frames = [data, target, device]
big_data = pd.concat(frames,axis=1)
print(big_data.shape)

from sklearn.model_selection import train_test_split
train, test= train_test_split(big_data, test_size=0.2)
#test, val = train_test_split(blah, test_size=0.5)
print(train.shape)

#splitting for target
#train_data =  np.exp(train.iloc[:,0:992])
train_data =  (train.iloc[:,0:992])
train_target_x = train.iloc[:,992]
train_target_y = train.iloc[:,993]
train_target_z = train.iloc[:,994]
print(type(train_data))
#splitting for validation
#val_data =val.iloc[:,0:992]
#val_target_x = val.iloc[:,992]
#val_target_y = val.iloc[:,993]
#val_target_z = val.iloc[:,994]

#splitting for validation
#test_data =  np.exp(test.iloc[:,0:992])
test_data =  (test.iloc[:,0:992])
test_target_x = test.iloc[:,992]
test_target_y = test.iloc[:,993]
test_target_z = test.iloc[:,994]
device.to_pickle('device')
big_data.to_pickle('big_data')
train_data.to_pickle('train_index')
test_data.to_pickle('test_index')
train_target_x.to_pickle('train_x')
train_target_y.to_pickle('train_y')
train_target_z.to_pickle('train_z')
test_target_x.to_pickle('test_x')
test_target_y.to_pickle('test_y')
test_target_z.to_pickle('test_z')
