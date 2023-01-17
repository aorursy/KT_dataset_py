import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
imglist = pd.read_csv('../input/driver_imgs_list.csv')
train_target = imglist['classname']
train_img = imglist['img']
tra

img=np.load('../input/imgs/img_44733.jpg')
img
