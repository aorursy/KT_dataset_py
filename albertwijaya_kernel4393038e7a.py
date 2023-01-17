!cd ../input && ls
ls
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
train = pd.read_csv('../input/shopee-product-detection-student/train.csv')
test = pd.read_csv('../input/shopee-product-detection-student/test.csv')
train['category'] = train['category'].astype('object')
category_replace = {0:'00',1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09'}
train['category'] = train['category'].replace(category_replace.keys(), category_replace.values()).astype('str')
train.info()
train['category'].nunique()
