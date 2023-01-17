import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
%matplotlib inline

f = '75efad62c1'
img = load_img('../input/train/images/' + f + '.png')
img = img_to_array(img)
img = img[:,:,0]
plt.imshow(img)
def MyTensorFlow(img):
    width = 101
    height = 101
    tmpdata = tf.cast(img, dtype=tf.float32)/255.
    
    paddings = tf.constant([[5, 5,], [5, 5]])
    data = (tf.cast(tf.pad(tmpdata, paddings, mode='REFLECT'), dtype=tf.float32))
    
    indices = tf.range(width*height,dtype=tf.int64)
    R, C = tf.meshgrid(tf.range(width), tf.range(height), indexing='ij')
    R = tf.reshape(R,[-1])
    C = tf.reshape(C,[-1])
    def GP(i, j):
        gp1 = ( -1.109375 +
                1.0*tf.tanh(((data[i + 6][j + 4]) - (tf.where(((data[i + 4][j + 5]) - (((data[i + 9][j + 6]) * (data[i + 9][j + 6]))))>0, data[i + 6][j + 4], (13.36622619628906250) )))) +
                1.0*tf.tanh(((data[i + 5][j + 3]) * (((((data[i + 5][j + 3]) + (data[i + 5][j + 3]))) * (((data[i + 10][j + 4]) - (((data[i + 5][j + 3]) * (data[i + 5][j + 3]))))))))) +
                1.0*tf.tanh(((tf.where(((((data[i + 1][j + 7]) - (data[i + 5][j + 7]))) * (data[i + 1][j + 7]))<0, data[i + 1][j + 7], data[i + 5][j + 7] )) - (data[i + 4][j + 7]))) +
                1.0*tf.tanh(((((data[i + 10][j + 2]) - (data[i + 6][j + 2]))) * (((((((data[i + 6][j + 2]) - (data[i + 10][j + 2]))) - (data[i + 10][j + 2]))) - (data[i + 10][j + 2]))))) +
                1.0*tf.tanh(((((data[i + 4][j + 4]) - (data[i + 0][j + 4]))) * (((data[i + 0][j + 4]) - (((data[i + 4][j + 4]) - (((data[i + 0][j + 4]) - (data[i + 4][j + 4]))))))))) +
                1.0*tf.tanh(((tf.where(data[i + 0][j + 3]>0, (0.27110701799392700), ((data[i + 9][j + 1]) + (data[i + 9][j + 3])) )) * (tf.where(data[i + 0][j + 4]>0, (0.27110701799392700), data[i + 6][j + 3] )))) +
                1.0*tf.tanh(tf.where(data[i + 9][j + 4]>0, ((data[i + 9][j + 4]) - (tf.where(((data[i + 9][j + 4]) - (data[i + 6][j + 4]))<0, data[i + 6][j + 4], data[i + 9][j + 4] ))), data[i + 4][j + 3] )) +
                1.0*tf.tanh(((((((data[i + 8][j + 6]) - (data[i + 3][j + 6]))) - (((data[i + 10][j + 10]) - (data[i + 8][j + 7]))))) * (((data[i + 3][j + 6]) - (data[i + 8][j + 6]))))) +
                1.0*tf.tanh(((data[i + 4][j + 4]) - (tf.where(((data[i + 7][j + 3]) - (data[i + 3][j + 3]))>0, data[i + 7][j + 3], data[i + 3][j + 3] )))) +
                0.923754*tf.tanh(tf.where(data[i + 5][j + 5]>0, tf.where(data[i + 3][j + 2]>0, (0.13770106434822083), data[i + 6][j + 3] ), ((((data[i + 9][j + 2]) - ((0.13770106434822083)))) - ((0.13770106434822083))) )) +
                1.0*tf.tanh(tf.where(((data[i + 0][j + 6]) - (data[i + 4][j + 6]))<0, ((data[i + 0][j + 6]) - (data[i + 4][j + 6])), ((data[i + 4][j + 6]) - (data[i + 0][j + 6])) )) +
                1.0*tf.tanh(((((data[i + 9][j + 9]) - (((data[i + 5][j + 9]) * (data[i + 5][j + 9]))))) * (((data[i + 5][j + 9]) - (((data[i + 9][j + 9]) - (data[i + 5][j + 9]))))))) +
                1.0*tf.tanh(((data[i + 6][j + 2]) - (tf.where(((data[i + 7][j + 4]) - (data[i + 8][j + 4]))>0, data[i + 6][j + 2], ((data[i + 10][j + 4]) * (data[i + 10][j + 4])) )))) +
                1.0*tf.tanh(((data[i + 10][j + 9]) * (((data[i + 10][j + 9]) * (((data[i + 6][j + 8]) - (tf.where(data[i + 0][j + 9]>0, (1.0), data[i + 1][j + 9] )))))))) +
                0.876833*tf.tanh(((((data[i + 6][j + 1]) * (((data[i + 6][j + 1]) + (((data[i + 6][j + 1]) + (data[i + 6][j + 1]))))))) * (((data[i + 3][j + 3]) - (data[i + 6][j + 1]))))) +
                1.0*tf.tanh(((tf.where(((data[i + 6][j + 4]) - (data[i + 9][j + 4]))>0, data[i + 9][j + 4], data[i + 6][j + 4] )) - (data[i + 9][j + 4]))) +
                1.0*tf.tanh(tf.where(((data[i + 7][j + 2]) - (data[i + 4][j + 2]))>0, ((data[i + 4][j + 2]) - (data[i + 7][j + 2])), ((data[i + 7][j + 2]) - (data[i + 4][j + 2])) )) +
                1.0*tf.tanh(((((data[i + 10][j + 7]) - (((data[i + 2][j + 3]) * (data[i + 2][j + 3]))))) * (((data[i + 10][j + 2]) - (((data[i + 4][j + 3]) * (data[i + 4][j + 3]))))))) +
                1.0*tf.tanh(((data[i + 5][j + 5]) - (tf.where(data[i + 10][j + 5]>0, tf.where(((data[i + 5][j + 5]) - (data[i + 10][j + 5]))>0, data[i + 5][j + 5], data[i + 10][j + 5] ), data[i + 8][j + 6] )))) +
                1.0*tf.tanh(((tf.where(data[i + 0][j + 0]>0, data[i + 0][j + 0], data[i + 5][j + 1] )) - (tf.where(((data[i + 0][j + 0]) - (data[i + 3][j + 1]))>0, data[i + 0][j + 0], data[i + 3][j + 1] )))) +
                0.996090*tf.tanh(tf.where(data[i + 0][j + 5]>0, (0.12835982441902161), ((((((data[i + 5][j + 6]) - ((0.12836340069770813)))) - ((0.12836340069770813)))) - ((0.12836340069770813))) )) +
                1.0*tf.tanh(((data[i + 7][j + 0]) - (tf.where(((data[i + 7][j + 0]) * (((data[i + 7][j + 0]) - (data[i + 2][j + 0]))))<0, data[i + 2][j + 0], data[i + 7][j + 0] )))) +
                1.0*tf.tanh(((tf.where(data[i + 0][j + 0]>0, data[i + 0][j + 0], data[i + 2][j + 4] )) * (((data[i + 2][j + 0]) - (tf.where(data[i + 5][j + 0]>0, data[i + 0][j + 0], data[i + 5][j + 0] )))))) +
                1.0*tf.tanh(((data[i + 7][j + 3]) - (tf.where(((data[i + 7][j + 3]) - (data[i + 10][j + 3]))<0, data[i + 10][j + 3], data[i + 7][j + 3] )))) +
                1.0*tf.tanh(((data[i + 8][j + 2]) - (tf.where(data[i + 8][j + 2]>0, tf.where(((data[i + 6][j + 2]) - (data[i + 8][j + 2]))>0, data[i + 6][j + 2], data[i + 8][j + 2] ), data[i + 8][j + 2] )))) +
                1.0*tf.tanh(((((((data[i + 3][j + 7]) - (((data[i + 7][j + 8]) - (data[i + 3][j + 8]))))) - (data[i + 7][j + 8]))) * (((data[i + 7][j + 8]) - (data[i + 10][j + 8]))))) +
                1.0*tf.tanh(((((data[i + 0][j + 1]) - ((((2.0)) * (((data[i + 3][j + 1]) - (data[i + 0][j + 1]))))))) * (((data[i + 3][j + 1]) - (data[i + 5][j + 5]))))) +
                1.0*tf.tanh(((((((data[i + 4][j + 5]) - (((data[i + 7][j + 6]) - (data[i + 4][j + 5]))))) - (data[i + 7][j + 6]))) * (((data[i + 7][j + 6]) - (data[i + 10][j + 6]))))) +
                1.0*tf.tanh(tf.where(((data[i + 4][j + 3]) + ((((0.06125451624393463)) - ((data[i + 6][j + 3])))))<0, (((0.06125451624393463)) - (data[i + 6][j + 3])), (0.06125451624393463) )) +
                0.956012*tf.tanh(((data[i + 7][j + 4]) - (tf.where(((data[i + 2][j + 6]) - (data[i + 4][j + 4]))<0, data[i + 4][j + 5], data[i + 2][j + 4] )))) +
                0.997067*tf.tanh(((data[i + 2][j + 6]) * (((data[i + 0][j + 5]) * (((data[i + 5][j + 6]) - (((data[i + 7][j + 5]) * (data[i + 2][j + 7]))))))))) +
                1.0*tf.tanh(tf.where(((data[i + 1][j + 8]) - (data[i + 3][j + 8]))<0, ((data[i + 1][j + 8]) - (data[i + 3][j + 8])), ((data[i + 3][j + 8]) - (data[i + 1][j + 8])) )) +
                1.0*tf.tanh(((((data[i + 6][j + 5]) - (data[i + 0][j + 5]))) * (((((((data[i + 2][j + 5]) + (data[i + 3][j + 9]))) * (data[i + 8][j + 6]))) - (data[i + 0][j + 5]))))) +
                1.0*tf.tanh(((data[i + 10][j + 2]) - (tf.where(((data[i + 8][j + 2]) - (data[i + 10][j + 2]))<0, data[i + 10][j + 2], data[i + 8][j + 2] )))) +
                1.0*tf.tanh(((((data[i + 5][j + 8]) - (((data[i + 8][j + 5]) * (data[i + 8][j + 5]))))) * (((data[i + 8][j + 6]) - (((data[i + 4][j + 4]) * (data[i + 10][j + 6]))))))) +
                1.0*tf.tanh(((data[i + 7][j + 6]) - (tf.where(((data[i + 9][j + 7]) - (data[i + 7][j + 6]))<0, data[i + 7][j + 6], data[i + 9][j + 7] )))) +
                0.642229*tf.tanh(((data[i + 3][j + 0]) - (((data[i + 3][j + 0]) * ((((0.26259189844131470)) + (data[i + 3][j + 0]))))))) +
                1.0*tf.tanh(((((((data[i + 0][j + 10]) + (data[i + 0][j + 10]))) * (((data[i + 10][j + 4]) * (data[i + 0][j + 10]))))) * (((data[i + 10][j + 8]) - (data[i + 0][j + 10]))))) +
                1.0*tf.tanh(((data[i + 4][j + 8]) - (tf.where(((data[i + 6][j + 8]) - (data[i + 4][j + 8]))>0, tf.where(data[i + 4][j + 8]>0, data[i + 6][j + 8], data[i + 4][j + 8] ), data[i + 4][j + 8] )))) +
                1.0*tf.tanh(((((data[i + 10][j + 5]) - (data[i + 4][j + 3]))) * (((data[i + 10][j + 3]) - (tf.where(data[i + 8][j + 6]>0, data[i + 6][j + 2], (data[i + 10][j + 3]) )))))) +
                1.0*tf.tanh(((((data[i + 1][j + 1]) - (data[i + 6][j + 2]))) * (((data[i + 10][j + 10]) - (tf.where(data[i + 9][j + 2]>0, data[i + 9][j + 2], data[i + 6][j + 1] )))))) +
                1.0*tf.tanh(tf.where(((data[i + 10][j + 5]) - (data[i + 7][j + 5]))<0, ((data[i + 10][j + 5]) - (data[i + 7][j + 5])), ((data[i + 7][j + 5]) - (data[i + 10][j + 5])) )) +
                0.717498*tf.tanh(tf.where(((data[i + 9][j + 3]) - (data[i + 10][j + 3]))>0, tf.where(data[i + 10][j + 3]>0, (0.0), (5.01269578933715820) ), ((data[i + 7][j + 3]) * (data[i + 7][j + 3])) )) +
                1.0*tf.tanh(((data[i + 4][j + 3]) * (((data[i + 4][j + 4]) * (((((data[i + 6][j + 4]) * (data[i + 6][j + 4]))) - (data[i + 9][j + 4]))))))) +
                1.0*tf.tanh(((((data[i + 5][j + 3]) - (((data[i + 8][j + 4]) - (((data[i + 5][j + 3]) - (data[i + 8][j + 4]))))))) * (((data[i + 8][j + 4]) - (data[i + 10][j + 4]))))) +
                0.874878*tf.tanh(((data[i + 10][j + 2]) * (((data[i + 4][j + 4]) * (((((data[i + 7][j + 3]) - (data[i + 10][j + 4]))) * ((((3.0)) - (data[i + 7][j + 4]))))))))) +
                1.0*tf.tanh(((((data[i + 9][j + 6]) - (data[i + 6][j + 6]))) * (((data[i + 10][j + 5]) * (((data[i + 6][j + 6]) - (((data[i + 9][j + 6]) - (data[i + 6][j + 6]))))))))) +
                1.0*tf.tanh(((((data[i + 6][j + 6]) - (tf.where(data[i + 2][j + 6]>0, data[i + 3][j + 7], data[i + 0][j + 6] )))) * (((data[i + 3][j + 5]) - (data[i + 0][j + 6]))))) +
                1.0*tf.tanh(((((data[i + 2][j + 3]) - (((data[i + 0][j + 2]) - (data[i + 2][j + 3]))))) * (((data[i + 0][j + 10]) * (((data[i + 0][j + 3]) - (data[i + 2][j + 3]))))))) +
                0.999022*tf.tanh(tf.where(((data[i + 7][j + 5]) - (((data[i + 10][j + 0]) + (data[i + 0][j + 3])))) > -1, (0.0), (1.0) )) +
                0.911046*tf.tanh((((((0.49628388881683350)) - (data[i + 4][j + 3]))) * (data[i + 0][j + 3]))) +
                1.0*tf.tanh(((data[i + 10][j + 8]) * (((((data[i + 10][j + 8]) - (((data[i + 8][j + 7]) - (data[i + 10][j + 8]))))) * (((data[i + 8][j + 7]) - (data[i + 10][j + 8]))))))) +
                1.0*tf.tanh(((((data[i + 3][j + 3]) - (data[i + 10][j + 3]))) * (((((data[i + 3][j + 3]) - (data[i + 10][j + 3]))) * (((data[i + 3][j + 3]) * (data[i + 3][j + 3]))))))) +
                1.0*tf.tanh(((data[i + 7][j + 6]) - (tf.where(((data[i + 7][j + 6]) - (data[i + 9][j + 7]))<0, data[i + 9][j + 7], data[i + 7][j + 6] )))) +
                1.0*tf.tanh(((data[i + 7][j + 9]) - (tf.where(((data[i + 7][j + 9]) - (data[i + 10][j + 9]))>0, data[i + 7][j + 9], data[i + 10][j + 9] )))) +
                1.0*tf.tanh(((data[i + 10][j + 9]) - (tf.where(((data[i + 7][j + 9]) - (data[i + 10][j + 10]))>0, tf.where(data[i + 10][j + 6]>0, data[i + 7][j + 9], data[i + 10][j + 6] ), data[i + 10][j + 10] )))) +
                1.0*tf.tanh((((0.27924305200576782)) * (tf.where(data[i + 7][j + 5]>0, (0.27924305200576782), tf.where(data[i + 4][j + 5]>0, data[i + 10][j + 4], ((data[i + 3][j + 8]) - ((6.0))) ) )))) +
                1.0*tf.tanh(((((data[i + 10][j + 5]) - (((data[i + 9][j + 8]) - (data[i + 10][j + 4]))))) * (((data[i + 9][j + 8]) - (data[i + 7][j + 8]))))) +
                1.0*tf.tanh(((data[i + 8][j + 7]) - (tf.where(((data[i + 8][j + 7]) - ((data[i + 10][j + 7])))>0, data[i + 8][j + 6], data[i + 10][j + 7] )))) +
                1.0*tf.tanh(((data[i + 1][j + 0]) * (((data[i + 0][j + 5]) * (((data[i + 1][j + 0]) * (((data[i + 10][j + 9]) - (((data[i + 1][j + 0]) * (data[i + 7][j + 6]))))))))))) +
                1.0*tf.tanh(((((data[i + 0][j + 10]) - (data[i + 1][j + 2]))) * (((((data[i + 0][j + 10]) + (data[i + 8][j + 2]))) * (((data[i + 8][j + 2]) - (data[i + 4][j + 2]))))))) +
                1.0*tf.tanh(((((data[i + 7][j + 3]) - (data[i + 2][j + 3]))) * (((data[i + 9][j + 3]) - (data[i + 10][j + 9]))))) +
                1.0*tf.tanh(((data[i + 10][j + 6]) - (tf.where(((data[i + 10][j + 6]) - (data[i + 8][j + 6]))>0, tf.where(data[i + 8][j + 6]>0, data[i + 10][j + 6], data[i + 8][j + 6] ), data[i + 8][j + 6] )))) +
                1.0*tf.tanh(((((data[i + 5][j + 4]) - (data[i + 10][j + 10]))) * (((data[i + 7][j + 4]) - (data[i + 10][j + 5]))))) +
                1.0*tf.tanh(tf.where(data[i + 9][j + 6]>0, ((((data[i + 3][j + 5]) * (((data[i + 5][j + 4]) - (data[i + 9][j + 5]))))) * (data[i + 3][j + 5])), data[i + 8][j + 8] )) +
                1.0*tf.tanh(((((((data[i + 9][j + 3]) - (data[i + 7][j + 0]))) * (data[i + 7][j + 3]))) * (data[i + 7][j + 0]))) +
                0.944282*tf.tanh(((((((data[i + 9][j + 2]) + (data[i + 9][j + 3]))) * (data[i + 2][j + 7]))) * (((data[i + 9][j + 2]) * (((data[i + 6][j + 3]) - (data[i + 2][j + 7]))))))) +
                1.0*tf.tanh(((data[i + 0][j + 7]) * (((data[i + 3][j + 7]) - (tf.where(data[i + 9][j + 2]>0, data[i + 5][j + 7], tf.where(data[i + 9][j + 1]>0, data[i + 9][j + 2], data[i + 0][j + 7] ) )))))) +
                1.0*tf.tanh(((((data[i + 0][j + 0]) * (((data[i + 6][j + 7]) - (data[i + 0][j + 0]))))) * (((data[i + 5][j + 5]) * (data[i + 0][j + 0]))))) +
                1.0*tf.tanh(((((data[i + 8][j + 0]) * (((data[i + 5][j + 0]) - (data[i + 8][j + 0]))))) * (((data[i + 8][j + 0]) - (((data[i + 5][j + 0]) - (data[i + 8][j + 0]))))))) +
                1.0*tf.tanh(tf.where(((data[i + 5][j + 2]) - (data[i + 0][j + 2]))>0, (0.02432585321366787), ((data[i + 5][j + 2]) - (data[i + 0][j + 2])) )) +
                1.0*tf.tanh(((tf.where(data[i + 10][j + 4]>0, data[i + 1][j + 2], ((data[i + 5][j + 10]) - (data[i + 0][j + 2])) )) * (((data[i + 0][j + 2]) - (data[i + 6][j + 2]))))) +
                1.0*tf.tanh(((data[i + 9][j + 4]) - (tf.where(((data[i + 9][j + 4]) - (data[i + 10][j + 3]))<0, data[i + 5][j + 5], data[i + 9][j + 4] )))) +
                1.0*tf.tanh(((data[i + 0][j + 0]) * (((data[i + 3][j + 4]) - (data[i + 4][j + 3]))))) +
                0.998045*tf.tanh(((data[i + 5][j + 6]) * (((data[i + 5][j + 6]) * (((data[i + 5][j + 6]) * (((data[i + 5][j + 6]) * (((data[i + 6][j + 6]) - (data[i + 10][j + 9]))))))))))) +
                1.0*tf.tanh(((((data[i + 8][j + 5]) - (data[i + 1][j + 3]))) * (((data[i + 1][j + 3]) * (((data[i + 1][j + 3]) * (((data[i + 6][j + 4]) - (data[i + 1][j + 3]))))))))) +
                0.965787*tf.tanh(tf.where(((data[i + 7][j + 6]) - (((data[i + 9][j + 7]) * (data[i + 0][j + 6]))))<0, (2.0), tf.where(data[i + 7][j + 5]>0, (0.0), data[i + 9][j + 7] ) )) +
                1.0*tf.tanh(((((data[i + 9][j + 0]) - (data[i + 7][j + 0]))) * (((((((data[i + 9][j + 10]) + (data[i + 7][j + 0]))) - (data[i + 9][j + 0]))) - (data[i + 9][j + 0]))))) +
                1.0*tf.tanh(((data[i + 5][j + 1]) * (((data[i + 5][j + 1]) * (((data[i + 5][j + 1]) * (((data[i + 7][j + 1]) - (data[i + 9][j + 4]))))))))) +
                1.0*tf.tanh(((((data[i + 8][j + 7]) - (((data[i + 5][j + 7]) - (data[i + 8][j + 7]))))) * (((data[i + 2][j + 7]) * (((data[i + 5][j + 7]) - (data[i + 2][j + 7]))))))) +
                1.0*tf.tanh(((tf.where(((data[i + 7][j + 4]) - (data[i + 5][j + 4]))<0, data[i + 7][j + 4], data[i + 5][j + 4] )) - (data[i + 7][j + 4]))) +
                1.0*tf.tanh(((((((data[i + 7][j + 5]) - (((data[i + 1][j + 2]) * (data[i + 1][j + 2]))))) * (data[i + 9][j + 5]))) * (((data[i + 1][j + 1]) * (data[i + 1][j + 1]))))) +
                1.0*tf.tanh(((data[i + 9][j + 4]) * (((data[i + 6][j + 4]) - (tf.where(data[i + 3][j + 5]>0, tf.where(data[i + 0][j + 1]>0, data[i + 4][j + 4], data[i + 6][j + 4] ), data[i + 9][j + 4] )))))) +
                1.0*tf.tanh(((((data[i + 8][j + 6]) - (data[i + 3][j + 6]))) * (((data[i + 10][j + 6]) - (data[i + 10][j + 2]))))) +
                1.0*tf.tanh(((data[i + 3][j + 5]) - (tf.where(((data[i + 0][j + 4]) - (data[i + 3][j + 5]))<0, data[i + 3][j + 5], data[i + 0][j + 4] )))) +
                1.0*tf.tanh(((data[i + 0][j + 3]) * (((data[i + 0][j + 3]) - (data[i + 5][j + 4]))))) +
                0.996090*tf.tanh(((data[i + 6][j + 9]) - (tf.where(((data[i + 6][j + 9]) - (data[i + 9][j + 9]))>0, data[i + 6][j + 9], ((data[i + 9][j + 9]) * (data[i + 9][j + 9])) )))) +
                1.0*tf.tanh(((data[i + 5][j + 10]) - (tf.where(((data[i + 4][j + 10]) - (data[i + 8][j + 10]))>0, data[i + 4][j + 10], data[i + 8][j + 10] )))) +
                1.0*tf.tanh(tf.where(data[i + 6][j + 10]<0, data[i + 6][j + 10], ((((data[i + 9][j + 9]) - (data[i + 6][j + 10]))) * (((data[i + 1][j + 3]) - (data[i + 3][j + 9])))) )) +
                0.855327*tf.tanh(((((((data[i + 8][j + 7]) * (data[i + 10][j + 7]))) - (data[i + 5][j + 7]))) * (((data[i + 10][j + 8]) * (data[i + 10][j + 7]))))) +
                1.0*tf.tanh(((((data[i + 4][j + 2]) - (data[i + 7][j + 3]))) * (((data[i + 7][j + 6]) - (data[i + 9][j + 3]))))) +
                1.0*tf.tanh(((data[i + 2][j + 7]) * (((data[i + 3][j + 7]) * (((data[i + 10][j + 0]) * (((data[i + 3][j + 7]) * (((data[i + 0][j + 10]) - (data[i + 10][j + 8]))))))))))) +
                1.0*tf.tanh(((((data[i + 9][j + 4]) - (data[i + 5][j + 5]))) * (((tf.where(data[i + 3][j + 5] > -1, data[i + 0][j + 5], data[i + 5][j + 5] )) - (data[i + 3][j + 5]))))) +
                1.0*tf.tanh(((data[i + 7][j + 6]) * (((data[i + 5][j + 6]) - (tf.where(((data[i + 7][j + 6]) - (data[i + 2][j + 6]))>0, data[i + 7][j + 6], data[i + 2][j + 6] )))))) +
                1.0*tf.tanh(tf.where(((((data[i + 5][j + 1]) - (data[i + 2][j + 0]))) + (data[i + 5][j + 4]))>0, tf.where(data[i + 5][j + 1]>0, (0.0), data[i + 2][j + 1] ), data[i + 2][j + 0] )) +
                0.990225*tf.tanh(((data[i + 1][j + 0]) * (((data[i + 9][j + 5]) * (((data[i + 1][j + 0]) * (((data[i + 6][j + 4]) - (((data[i + 1][j + 0]) * (data[i + 1][j + 0]))))))))))) +
                1.0*tf.tanh(((tf.where(data[i + 1][j + 4]>0, data[i + 1][j + 4], data[i + 4][j + 4] )) - (tf.where(((data[i + 1][j + 4]) - (data[i + 4][j + 4]))>0, data[i + 1][j + 4], data[i + 4][j + 4] )))) +
                1.0*tf.tanh(((data[i + 7][j + 4]) * (((data[i + 1][j + 0]) * (((data[i + 5][j + 4]) - (data[i + 0][j + 3]))))))) +
                1.0*tf.tanh(tf.where(((data[i + 5][j + 7]) + (((data[i + 4][j + 6]) - (data[i + 0][j + 7]))))<0, data[i + 0][j + 7], ((data[i + 6][j + 5]) - (data[i + 6][j + 5])) )) +
                1.0*tf.tanh(((((data[i + 5][j + 6]) - (tf.where(data[i + 9][j + 6]>0, data[i + 9][j + 6], data[i + 3][j + 5] )))) * (((data[i + 3][j + 5]) - (data[i + 9][j + 6]))))))
        
        
        gp2 = ( -1.110189 +
                1.0*tf.tanh(((data[i + 4][j + 4]) - (tf.where(((data[i + 4][j + 4]) - (data[i + 10][j + 4]))<0, data[i + 10][j + 5], data[i + 4][j + 4] )))) +
                1.0*tf.tanh(((((data[i + 9][j + 7]) - (((data[i + 4][j + 7]) * (data[i + 4][j + 7]))))) * (((data[i + 4][j + 7]) - (((data[i + 9][j + 7]) - (data[i + 4][j + 7]))))))) +
                1.0*tf.tanh(((data[i + 2][j + 5]) - (tf.where(((data[i + 2][j + 5]) - (data[i + 8][j + 5]))<0, data[i + 8][j + 5], data[i + 2][j + 5] )))) +
                1.0*tf.tanh(((data[i + 5][j + 1]) - (tf.where(((data[i + 4][j + 1]) - (data[i + 5][j + 1]))>0, data[i + 4][j + 1], ((data[i + 7][j + 2]) * (data[i + 7][j + 1])) )))) +
                1.0*tf.tanh(((tf.where(data[i + 1][j + 3]>0, data[i + 1][j + 2], data[i + 6][j + 1] )) - (tf.where(((data[i + 1][j + 2]) - (data[i + 6][j + 1]))>0, data[i + 1][j + 2], data[i + 6][j + 1] )))) +
                1.0*tf.tanh(((data[i + 5][j + 6]) - (tf.where(((data[i + 5][j + 6]) - (tf.where(data[i + 10][j + 6]>0, data[i + 10][j + 6], data[i + 8][j + 2] )))>0, data[i + 5][j + 6], data[i + 10][j + 6] )))) +
                1.0*tf.tanh(((data[i + 6][j + 6]) * (((data[i + 10][j + 6]) - (tf.where(data[i + 2][j + 5]>0, data[i + 6][j + 6], ((data[i + 1][j + 3]) - (data[i + 10][j + 6])) )))))) +
                1.0*tf.tanh(((((data[i + 4][j + 7]) - (data[i + 0][j + 7]))) * (((((data[i + 0][j + 7]) - (data[i + 4][j + 7]))) - (((data[i + 4][j + 7]) - (data[i + 0][j + 7]))))))) +
                0.978495*tf.tanh(((tf.where(data[i + 3][j + 3]>0, data[i + 3][j + 3], data[i + 10][j + 4] )) * (((data[i + 6][j + 5]) - (((data[i + 3][j + 3]) * (data[i + 3][j + 3]))))))) +
                1.0*tf.tanh(((((data[i + 10][j + 9]) - (data[i + 6][j + 9]))) * (((((data[i + 6][j + 9]) - (((data[i + 2][j + 8]) - (data[i + 6][j + 9]))))) - (data[i + 10][j + 9]))))) +
                1.0*tf.tanh(tf.where(data[i + 9][j + 7]>0, ((data[i + 8][j + 9]) * (((data[i + 8][j + 9]) * (((data[i + 5][j + 8]) - (data[i + 8][j + 9])))))), data[i + 3][j + 7] )) +
                1.0*tf.tanh(((((((data[i + 5][j + 3]) - (((data[i + 9][j + 3]) - (data[i + 5][j + 3]))))) - (data[i + 9][j + 3]))) * (((data[i + 9][j + 3]) - (data[i + 5][j + 3]))))) +
                1.0*tf.tanh(((((data[i + 5][j + 6]) - (data[i + 10][j + 6]))) * (((data[i + 3][j + 5]) - (tf.where(data[i + 10][j + 6]>0, data[i + 5][j + 6], data[i + 10][j + 6] )))))) +
                1.0*tf.tanh(((data[i + 9][j + 1]) - (tf.where(data[i + 5][j + 5]>0, data[i + 9][j + 1], (((0.71906465291976929)) - (((data[i + 5][j + 5]) - ((0.71906465291976929))))) )))) +
                1.0*tf.tanh(((data[i + 4][j + 2]) - (tf.where(data[i + 0][j + 2]>0, tf.where(((data[i + 4][j + 2]) - (data[i + 0][j + 2]))>0, data[i + 4][j + 2], data[i + 0][j + 2] ), data[i + 0][j + 2] )))) +
                0.773216*tf.tanh(((data[i + 4][j + 1]) * (((data[i + 0][j + 2]) - (((data[i + 5][j + 1]) * (data[i + 5][j + 1]))))))) +
                1.0*tf.tanh(((data[i + 5][j + 8]) - (tf.where(((data[i + 5][j + 8]) - (data[i + 9][j + 9]))<0, data[i + 9][j + 9], data[i + 5][j + 8] )))) +
                1.0*tf.tanh(((((data[i + 2][j + 2]) - (data[i + 5][j + 2]))) * (((data[i + 4][j + 2]) - (((data[i + 2][j + 2]) - (((data[i + 5][j + 2]) - (data[i + 9][j + 2]))))))))) +
                1.0*tf.tanh(((((data[i + 10][j + 5]) * (((data[i + 5][j + 8]) - (((data[i + 8][j + 8]) - (data[i + 5][j + 8]))))))) * (((data[i + 8][j + 8]) - (data[i + 5][j + 8]))))) +
                1.0*tf.tanh(((data[i + 8][j + 1]) - (tf.where(((data[i + 10][j + 0]) - (data[i + 8][j + 0]))>0, data[i + 10][j + 0], data[i + 8][j + 1] )))) +
                1.0*tf.tanh(((data[i + 5][j + 10]) - (tf.where(((data[i + 5][j + 10]) - (data[i + 6][j + 10]))>0, data[i + 4][j + 8], ((data[i + 7][j + 10]) * (data[i + 7][j + 10])) )))) +
                1.0*tf.tanh(((((data[i + 0][j + 0]) * (data[i + 9][j + 7]))) * (((((data[i + 10][j + 0]) * (data[i + 4][j + 1]))) - (data[i + 9][j + 7]))))) +
                1.0*tf.tanh(((((data[i + 2][j + 1]) - (data[i + 6][j + 1]))) * (tf.where(data[i + 2][j + 1]>0, data[i + 6][j + 1], ((data[i + 2][j + 1]) - ((data[i + 5][j + 7]))) )))) +
                1.0*tf.tanh(((data[i + 5][j + 5]) - (tf.where(((data[i + 5][j + 5]) - (data[i + 8][j + 5]))<0, data[i + 8][j + 5], data[i + 5][j + 5] )))) +
                1.0*tf.tanh(((data[i + 5][j + 3]) - (tf.where(((data[i + 5][j + 3]) - (data[i + 1][j + 3]))>0, data[i + 5][j + 3], data[i + 1][j + 3] )))) +
                0.930596*tf.tanh(((data[i + 9][j + 5]) * (((((data[i + 10][j + 0]) - (((((data[i + 2][j + 0]) * (data[i + 2][j + 0]))) * (data[i + 2][j + 0]))))) * (data[i + 2][j + 0]))))) +
                1.0*tf.tanh(((data[i + 3][j + 4]) - (tf.where(data[i + 3][j + 4]>0, tf.where(((data[i + 5][j + 4]) - (data[i + 3][j + 4]))>0, data[i + 5][j + 4], data[i + 3][j + 4] ), data[i + 3][j + 4] )))) +
                1.0*tf.tanh(((data[i + 2][j + 7]) - (tf.where(data[i + 5][j + 5]>0, data[i + 2][j + 7], tf.where(data[i + 3][j + 4]>0, data[i + 5][j + 5], (10.0) ) )))) +
                1.0*tf.tanh(((data[i + 7][j + 3]) - (tf.where(((data[i + 7][j + 3]) - (data[i + 8][j + 3]))>0, data[i + 6][j + 3], ((data[i + 9][j + 3]) * (data[i + 9][j + 2])) )))) +
                1.0*tf.tanh(((data[i + 10][j + 3]) * (((data[i + 10][j + 3]) * (((((data[i + 8][j + 5]) * (data[i + 6][j + 2]))) - (data[i + 10][j + 3]))))))) +
                1.0*tf.tanh(((data[i + 8][j + 4]) - (tf.where(((data[i + 8][j + 4]) - (data[i + 10][j + 4]))>0, data[i + 8][j + 4], data[i + 10][j + 4] )))) +
                1.0*tf.tanh(((((((data[i + 3][j + 4]) - (data[i + 8][j + 4]))) - (data[i + 8][j + 4]))) * (((data[i + 8][j + 4]) - (data[i + 10][j + 4]))))) +
                1.0*tf.tanh(tf.where(((data[i + 1][j + 4]) - (data[i + 2][j + 4]))<0, (((10.02525615692138672)) * (((data[i + 1][j + 4]) - (data[i + 2][j + 4])))), (0.0) )) +
                1.0*tf.tanh(((((data[i + 3][j + 4]) - (((data[i + 0][j + 5]) * (data[i + 0][j + 4]))))) * (tf.where(data[i + 6][j + 3]>0, data[i + 6][j + 3], data[i + 1][j + 4] )))) +
                1.0*tf.tanh(((((data[i + 3][j + 4]) - (data[i + 1][j + 4]))) * (tf.where(((data[i + 3][j + 4]) - (data[i + 1][j + 4]))>0, data[i + 2][j + 6], (2.22467470169067383) )))) +
                0.967742*tf.tanh(tf.where(data[i + 5][j + 4]>0, tf.where(data[i + 10][j + 4]>0, ((data[i + 0][j + 3]) * (((data[i + 1][j + 4]) - (data[i + 5][j + 4])))), data[i + 7][j + 9] ), data[i + 0][j + 3] )) +
                0.856305*tf.tanh(((((data[i + 7][j + 10]) + (data[i + 7][j + 9]))) * (((((data[i + 10][j + 9]) - (data[i + 7][j + 10]))) * (data[i + 7][j + 9]))))) +
                1.0*tf.tanh(((data[i + 2][j + 8]) * (((((((data[i + 0][j + 7]) - (data[i + 8][j + 1]))) - (data[i + 8][j + 1]))) * (((data[i + 1][j + 3]) - (data[i + 5][j + 1]))))))) +
                1.0*tf.tanh(((((data[i + 7][j + 6]) - (data[i + 10][j + 6]))) * (((data[i + 4][j + 6]) - (((data[i + 7][j + 6]) - (((data[i + 4][j + 6]) - (data[i + 7][j + 6]))))))))) +
                1.0*tf.tanh(((((data[i + 10][j + 10]) - (((((data[i + 9][j + 9]) - (data[i + 10][j + 10]))) - (data[i + 4][j + 9]))))) * (((data[i + 9][j + 10]) - (data[i + 10][j + 10]))))) +
                1.0*tf.tanh(((((((data[i + 2][j + 7]) - (((data[i + 0][j + 7]) - (data[i + 2][j + 7]))))) - (data[i + 0][j + 7]))) * (((data[i + 5][j + 7]) - (data[i + 2][j + 7]))))) +
                1.0*tf.tanh(((((((data[i + 0][j + 4]) - (data[i + 2][j + 5]))) - (((data[i + 2][j + 3]) - (data[i + 0][j + 4]))))) * (((data[i + 2][j + 5]) - (data[i + 5][j + 5]))))) +
                0.639296*tf.tanh(tf.where(((data[i + 1][j + 5]) - (data[i + 7][j + 5]))>0, (0.0), ((data[i + 1][j + 5]) - (data[i + 7][j + 5])) )) +
                1.0*tf.tanh((((0.20925526320934296)) * (tf.where(((data[i + 0][j + 4]) - (data[i + 8][j + 5]))<0, data[i + 10][j + 5], (((0.20925526320934296)) - (data[i + 10][j + 6])) )))) +
                1.0*tf.tanh(((((((data[i + 10][j + 3]) - (data[i + 7][j + 2]))) * (((data[i + 4][j + 2]) + (data[i + 4][j + 2]))))) * (((data[i + 7][j + 2]) - (data[i + 4][j + 2]))))) +
                1.0*tf.tanh(((tf.where(((data[i + 3][j + 0]) - (data[i + 0][j + 0]))<0, data[i + 3][j + 0], data[i + 0][j + 0] )) - (data[i + 0][j + 0]))) +
                1.0*tf.tanh(((data[i + 0][j + 0]) - (tf.where(((data[i + 7][j + 1]) - (data[i + 4][j + 0]))>0, data[i + 0][j + 0], tf.where(data[i + 0][j + 0]>0, data[i + 4][j + 0], data[i + 0][j + 0] ) )))) +
                0.999022*tf.tanh(((data[i + 3][j + 4]) * (((data[i + 3][j + 4]) * (((data[i + 3][j + 4]) * (((data[i + 6][j + 3]) - (((data[i + 10][j + 4]) * (data[i + 3][j + 5]))))))))))) +
                0.984360*tf.tanh(((((data[i + 8][j + 3]) - (((data[i + 5][j + 4]) - (data[i + 8][j + 3]))))) * (((data[i + 10][j + 2]) - (data[i + 8][j + 3]))))) +
                1.0*tf.tanh(((((data[i + 1][j + 2]) - (data[i + 5][j + 3]))) * (((data[i + 0][j + 3]) - (data[i + 2][j + 3]))))) +
                0.954057*tf.tanh(tf.where(data[i + 10][j + 5]>0, ((data[i + 0][j + 8]) * (((data[i + 3][j + 8]) - (data[i + 4][j + 5])))), ((data[i + 7][j + 5]) * (data[i + 8][j + 6])) )) +
                1.0*tf.tanh(((((data[i + 10][j + 1]) * (((data[i + 10][j + 1]) - (((data[i + 7][j + 2]) - (data[i + 10][j + 1]))))))) * (((data[i + 7][j + 2]) - (data[i + 10][j + 1]))))) +
                1.0*tf.tanh(((data[i + 10][j + 8]) * (((data[i + 5][j + 5]) * (((data[i + 5][j + 5]) * (((data[i + 7][j + 6]) - (data[i + 3][j + 8]))))))))) +
                1.0*tf.tanh(((((data[i + 5][j + 4]) - ((data[i + 7][j + 2])))) * (((data[i + 7][j + 3]) - (data[i + 9][j + 0]))))) +
                0.784946*tf.tanh(tf.where(((data[i + 10][j + 1]) - (data[i + 9][j + 1]))<0, ((data[i + 10][j + 4]) - (data[i + 8][j + 4])), ((data[i + 7][j + 2]) - (data[i + 6][j + 2])) )) +
                1.0*tf.tanh(((((((data[i + 7][j + 4]) - (data[i + 10][j + 4]))) * (((data[i + 10][j + 4]) - (((data[i + 7][j + 4]) - (data[i + 10][j + 4]))))))) * (data[i + 10][j + 4]))) +
                1.0*tf.tanh(((((((data[i + 6][j + 2]) - (((data[i + 10][j + 2]) - (data[i + 7][j + 2]))))) - (data[i + 10][j + 2]))) * (((data[i + 5][j + 2]) - (data[i + 7][j + 2]))))) +
                1.0*tf.tanh(tf.where(data[i + 0][j + 7]>0, tf.where(((data[i + 5][j + 6]) + (((data[i + 4][j + 8]) - (data[i + 0][j + 7]))))<0, data[i + 0][j + 8], (0.03671050816774368) ), data[i + 0][j + 8] )) +
                1.0*tf.tanh(((tf.where(((data[i + 7][j + 7]) - (data[i + 4][j + 7]))<0, data[i + 7][j + 7], data[i + 4][j + 7] )) - (data[i + 7][j + 7]))) +
                1.0*tf.tanh(((data[i + 7][j + 7]) - (tf.where(((data[i + 7][j + 7]) - (data[i + 10][j + 8]))>0, data[i + 7][j + 7], data[i + 10][j + 8] )))) +
                1.0*tf.tanh(((data[i + 10][j + 3]) * (((tf.where(((data[i + 10][j + 3]) - (data[i + 10][j + 8]))>0, data[i + 10][j + 8], data[i + 10][j + 3] )) - (data[i + 4][j + 7]))))) +
                1.0*tf.tanh(((data[i + 0][j + 3]) * (((data[i + 2][j + 3]) - (tf.where(((data[i + 1][j + 3]) - (data[i + 0][j + 3]))>0, data[i + 2][j + 3], data[i + 7][j + 3] )))))) +
                0.999022*tf.tanh((((((0.19441728293895721)) + ((0.19441728293895721)))) * ((((((0.19441370666027069)) + ((0.19441370666027069)))) - (((data[i + 3][j + 9]) * (data[i + 1][j + 4]))))))) +
                1.0*tf.tanh(((((data[i + 10][j + 1]) - (data[i + 4][j + 8]))) * (((tf.where(data[i + 10][j + 1]>0, data[i + 9][j + 7], data[i + 1][j + 7] )) - (data[i + 6][j + 8]))))) +
                1.0*tf.tanh(((data[i + 0][j + 10]) * (((((data[i + 3][j + 8]) - (data[i + 0][j + 10]))) * (((data[i + 0][j + 10]) - (((data[i + 3][j + 9]) - (data[i + 0][j + 8]))))))))) +
                0.924731*tf.tanh(((((data[i + 1][j + 0]) + (data[i + 6][j + 10]))) * (((data[i + 0][j + 0]) * (((data[i + 10][j + 1]) * (((data[i + 0][j + 10]) - (data[i + 1][j + 0]))))))))) +
                1.0*tf.tanh(((data[i + 8][j + 3]) * (((((data[i + 8][j + 2]) - (((data[i + 6][j + 2]) - (data[i + 8][j + 2]))))) * (((data[i + 6][j + 3]) - (data[i + 8][j + 2]))))))) +
                1.0*tf.tanh(((((data[i + 10][j + 5]) - (data[i + 5][j + 3]))) * (((data[i + 0][j + 3]) - (data[i + 3][j + 3]))))) +
                1.0*tf.tanh(((((data[i + 7][j + 9]) - (data[i + 10][j + 9]))) * (tf.where(data[i + 10][j + 9]>0, ((data[i + 10][j + 10]) - (data[i + 7][j + 8])), data[i + 9][j + 7] )))) +
                1.0*tf.tanh(((((data[i + 8][j + 0]) - (data[i + 10][j + 0]))) * (((data[i + 10][j + 0]) - (((data[i + 8][j + 0]) - (((data[i + 5][j + 0]) - (data[i + 8][j + 0]))))))))) +
                1.0*tf.tanh(((data[i + 5][j + 5]) * (((((data[i + 8][j + 5]) - (data[i + 10][j + 5]))) * (((data[i + 4][j + 5]) - (((data[i + 8][j + 5]) - (data[i + 5][j + 5]))))))))) +
                1.0*tf.tanh(tf.where(data[i + 7][j + 3]>0, ((data[i + 10][j + 2]) * (((data[i + 5][j + 3]) * (((data[i + 7][j + 3]) - (data[i + 4][j + 5])))))), data[i + 10][j + 2] )) +
                1.0*tf.tanh(((((data[i + 10][j + 5]) * (((data[i + 10][j + 10]) - (data[i + 8][j + 10]))))) * (((data[i + 8][j + 10]) - (((data[i + 10][j + 10]) - (data[i + 8][j + 10]))))))) +
                1.0*tf.tanh(((((((data[i + 0][j + 0]) - (data[i + 6][j + 2]))) * (tf.where(data[i + 0][j + 0]>0, data[i + 10][j + 10], (0.0) )))) * (data[i + 10][j + 10]))) +
                1.0*tf.tanh(((((data[i + 10][j + 1]) - (((data[i + 7][j + 3]) - (data[i + 10][j + 1]))))) * (((data[i + 10][j + 2]) * (((data[i + 7][j + 2]) - (data[i + 10][j + 2]))))))) +
                1.0*tf.tanh(((data[i + 9][j + 2]) - (tf.where(((tf.where(data[i + 9][j + 2]>0, data[i + 8][j + 2], data[i + 10][j + 1] )) - (data[i + 10][j + 1]))>0, data[i + 8][j + 2], data[i + 10][j + 1] )))) +
                0.998045*tf.tanh(((((data[i + 0][j + 7]) - (((data[i + 7][j + 5]) * (data[i + 0][j + 7]))))) * (((data[i + 10][j + 0]) - (((data[i + 0][j + 8]) * (data[i + 6][j + 6]))))))) +
                1.0*tf.tanh(((data[i + 0][j + 10]) - (tf.where(((data[i + 7][j + 5]) - (data[i + 2][j + 10]))>0, data[i + 0][j + 10], tf.where(data[i + 3][j + 5]>0, data[i + 1][j + 10], (3.01419210433959961) ) )))) +
                1.0*tf.tanh(((((((data[i + 4][j + 7]) * (((data[i + 4][j + 7]) * (data[i + 4][j + 7]))))) * (data[i + 4][j + 7]))) * (((data[i + 4][j + 7]) - (data[i + 10][j + 7]))))) +
                1.0*tf.tanh(((((data[i + 7][j + 9]) * (((data[i + 0][j + 10]) * (((data[i + 3][j + 10]) - (data[i + 0][j + 10]))))))) * (((data[i + 0][j + 10]) + (data[i + 0][j + 10]))))) +
                1.0*tf.tanh(((tf.where(((data[i + 5][j + 0]) - (data[i + 3][j + 0]))>0, data[i + 3][j + 0], data[i + 5][j + 0] )) - (data[i + 3][j + 0]))) +
                0.999022*tf.tanh(((data[i + 0][j + 0]) * (((data[i + 3][j + 0]) - (data[i + 5][j + 1]))))) +
                1.0*tf.tanh(tf.where(((data[i + 5][j + 6]) + (((data[i + 5][j + 7]) - (data[i + 2][j + 8]))))<0, data[i + 2][j + 8], (0.0) )) +
                1.0*tf.tanh(((data[i + 6][j + 8]) - (tf.where(((data[i + 6][j + 8]) * (((data[i + 6][j + 8]) - (data[i + 1][j + 8]))))<0, data[i + 1][j + 8], data[i + 6][j + 8] )))) +
                0.999022*tf.tanh(((((data[i + 3][j + 6]) - (((data[i + 5][j + 8]) * (data[i + 5][j + 8]))))) * (((data[i + 1][j + 9]) - (((data[i + 6][j + 10]) * (data[i + 5][j + 3]))))))) +
                1.0*tf.tanh(((((data[i + 6][j + 8]) - (data[i + 9][j + 7]))) * (((data[i + 3][j + 7]) - (tf.where(data[i + 9][j + 7]>0, data[i + 6][j + 7], data[i + 2][j + 5] )))))) +
                1.0*tf.tanh(tf.where(((((data[i + 4][j + 3]) - (data[i + 1][j + 3]))) + (data[i + 5][j + 3]))>0, (0.0), data[i + 0][j + 2] )) +
                1.0*tf.tanh(((((data[i + 7][j + 10]) - (((data[i + 5][j + 10]) - (((data[i + 3][j + 10]) - (data[i + 5][j + 10]))))))) * (((data[i + 5][j + 10]) - (data[i + 7][j + 10]))))) +
                1.0*tf.tanh(((((data[i + 5][j + 8]) - (data[i + 8][j + 8]))) * (((((data[i + 7][j + 9]) * (data[i + 8][j + 8]))) - (data[i + 10][j + 9]))))) +
                0.998045*tf.tanh(((((data[i + 6][j + 3]) - (data[i + 7][j + 3]))) * (((data[i + 3][j + 3]) - (((data[i + 5][j + 4]) - (data[i + 3][j + 3]))))))) +
                1.0*tf.tanh(((((((data[i + 1][j + 0]) - (((data[i + 10][j + 6]) - (data[i + 0][j + 2]))))) * (((data[i + 4][j + 3]) - (data[i + 1][j + 0]))))) * (data[i + 1][j + 8]))) +
                1.0*tf.tanh(((((data[i + 9][j + 7]) + (data[i + 10][j + 5]))) * (((((data[i + 9][j + 7]) - (data[i + 6][j + 7]))) * (((data[i + 6][j + 7]) - (data[i + 9][j + 7]))))))) +
                1.0*tf.tanh(((((data[i + 5][j + 7]) - (data[i + 6][j + 7]))) * (((((data[i + 6][j + 8]) - (data[i + 9][j + 7]))) - (((data[i + 9][j + 7]) - (data[i + 6][j + 8]))))))) +
                1.0*tf.tanh(((((((data[i + 8][j + 4]) - (data[i + 5][j + 4]))) * (((data[i + 10][j + 4]) + (data[i + 10][j + 4]))))) * (((data[i + 5][j + 4]) - (data[i + 8][j + 4]))))) +
                1.0*tf.tanh(((data[i + 2][j + 3]) * (((((((data[i + 8][j + 2]) - (data[i + 2][j + 4]))) * (((data[i + 6][j + 1]) - (data[i + 2][j + 4]))))) * (data[i + 2][j + 4]))))) +
                0.999022*tf.tanh(((data[i + 6][j + 9]) * (((data[i + 1][j + 8]) * (((data[i + 4][j + 9]) - (data[i + 6][j + 8]))))))) +
                1.0*tf.tanh(((((((data[i + 8][j + 10]) - (data[i + 5][j + 10]))) * (data[i + 9][j + 7]))) * (((data[i + 5][j + 10]) - (((data[i + 8][j + 10]) - (data[i + 5][j + 10]))))))) +
                1.0*tf.tanh(((data[i + 1][j + 8]) * (((data[i + 1][j + 8]) * (((((data[i + 6][j + 6]) - (data[i + 0][j + 6]))) * (((data[i + 1][j + 8]) * (data[i + 8][j + 5]))))))))) +
                1.0*tf.tanh(((data[i + 6][j + 10]) - (tf.where(((data[i + 9][j + 10]) - (data[i + 6][j + 10]))>0, data[i + 9][j + 10], data[i + 6][j + 10] )))) +
                1.0*tf.tanh(((((data[i + 6][j + 2]) - (data[i + 4][j + 5]))) * (((data[i + 8][j + 1]) - (tf.where(data[i + 10][j + 2]>0, data[i + 6][j + 6], data[i + 10][j + 2] )))))))
        
        return tf.sigmoid(.5*gp1+.5*gp2)


    predictions = tf.map_fn(lambda a: GP(C[a],
                                         R[a]),
                            indices,
                            dtype=tf.float32,
                            parallel_iterations=10,
                            swap_memory=False)
    return tf.reshape(predictions,(width,height))
predictions = None
graph = tf.Graph()
with tf.Session(graph=graph) as session:
    predictions = (MyTensorFlow(img).eval())
predictions.shape
_ = plt.imshow(predictions)
img_mask = load_img('../input/train/masks/' + f + '.png')
img_mask = img_to_array(img_mask)
img_mask = img_mask[:,:,0]
_ = plt.imshow(img_mask)