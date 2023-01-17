import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import numpy as np

import os
print(os.listdir("../input/"))
image = cv2.resize(cv2.cvtColor(cv2.imread("../input/motif4.jpg"), cv2.COLOR_BGR2RGB),(224,224))

plt.imshow(image)
img = cv2.resize(cv2.cvtColor(cv2.imread("../input/motif4.jpg"), cv2.COLOR_BGR2GRAY),(224,224))

plt.imshow(img)

imgR = np.reshape(img,[1,224,224,1])

filt_h = [[1,2,1],

        [0,0,0],

        [-1,-2,-1]]

filt_v = [[1,0,-1],

        [2,0,-2],

        [1,0,-1]]

filt_h = np.reshape(filt_h,[3,3,1,1])

# print(filt_h)

filt_v = np.reshape(filt_v,[3,3,1,1])

filt_h= tf.constant(filt_h,dtype=tf.float32,)

filt_v= tf.constant(filt_v,dtype=tf.float32,)

x = tf.placeholder(tf.float32, [None,224,224,1])


init = tf.global_variables_initializer()
imgConv_h = tf.nn.conv2d(x,filter=filt_h,strides=[1,1,1,1], padding="SAME")

imgConv_v = tf.nn.conv2d(x,filter=filt_v,strides=[1,1,1,1], padding="SAME")
sess = tf.Session()

sess.run(init)

out_h = sess.run(imgConv_h, feed_dict={x:imgR})

out_h = np.reshape(out_h,[224,224,1])

out_v = sess.run(imgConv_v, feed_dict={x:imgR})

out_v = np.reshape(out_v,[224,224,1])
outRGB_h = cv2.cvtColor(out_h,cv2.COLOR_GRAY2RGB)
outRGB_v = cv2.cvtColor(out_v,cv2.COLOR_GRAY2RGB)
plt.imshow(outRGB_h)
plt.imshow(outRGB_v)
outRGB_h.shape
out_v.shape
result_length = ((out_v**2) + (out_h**2))**0.5  

result_length_rgb=cv2.cvtColor(out_v,cv2.COLOR_GRAY2RGB)

plt.imshow(result_length_rgb)
result_angle = (np.arctan(outRGB_v/(outRGB_h+0.00000001)))#*(2*math.pi)

plt.imshow(result_angle, cmap='hot')
#normalize like crazy

result_length_norm = (result_length_rgb + (np.min(result_length_rgb)*-1) ) / (np.min(result_length_rgb)*-1 + np.max(result_length_rgb))

result_angle_norm = result_angle

result_length_norm.shape

result_red = np.absolute(result_length_norm * np.cos(result_angle_norm+4.2))

result_green = np.absolute(result_length_norm * np.cos(result_angle_norm+2.1))

result_blue = np.absolute(result_length_norm * np.cos(result_angle_norm))

result_rgb = np.zeros((224,224,3)) 

result_r = (result_red + (np.min(result_red)*-1) ) / (np.min(result_red)*-1 + np.max(result_red)) 

result_g = (result_green + (np.min(result_green)*-1) ) / (np.min(result_green)*-1 + np.max(result_green)) 

result_b = (result_blue + (np.min(result_blue)*-1) ) / (np.min(result_blue)*-1 + np.max(result_blue))

# #result_rgb

plt.imshow(result_g)
result_red[223][0]
result_blue[0][0]
result_green[0][0][1]
import itertools

for x, y, z in itertools.product(range(224),range(224),range(3)):

    if z==0:

        result_rgb[x][y][z] = result_r[x][y][z]

    elif z==1:

        result_rgb[x][y][z] = result_g[x][y][z]

    else:

        result_rgb[x][y][z] = result_b[x][y][z]

        
plt.imshow(result_rgb)