%matplotlib inline
import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
psp_base_dir = os.path.join('..', 'input')
psp_model_dir = os.path.join(psp_base_dir, 'model', 'model')
cityscape_weights = os.path.join(psp_base_dir, 'model', 'model', 'pspnet101-cityscapes')
psp_code_dir = os.path.join(psp_base_dir, 'pspnet-tensorflow-master', 'PSPNet-tensorflow-master')
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
sys.path.append(psp_code_dir)
from model import PSPNet101, PSPNet50
from tools import *
# TODO: Change these values to where your model files are
ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, 
                'model': PSPNet50,
                'weights_path': os.path.join(psp_model_dir, 'pspnet50-ade20k/model.ckpt-0')}

cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': os.path.join(psp_model_dir,'pspnet101-cityscapes/model.ckpt-0')}

IMAGE_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
# TODO: If you want to inference on indoor data, change this value to `ADE20k_param`
param = cityscapes_param 
# pre-proecess image
image_path = os.path.join(psp_code_dir, 'input/test1.png')
img_np, filename = load_img(image_path)
img_shape = tf.shape(img_np)
h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
img = preprocess(img_np, h, w)
plt.imshow(imread(image_path))
# Create network.
PSPNet = param['model']
net = PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])
raw_output = net.layers['conv6']

# Predictions.
raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
raw_output_up = tf.argmax(raw_output_up, dimension=3)
pred = decode_labels(raw_output_up, img_shape, param['num_classes'])

# Init tf Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)

ckpt_path = param['weights_path']
loader = tf.train.Saver(var_list=tf.global_variables())
loader.restore(sess, ckpt_path)
print("Restored model parameters from {}".format(ckpt_path))
%%time
# Run and get result image
preds = sess.run(pred)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 30))
ax1.imshow(imread(image_path))
ax1.axis('off')

ax2.imshow(preds[0]/255.0)
ax2.axis('off')
