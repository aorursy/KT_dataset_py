import os
print(os.listdir('../input'))
print(os.listdir('../input/exported-models/light_shinkai_SavedModel'))
print(os.listdir('../input/input-images'))
m_path   = '../input/exported-models/light_shinkai_SavedModel/'
img_path = '../input/input-images/'
import numpy as np
from imageio import imwrite
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
model = tf.saved_model.load(m_path)
cartoonGAN = model.signatures["serving_default"]
filename = 'train.jpg'
img = np.array(Image.open(img_path+filename).convert("RGB"))
img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
out = cartoonGAN(tf.constant(img))['output_1']
out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
plt.imshow(out)
plt.imsave('CartoonGan_out.jpg',out)

filename = 'memepower.jpg'
img = np.array(Image.open(img_path+filename).convert("RGB"))
img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
out = cartoonGAN(tf.constant(img))['output_1']
out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
plt.imshow(out)
