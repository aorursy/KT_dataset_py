import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
def _read_to_tensor(fname, output_height=256, output_width=512, normalize_data=False):
    '''Original images are 256 x 512 x 3. Left half is original image, right is semantic seg'''
    img_strings = tf.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings)
    output = tf.image.resize_images(imgs_decoded, [output_height, output_width])
    if normalize_data:
        output = (output - 128) / 128
    return output

def _get_left_img_half(inp, width=256):
    return inp[:, :width, :]

def _get_right_img_half(inp, width=256):
    return inp[:, width:, :]
    
img_dir = '../input/cityscapes_data/cityscapes_data/train'
file_list = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, fname) for fname in file_list]

# Start with a dataset of directory names.
output_height = 256
output_width = 256
my_data = tf.data.Dataset.from_tensor_slices(img_paths)
img_tensors = my_data.map(_read_to_tensor)
left_imgs = img_tensors.map(_get_left_img_half)
right_imgs = img_tensors.map(_get_right_img_half)
left_batches = tfe.Iterator(left_imgs)  # outside of TF Eager, we would use make_one_shot_iterator
right_batches = tfe.Iterator(right_imgs)
n_images_to_show = 5

for i in range(n_images_to_show):
    left_img = left_batches.next().numpy().astype(np.uint8)
    right_img = right_batches.next().numpy().astype(np.uint8)
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(left_img)
    fig.add_subplot(1,2,2)
    plt.imshow(right_img)
    plt.show()