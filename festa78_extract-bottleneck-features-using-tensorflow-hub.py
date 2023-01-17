"""Script to extract bottleneck features using Tensorflow-hub.
cf. https://www.tensorflow.org/hub/
"""

import os

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# TF hub module path.
TFHUB_MODULE = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
# Directory to the downloaded image data.
PARENT_PATH = '../input/'
IMAGE_DIR = os.path.join(PARENT_PATH, 'googlelandmark-sampledata')
BOTTLENECK_DIR = ''
# Read image list.
image_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
print(image_list)
# Set up the pre-trained graph.
# This fails because of kaggle issue
module_spec = hub.load_module_spec(TFHUB_MODULE)
height, width = hub.get_expected_image_size(module_spec)
with tf.Graph().as_default() as graph:
    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
    m = hub.Module(module_spec)
    bottleneck_tensor = m(resized_input_tensor)
def add_jpeg_decoding(module_spec):                                       
    """Adds operations that perform JPEG decoding and resizing to the graph.

    Args:
    module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
    return jpeg_data, resized_image
with tf.Session(graph=graph) as sess:
    # Initialize all weights: for the module to their pretrained values,
    # and for the newly added retraining layer to random initial values.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

    # Compute bottleneck features.
    for image_path in image_list:
        bottleneck_path = os.path.join(BOTTLENECK_DIR,
                                       os.path.splitext(os.path.basename(image_path))[0]
                                       + '.txt')
        print('Creating bottleneck at ' + bottleneck_path)
        if not tf.gfile.Exists(image_path):
            print('File does not exist:', image_path)
            raise FileNotFoundError
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        try:
            # First decode the JPEG image, resize it, and rescale the pixel values.
            resized_input_values = sess.run(decoded_image_tensor,
                                            {jpeg_data_tensor: image_data})
            # Then run it through the recognition network.
            bottleneck_values = sess.run(bottleneck_tensor,
                                         {resized_input_tensor: resized_input_values})
            bottleneck_values = np.squeeze(bottleneck_values)
        except Exception as e:
            raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                         str(e)))
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
