import tensorflow as tf

print(tf.__version__)
!pip install tfa-nightly
import tensorflow_addons as tfa
tf.debugging.set_log_device_placement(True)



x = tf.constant([[0.5, 1.2, -0.3]])

layer = tfa.layers.GELU()

print(layer(x))
