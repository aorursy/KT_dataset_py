from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())

# Check GPU availability
if (tf.test.is_gpu_available()) :
    print("GPU is available")
else:
    print("GPU not available")

# Get the GPU name
print(tf.test.gpu_device_name())