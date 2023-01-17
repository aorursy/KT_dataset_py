!pip install albumentations > /dev/null
import albumentations
albumentations.__version__
from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
[d.name for d in local_device_protos if d.device_type == 'GPU']
