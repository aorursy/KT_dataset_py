!mkdir -p /tmp/pip/cache/

!cp -r ../input/tf-1140/ /tmp/pip/cache/tensorflow14

!pip install --no-index --find-links /tmp/pip/cache/tensorflow14 tensorflow==1.14.0
# Check tensorflow version

import tensorflow as tf

print('Current tensorflow version is {}'.format(tf.__version__))