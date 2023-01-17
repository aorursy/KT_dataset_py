# install if kernel has internet access

# !pip install jupyternotify
# Load the extension!

%load_ext jupyternotify
%%notify

import time

time.sleep(10)
%%notify -m 'finished sleeping'

import time

time.sleep(25)