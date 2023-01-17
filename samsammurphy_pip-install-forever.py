# example package

!pip install art --target=/kaggle/working/mysitepackages
# add to system path

import sys

sys.path.append('/kaggle/working/mysitepackages')
# run an example

from art import art

art('coffee')
!ls /kaggle/working/mysitepackages/