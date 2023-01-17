import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Install Viz Library

!pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class
df = pd.read_csv('/kaggle/input/running-log-insight/activity_log.csv')
df.head()
AV = AutoViz_Class()
df = AV.AutoViz('/kaggle/input/running-log-insight/activity_log.csv')