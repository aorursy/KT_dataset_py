import os

os.listdir('../input/')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



sub = None

for dirname, _, filenames in os.walk('/kaggle/input/ultimaincercare/'):

    for filename in filenames:

        filename = os.path.join(dirname, filename)

        print(filename)

        if sub is None:

            sub = pd.read_csv(filename)

        else:

            sub.meter_reading += pd.read_csv(filename, usecols=['meter_reading']).meter_reading

    sub.meter_reading = sub.meter_reading.clip(lower=0.0) / len(filenames)



sub.describe()
sub.to_csv(f'submission-final-1.csv', index=False, float_format='%g')
from IPython.display import FileLink

FileLink('submission-final-1.csv')