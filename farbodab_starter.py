from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

tests = pd.read_csv('/kaggle/input/covidtests');tests.head()
cases = pd.read_csv('/kaggle/input/covid');cases.head()
international = pd.read_csv('/kaggle/input/internationaldata');international.head()