# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
dataset= [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107, 10,13,12,14,12,108,12,11,14,13,15,10,15,12,10,14,13,15,10]
outliers = []

def detect_outliers(data):

    threshold = 3

    mean = np.mean(data)

    std = np.std(data)

    

    for i in data:

        z_score = (i-mean)/std

        if np.abs(z_score)>threshold:

            outliers.append(i)

    return outliers
outliers_pts = detect_outliers(dataset)

print (outliers_pts)
sorted(dataset)
q1, q3 = np.percentile(dataset,[25,75])

print(q1,q3)
iqr_value = q3-q1

print(iqr_value)
lower_bound_value = q1-(1.5*iqr_value)

upper_bound_value = q3+(1.5*iqr_value)

print(lower_bound_value, upper_bound_value)