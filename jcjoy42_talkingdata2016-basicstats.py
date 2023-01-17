# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
group_df = pd.read_csv('../input/gender_age_train.csv')

group_df.info() # Get a basic summary of the data
import matplotlib.pyplot as plt
cnts = group_df['group'].value_counts(normalize = True)
plt.figure()
cnts.plot.bar()
# Let's Look and see if there is any bias in device type amongst the different groups. To do this,
# we will first have to import the device data into a new dataframe
device_model = pd.read_csv('../input/phone_brand_device_model.csv',encoding = 'utf-8')
device_model.head(100)
#cnts = group_df['device_id'][group_df['group'] == 'M23-26']

