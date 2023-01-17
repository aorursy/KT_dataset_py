# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline


meta_data = pd.read_csv('../input/training_set_metadata.csv')
test_meta_data = pd.read_csv('../input/test_set_metadata.csv')
galactic_cut = meta_data['hostgal_specz'] == 0
test_galactic_cut = test_meta_data['hostgal_photoz'] == 0




plt.hist(meta_data[~galactic_cut]["hostgal_photoz"], 14, (0, 3.2))
plt.xlabel("redshift(photo_z)")
plt.ylabel("counts")
plt.show()

plt.hist(meta_data[~galactic_cut]["hostgal_specz"], 14, (0, 3.2))
plt.xlabel("redshift(spec_z)")
plt.ylabel("counts")
plt.show()
plt.hist(test_meta_data[~test_galactic_cut]["hostgal_photoz"], 14, (0, 3.2))
plt.xlabel("redshift(photo_z)")
plt.ylabel("counts")
plt.show()

plt.hist(test_meta_data[~test_galactic_cut]["hostgal_specz"], 14, (0, 3.2))
plt.xlabel("redshift(spec_z)")
plt.ylabel("counts")
plt.show()
#meta_data[meta_data["hostgal_photoz"] > 2.5]

meta_data[meta_data["hostgal_specz"] > 2]
