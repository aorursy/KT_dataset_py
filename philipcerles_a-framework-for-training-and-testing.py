# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
project_data = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,parse_dates=True,engine='python')
donation_data = pd.read_csv('../input/Donations.csv', error_bad_lines=False)
num_donations_per_donor = donation_data.groupby('Donor ID')['Donor Cart Sequence'].max()
print("{:10.2f}% of donors gave only once".format((num_donations_per_donor == 1).mean() * 100))
print("{:10.2f}% of donors gave 5 or fewer times".format((num_donations_per_donor <= 5).mean() * 100))
