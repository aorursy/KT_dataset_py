# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.misc import imread

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



sub_folders = check_output(["ls", "../input/train/"]).decode("utf8").strip().split('\n')

count_dict = {}

for sub_folder in sub_folders:

    num_of_files = len(check_output(["ls", "../input/train/"+sub_folder]).decode("utf8").strip().split('\n'))

    print("Number of files for the species",sub_folder,":",num_of_files)

    count_dict[sub_folder] = num_of_files

    

plt.figure(figsize=(12,4))

sns.barplot(list(count_dict.keys()), list(count_dict.values()), alpha=0.8)

plt.xlabel('Fish Species', fontsize=12)

plt.ylabel('Number of Images', fontsize=12)

plt.show()
subpd = pd.read_csv('../input/sample_submission_stg1.csv')

subpd.to_csv('sub.csv', index=False)

print(subpd.head(5))