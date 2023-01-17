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
import dicom
import multiprocessing

import psutil; psutil.cpu_percent() 

cp = multiprocessing.cpu_count()



mmory = psutil.virtual_memory();divd = 1e9

#print('This computer has {0} Total RAM and {1} Available'.format(mmory[0]/1e9,cp))

print(mmory)

print('This computer has {0} physical cores and {1} Logical cores'.format(cp/2,cp))
pyco = check_output(["ls", "../input"]).decode("utf8")