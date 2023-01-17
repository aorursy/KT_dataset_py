# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
filenames = check_output(["ls", "../input"]).decode("utf8").strip().split('\n')

# Any results you write to the current directory are saved as output.
filenames
df = pd.read_csv('../input/basic_data_files.csv', error_bad_lines= False)
#https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
df.head()

