# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/ppi"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/ppi"]).decode("utf8").split('\n')
print(filenames)
df = pd.read_csv("../input/ppi/" + filenames[25], thousands=",")
meanvalue = df.groupby('year')['value'].mean()
meanvalue.plot(x = 'year', y = 'value')