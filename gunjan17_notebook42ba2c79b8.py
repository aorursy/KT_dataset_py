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
import seaborn as sns

from sklearn import svm

import matplotlib.pyplot as plt

% matplotlib inline 

data = pd.read_csv('../input/adult-training.csv')
data.describe()
data.head()
Type_employee = data.groupby("White")['39'].count().reset_index().sort_values(by='39',ascending=False).reset_index(drop=True)