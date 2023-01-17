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
import pandas as pd

deliveries = pd.read_csv("../input/ipldata/deliveries.csv")

matches = pd.read_csv("../input/ipldata/matches.csv")

matches['city'][0:3]
import pandas as pd

StudentData = pd.read_csv("../input/studentdata/StudentData.csv")



StudentData.head(10)



%matplotlib inline

import matplotlib.pyplot as plt



plt.hist(StudentData['marks(out of 100)'],color='g')

plt.xlabel('marks out of 100')

plt.ylabel("Number of Students")



matches.isnull().any()




y = matches['winner']

X = matches.drop(['winner'],axis=1)



import seaborn as sns

matches.boxplot()