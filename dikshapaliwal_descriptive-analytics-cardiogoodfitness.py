# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print(filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv")
data.head()
data.describe(include ="all")  
#mean>median therefore, here we have right skewed data.
data.info()
import matplotlib.pyplot as plt
%matplotlib inline

data.hist(figsize = (20,30))
import seaborn as sns

sns.boxplot(x = "Gender", y = "Age", data = data)
pd.crosstab(data['Product'],data['Gender'])  #gives count of which product how many females or men were using
pd.crosstab(data['Product'],data['Age'])
pd.crosstab(data['Product'],data['MaritalStatus'])