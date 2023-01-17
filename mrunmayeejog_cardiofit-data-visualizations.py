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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cdata = pd.read_csv("/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv")
cdata.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Descriptive stats
cdata.describe()
# We can see below there is positive correlation between customer's predicted usage and miles 
# As well as self rated fitness level and Miles, Education and income
cc = cdata.corr()
sns.heatmap(cc, annot = True)
# Higher the Education years more is the income and preference for TM798 also increases. 
sns.barplot(x='Education',y='Income',hue="Product",data=cdata)
sns.countplot(x="MaritalStatus", hue = "Product", data=cdata)
#Fewer female customers opt for TM798 model
sns.countplot(x="Product", hue = "Gender", data=cdata, color="red")
# Higher Fitness rating customers prefer TM798 Model
sns.countplot(x="Fitness",hue = "Product", data=cdata, color = "blue")
# Higher the fitness level greater the miles targetted ie usage also is higher.
# As seen from above higher fitness customer prefer TM798 Model
sns.swarmplot(x='Fitness', y='Miles', data=cdata, hue = "Product")
sns.lmplot(x='Income', y = 'Miles', data = cdata, scatter_kws ={'s':20})
sns.stripplot(x='Fitness', y="Age", data = cdata, hue = "Product")
sns.lmplot(x='Age', y = 'Fitness', data = cdata, col= "Product", aspect = 0.6, height = 5, hue = "Gender", palette="magma")
