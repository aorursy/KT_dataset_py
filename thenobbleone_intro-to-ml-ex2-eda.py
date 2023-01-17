# This notebook is to accompany pinata data's intro to course on ml on skillshare. 
# if you want to find out more please go to pinatadata.com

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris_data = pd._______('/kaggle/input/iris/Iris.csv')
iris_info = iris_data.____()

iris_describe = iris_data.________()

iris_head = iris_data.____()

print(iris_info)

print(iris_describe)

print(iris_head)
# Print the descriptive statistics for each of the different species. 
# What do you notice about the mean Petal length across the 3 species?
for t in iris_data['_______'].unique():
    print(t)
    print(iris_data[iris_data['_______'] == t].________())
sns._______(x='Species', 
            y='PetalLengthCm',
            data=iris_data
            )
sns.__________(x='Species', 
               y='PetalWidthCm',
               data=iris_data
                )
# remove id as it tells us nothing but is in the plot
sns.________(data = iris_data.drop('Id',axis =1), 
             hue='Species')
