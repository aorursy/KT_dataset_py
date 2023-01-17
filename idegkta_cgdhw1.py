# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataframe = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

dataframe.head(10)
import matplotlib.pyplot as plt



dataframe.plot(kind="scatter", x ="math score", y="reading score", alpha=0.5,color="g")

plt.xlabel("Math Score")

plt.ylabel("Reading Score")

plt.title("Corelation")

plt.show()
import matplotlib.pyplot as plt



writing_sc=dataframe["writing score"]

writing_sc.plot(kind="hist",bins=40, figsize=(15,15))

plt.title("Histogram")

plt.show()
import numpy as np



dataframe[np.logical_and(dataframe["gender"]=="male",dataframe["math score"]>=90)]
import pandas as pd



for index, y in dataframe[["writing score"]][9:20].iterrows():

    print(index+1,":", y)