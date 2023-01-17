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
dataset = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
dataset.head()
dataset.info()
dataset = dataset.drop(["id"], axis = 1)
dataset = dataset.drop(["Unnamed: 32"], axis = 1)
dataset.head(3)
M = dataset[dataset.diagnosis == "M"]
M.head(5)
B = dataset[dataset.diagnosis == "B"]
B.head(5)
plt.title("Malignant vs Benign Tumor")

plt.xlabel("Radius Mean")

plt.ylabel("Texture Mean")

plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)

plt.scatter(B.radius_mean, B.texture_mean, color = "lime", label = "Benign", alpha = 0.3)

plt.legend()

plt.show()
dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]
x = dataset.drop(["diagnosis"], axis = 1)

y = dataset.diagnosis.values
# Normalization:

x = (x - np.min(x)) / (np.max(x) - np.min(x))
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
# prediction

dt.score(x_test, y_test)