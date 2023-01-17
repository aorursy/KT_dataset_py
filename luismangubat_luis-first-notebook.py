# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the data and displaying some rows
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")

# df aka dataframe, is the the dataset can be manipulated via the panda library.
display(data_train.head(25))
# Check typical statistic of the model such as its attributes, average, min, max and etc. 
data_train.describe()

# My first intuition is to find patterns in data that could be seen useful
# I will use seasborn graphing api to see possible patterns in the dataset
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);


sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "red"},
              markers=["*", "o"], linestyles=["-", "--"]);


print(data_train.info())
women = data_train.loc[data_train.Sex == 'female']["Survived"]

rate = sum(women) / len(women)
print("Percentage of women who survived:", 100 * rate)


men = data_train.loc[data_train.Sex == 'male']["Survived"]

rate = sum(men) / len(men)
print("Percentage of men who survived:", 100 * rate)
g = sns.FacetGrid(data_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# Importing the RandomForestClassifier from Scitkit learn
from sklearn.ensemble import RandomForestClassifier

# Our Y value is the predictor column in the dataset. 
# In this case Survive, where 0 means death, and 1 means survived

y = data_train["Survived"]
