# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm
from pandas.plotting import parallel_coordinates
%matplotlib inline
data= pd.read_csv("../input/Iris.csv")
data.columns
data.dtypes
data.set_index('Id', inplace=True)
data.head(10)
data.corr()
sns.clustermap(data.corr(), method = 'single', cmap = 'coolwarm')
sns.countplot(x='Species',data=data)
sns.violinplot(x="Species", y="PetalLengthCm", data=data, palette='rainbow')
sns.boxplot(x = 'Species', y = 'PetalWidthCm', data =data, palette='rainbow')
sns.swarmplot(x = 'Species', y = 'SepalLengthCm', data = data, palette = 'rainbow')
sns.stripplot(x="Species", y="SepalWidthCm", data=data)
sns.pairplot(data, hue="Species", palette="husl")
sns.pairplot(data,
             x_vars=["SepalWidthCm", "SepalLengthCm"],
             y_vars=["PetalWidthCm", "PetalLengthCm"], hue = 'Species')
parallel_coordinates(data, 'Species')
from sklearn.model_selection import train_test_split
X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = data.Species
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.3)
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn. linear_model import LogisticRegression
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(X_train,y_train)
model.score(X_test,y_test)
