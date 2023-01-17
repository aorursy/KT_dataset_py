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
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df
X = df

y = df.target
y
X.drop("target",axis = 1, inplace = True)
X
X.corr()
X.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.title("Age with Cholestrol level")

sns.lineplot(x = df['age'], y = df['chol'],data = X)
X.columns
plt.title("Age with resting blood pressure level")

sns.lineplot(x = df['age'], y = df['trestbps'],data = X)
plt.title("Age with Constrictive pericarditis i.e. cp")

sns.lineplot(x = df['age'], y = df['cp'],data = X)
plt.title("Age with Fasting blood sugar i.e. cp")

sns.lineplot(x = df['age'], y = df['fbs'],data = X)
plt.title("Age with resting electrocardiograph i.e. cp")

sns.lineplot(x = df['age'], y = df['restecg'],data = X)
plt.title("Age with Constrictive pericarditis i.e. cp")

sns.lineplot(x = df['age'], y = df['thalach'],data = X)
plt.title("Cholestrol level vs Maximum heart rate of an individual")

sns.lineplot(x = df['chol'], y = df['thalach'],data = X)
plt.title("Age vs exang")

sns.lineplot(x = df['age'], y = df['exang'],data = X)
plt.title("Age vs oldpeak")

sns.lineplot(x = df['age'], y = df['oldpeak'],data = X)
X.columns
plt.title("Age vs exang")

sns.lineplot(x = df['age'], y = df['thal'],data = X)
sns.heatmap(data = X,annot = True)