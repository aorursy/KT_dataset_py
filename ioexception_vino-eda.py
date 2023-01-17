import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from contextlib import contextmanager
from sklearn.model_selection import train_test_split

# Set style for plots
style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# https://www.sciencedirect.com/science/article/pii/S0167923609001377
# http://www3.dsi.uminho.pt/pcortez/wine5.pdf
wine_quality = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
print("Dataset length", len(wine_quality))
wine_quality.head()
wine_quality.columns
wine_train, wine_test = train_test_split(wine_quality)
wine_train.info()
wine_test.info()
wine_train.describe()
@contextmanager
def plot(title=None, xlabel=None, ylabel=None, figsize=(9,5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    yield ax
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel, size=15)
    ax.set_ylabel(ylabel, size=15)
    
with plot(title="Counts of `qualuty`", xlabel="Quality", ylabel="Count") as ax:
    sns.countplot(x="quality", palette=("Accent"), data=wine_train, ax=ax)
with plot(title="Fixed Acidity distribution", xlabel="Acidity") as ax:
    sns.distplot(wine_train["volatile acidity"], ax=ax)
variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']

columns = 4

fig, axes = plt.subplots(len(variables) //columns, columns, figsize=(15,8))

for current_idx, variable in enumerate(variables):
    i = current_idx // columns
    j = current_idx % columns
    sns.distplot(wine_train[variable], ax=axes[i][j])
    axes[i][j].set_title(variable)
    axes[i][j].set_xlabel("")
    
plt.tight_layout()
variables = ['fixed acidity', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 
             'total sulfur dioxide','sulphates', 'alcohol']

fig, axes = plt.subplots(1, len(variables), figsize=(15,6))

for ax, variable in zip(axes, variables):
    ax = sns.boxplot( y=variable, data=wine_train, ax=ax)
plt.tight_layout()
with plot(title="Alcohol - Quality scatterplot", xlabel="Alcohol", ylabel="Quality") as ax:
    sns.scatterplot(x="alcohol", y="quality", data=wine_train, ax=ax) 
with plot(title="Free Sulfur Dioxide - Total Sulfur Dioxide scatterplot", xlabel="Free Sulfur Dioxide", ylabel="Total Sulfur Dioxide") as ax:
    sns.scatterplot(x="free sulfur dioxide", y="total sulfur dioxide", data=wine_train, ax=ax) 
correlation = wine_train.corr(method="pearson")
correlation.head()
with plot(title="Free Sulfur Dioxide - Total Sulfur Dioxide scatterplot", xlabel="Free Sulfur Dioxide", ylabel="Total Sulfur Dioxide") as ax:
    sns.heatmap(correlation, vmin=-1,cmap= 'coolwarm',  ax=ax) 
