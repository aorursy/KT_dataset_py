# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
print(plt.style.available)
plt.style.use('ggplot')
data.head()
data.info()
data.describe()
def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count    
    """
    # get feature
    var = data[variable]
    # caount number of categorical variable (value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
category = ["sex", "cp", "restecg", "exang", "slope", "ca", "thal", "target"]
for c in category:
    bar_plot(c)

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(data[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["age", "trestbps", "chol", "fbs", "thalach", "oldpeak"]
for n in numericVar:
    plot_hist(n)
# sex - target
data[["sex", "target"]].groupby(["sex"], as_index = False).mean().sort_values(by = "target", ascending =False)
# cp - target
data[["cp", "target"]].groupby(["cp"], as_index = False).mean().sort_values(by = "target", ascending =False)
# restecg - target
data[["restecg", "target"]].groupby(["restecg"], as_index = False).mean().sort_values(by = "target", ascending =False)
# exang - target
data[["exang", "target"]].groupby(["exang"], as_index = False).mean().sort_values(by = "target", ascending =False)
# slope - target
data[["slope", "target"]].groupby(["slope"], as_index = False).mean().sort_values(by = "target", ascending =False)
# ca - target
data[["ca", "target"]].groupby(["ca"], as_index = False).mean().sort_values(by = "target", ascending =False)
# thal - target
data[["thal", "target"]].groupby(["thal"], as_index = False).mean().sort_values(by = "target", ascending =False)
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        #Detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
data.loc[detect_outliers(data,["age", "trestbps", "chol", "fbs", "thalach", "oldpeak"])]
data.columns[data.isnull().any()]
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(data[["age", "trestbps", "chol", "fbs", "thalach", "oldpeak",
                      "sex", "cp", "restecg", "exang", "slope", "ca", "thal", "target"]].corr(), annot = True)
plt.show()
g = sns.factorplot(x = "thal", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()
g = sns.factorplot(x = "ca", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()
g = sns.factorplot(x = "slope", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()
g = sns.factorplot(x = "exang", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()
g = sns.factorplot(x = "cp", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()
g = sns.FacetGrid(data, col = "target", size = 6)
g.map(sns.distplot, "oldpeak", bins = 25)
plt.show()
g = sns.FacetGrid(data, col = "target")
g.map(sns.distplot, "thalach", bins = 25)
plt.show()
g = sns.FacetGrid(data, col = "target", row = "slope", size = 3)
g.map(plt.hist, "oldpeak", bins = 25)
g.add_legend()
plt.show()
g = sns.FacetGrid(data, col = "target", row = "slope", size = 3)
g.map(plt.hist, "thalach", bins = 25)
g.add_legend()
plt.show()
g = sns.FacetGrid(data, col = "target", row = "exang", size = 4)
g.map(plt.hist, "cp", bins = 25)
g.add_legend()
plt.show()
g = sns.FacetGrid(data, col = "target", row = "exang", size = 4)
g.map(plt.hist, "thalach", bins = 25)
g.add_legend()
plt.show()
g = sns.FacetGrid(data, col = "target", row = "cp", size = 2)
g.map(plt.hist, "thalach", bins = 25)
g.add_legend()
plt.show()
g = sns.FacetGrid(data, col="target", size = 8)
g.map(plt.scatter, "oldpeak", "thalach", edgecolor="w")
g.add_legend()
plt.show()
g = sns.FacetGrid(data, col="target", size = 8)
g.map(sns.kdeplot, "age", "thalach", edgecolor="w")
g.add_legend()
plt.show()