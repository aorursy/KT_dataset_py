# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.columns
df.head()
df.describe()
df.info()
def bar_plot(variable):

    

    

    

    var = df[variable]

    var_value = var.value_counts()

    

    #visualize

    

    plt.figure(figsize = (10,3))

    plt.bar(var_value.index, var_value)

    plt.xticks(var_value.index)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    

    print("{} \n {}".format(variable,var_value))
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal','target']

for i in categorical_cols:

    bar_plot(i)
def plot_hist(variable):

    """

    age, trestbps, chol, thalach, oldpeak

    

    """

    

    var = df[variable]

    

    #visualize

    plt.figure(figsize = (10,3))

    plt.hist(var,bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with histogram".format(variable))

    plt.show()
numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for i in numerical:

    plot_hist(i)
df[["sex","target"]].groupby(["sex"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["cp","target"]].groupby(["cp"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["fbs","target"]].groupby(["fbs"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["restecg","target"]].groupby(["restecg"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["exang","target"]].groupby(["exang"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["slope","target"]].groupby(["slope"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["ca","target"]].groupby(["ca"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["thal","target"]].groupby(["thal"], as_index = False).mean().sort_values(by = 'target', ascending = False)
df[["sex","target","cp"]].groupby(["sex","cp"], as_index = False).mean().sort_values(by = 'target', ascending = False)
def detect_outliers(data,features):

    outlier_indices = []

    for i in features:

        #1st quartile

        Q1 = np.percentile(data[i],25)

        #3rd quartile

        Q3 = np.percentile(data[i],75)

        #IQR

        IQR = Q3 - Q1

        #Outlier step

        outlier_step = IQR * 1.5

        #detect outlier and their indices

        outlier_list_cols = data[(data[i] <  Q1 - outlier_step) | (data[i] >  Q3 + outlier_step)].index

        

        #store indices

        

        outlier_indices.extend(outlier_list_cols)

        

    

    outlier_indices = Counter(outlier_indices)

    

    multiple_outliers = list(c for c,k in outlier_indices.items() if k>2)

    

    return multiple_outliers

        

        
df.loc[detect_outliers(df,["age","trestbps","chol","thalach","oldpeak"])]
df.columns[df.isnull().any()]
plt.subplots(figsize = (15,15))

sns.heatmap(df.corr(), annot=True, fmt='.2f')

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='cp',y='target',data=df)

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='slope',y='target',data=df)

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.boxplot(data=df,x='target',y='thalach')

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='sex',y='target',data=df)

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='exang',y='target',data=df)

plt.show()
ax = sns.FacetGrid(df, col="target", size=5)

ax.map(sns.distplot, 'oldpeak',bins=10)

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='ca',y='target',data=df)

plt.show()
f,ax = plt.subplots(figsize = (10,10))

ax = sns.barplot(x='thal',y='target',data=df)

plt.show()