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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from mlxtend.plotting import plot_decision_regions

sns.set_style("darkgrid")
df = pd.read_csv("/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv")

df.head()
df.isnull().sum()
def histplot(feature):

    plt.figure(figsize=(16,8))

    sns.set(font_scale=1.2)

    sns.distplot(df[feature],color='blue')

    plt.show()

    
histplot("inflation")
histplot("total_fer")
histplot("gdpp")
df.columns
histplot("income")
histplot("imports")
histplot("exports")
histplot("health")
histplot("child_mort")
df.columns
import plotly.express as px
def groupbyplot(feature):

    group = df.groupby("country")[feature].sum().sort_values(ascending=False).reset_index().head(10)

    fig = plt.figure(figsize=(16,8))

    fig = px.bar(group, x='country',y=feature,color=feature)

    fig.show()

    

    
groupbyplot("gdpp")
groupbyplot("income")
groupbyplot("life_expec")
groupbyplot("inflation")
groupbyplot("imports")
groupbyplot("exports")
groupbyplot("health")
groupbyplot("child_mort")
df.columns
groupbyplot("total_fer")
def scatter(x,y,size="gdpp",color="country"):

    fig = px.scatter(df,x=x,y=y,size=size,color=color)

    fig.show()

    
scatter("income","gdpp")
scatter("imports","exports",size="income")
df.columns
scatter("imports","income")
scatter("inflation","gdpp",size="health")
scatter("health","child_mort",size="child_mort")
scatter("health","life_expec",size="income")
scatter("total_fer","health")
df.columns
facet = ['child_mort', 'exports', 'health', 'imports', 'income',

       'inflation', 'life_expec', 'total_fer', 'gdpp']

facetdata = df[facet]

facetdata
sns.pairplot(facetdata)