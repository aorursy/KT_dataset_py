# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#read the dataset
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()
#finding different types of species
df['Species'].unique()
# use counplot from seaborn library to count the observations in Species
sns.countplot(x="Species", data = df)
#distribution with respect to SepalLenghtCm
sns.FacetGrid(df, hue="Species", size = 4).map(sns.kdeplot, "SepalLengthCm").add_legend()
#distribution with respect to SepalWidthCm
sns.FacetGrid(df, hue="Species", size = 4).map(sns.kdeplot, "SepalWidthCm").add_legend()
#distribution with respect to PetalLengthCm
sns.FacetGrid(df, hue="Species", size = 4).map(sns.kdeplot, "PetalLengthCm").add_legend()
#distribution with respect to PetalWidthCm
sns.FacetGrid(df, hue="Species", size = 4).map(sns.kdeplot, "PetalWidthCm").add_legend()
# Sepal Length vs Sepal Width
sns.relplot(x="SepalLengthCm", y="SepalWidthCm", data=df, hue='Species')
# Sepal Length vs Petal Width
sns.relplot(x="SepalLengthCm", y="PetalWidthCm", data=df, hue='Species')
# Sepal Length vs Petal Length
sns.relplot(x="SepalLengthCm", y="PetalLengthCm", data=df, hue='Species')
# Sepal Width vs Petal Width
sns.relplot(x="SepalWidthCm", y="PetalWidthCm", data=df, hue='Species')
# Sepal Width vs Petal Length
sns.relplot(x="SepalWidthCm", y="PetalLengthCm", data=df, hue='Species')
# Petal Length vs Petal Width 
sns.relplot(x="PetalLengthCm", y="PetalWidthCm", data=df, hue='Species')
# Sepal Length using a strip plot
sns.catplot(y ='SepalLengthCm', x = 'Species', data =df, kind = 'strip')
# Sepal Width using swarm plot
# A Swarm Plot is similar to a strip plot but it also prevents the data points
# from overlapping by adjsuting them along the categorical axis. This gives a 
# better representation of the distribution of observations,
# although it only works well for relatively small dataset such as Iris.
sns.catplot(y ='SepalWidthCm', x = 'Species', data =df,  kind = 'swarm')
# Petal Length using Box plot
# A Box Plot is widely used as it not only provide relationship between the two features
# but also shows the three quartile values of the distribution along with extreme values.
sns.catplot(y ='PetalLengthCm', x = 'Species', data =df, kind = 'box')
# Petal Length using Violin Plot
# A violin plot basically combines the features of a kde plot and a box plot.
sns.catplot(y ='PetalWidthCm', x = 'Species', data =df, kind = 'violin')
sns.pairplot(df.drop('Id',axis=1), hue='Species')
cols = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
rows = ['Iris-setosa', 'Iris-versicolor','Iris-virginica']

data = [['Low','High','Low','Low'],['Medium','Medium','Medium','Medium'],['High','Medium','High','High']]

res = pd.DataFrame(data, index = rows, columns = cols)
res