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
import seaborn as sns

import matplotlib.pyplot as plt

#import libraries
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

#import dataset
#data-points and features?

df.shape
df.head()

#topmost values  of the dataset
df.columns

#column names in our dataset?
df = df.drop('Id', axis=1)

#drop id column
df['Species'].value_counts()

#balanced dataset as the number of data points for every class is 50.
df.describe()

#gives info about the min, max standard deviation, etc
df.info()

#info about dataset no missing values
sns.set_style("darkgrid");

sns.FacetGrid(df, hue='Species' ,size=6).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm').add_legend();

plt.show();

#iris-setosa can be easily seperated

#but iris-versicolor, iris-virginica both overlapping each other
import plotly.express as px

fig = px.scatter_3d(df, 'SepalLengthCm', y='SepalWidthCm', z='PetalWidthCm',

                    color='PetalLengthCm', symbol='Species')

fig.show()
# pairwise scatter plot

plt.close();

sns.set_style("darkgrid");

sns.pairplot(df, hue='Species',height=4);

plt.show()

#shows distribution of features

#petallengthcm and petalwidthcm 
df.Species.value_counts()

#3 type of flower species total=150

#and each have 50 observations
import numpy as np

iris_setosa = df.loc[df['Species'] == 'Iris-setosa'];

iris_virginica = df.loc[df['Species'] == 'Iris-virginica'];

iris_versicolor = df.loc[df['Species'] == 'Iris-versicolor'];



plt.plot(iris_setosa['PetalLengthCm'])

plt.plot(iris_versicolor['PetalLengthCm'])

plt.plot(iris_virginica['PetalLengthCm'])

plt.show()
sns.FacetGrid(df, hue='Species', height=5).map(sns.distplot, 'PetalLengthCm').add_legend();

plt.show();

#flowers of petallengthcm having 1-2cm are iris setosa 

#flowers petallengthcm having more(2.3 to 5.6 for versicolor, 4 to 7.9 virginica) then that are or iris-virginica and iris-versicolor
sns.FacetGrid(df, hue='Species', height=5).map(sns.distplot, 'PetalWidthCm').add_legend();

plt.show();

#not much of spread, setosa is peaked
sns.FacetGrid(df, hue='Species', size=5).map(sns.distplot, 'SepalLengthCm').add_legend();

plt.show();

#all overlapped 
sns.FacetGrid(df, hue='Species', size=5).map(sns.distplot, 'SepalWidthCm').add_legend();

plt.show();
counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf)

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf);

plt.plot(bin_edges[1:], cdf);



counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'], bins=20, density=True)

pdf = counts/(sum(counts))

plt.plot(bin_edges[1:], pdf);

plt.show();
#Cumulative Distribution Function (CDF)

counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:] , pdf)

plt.plot(bin_edges[1:], cdf)



plt.show();
counts, bin_edges = np.histogram(iris_versicolor['PetalLengthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:] , pdf)

plt.plot(bin_edges[1:], cdf)



plt.show();
counts, bin_edges = np.histogram(iris_versicolor['PetalWidthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:] , pdf)

plt.plot(bin_edges[1:], cdf)



plt.show();
counts, bin_edges = np.histogram(iris_virginica['PetalWidthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)



cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:] , pdf)

plt.plot(bin_edges[1:], cdf)



plt.show();
counts, bin_edges = np.histogram(iris_setosa['PetalLengthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf)

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)



#virginica 

counts, bin_edges = np.histogram(iris_virginica['PetalLengthCm'], bins=10, density=True)

pdf = counts/(sum(counts))

print(pdf)

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)



#versicolor

counts, bin_edges = np.histogram(iris_versicolor['PetalLengthCm'], bins=10, 

                                 density = True)



pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)





plt.show();

#pdf and cdf of all three flower species
print('means')

print(np.mean(iris_setosa['PetalLengthCm']))

print(np.mean(iris_virginica['PetalLengthCm']))

print(np.mean(iris_versicolor['PetalLengthCm']))



print('\nstd-dev')

print(np.std(iris_setosa['PetalLengthCm']))

print(np.std(iris_virginica['PetalLengthCm']))

print(np.std(iris_versicolor['PetalLengthCm']))


print("\nMedians:")

print(np.median(iris_setosa["PetalLengthCm"]))

print(np.median(iris_virginica["PetalLengthCm"]))

print(np.median(iris_versicolor["PetalLengthCm"]))



print("\nQuantiles:")

print(np.percentile(iris_setosa["PetalLengthCm"],np.arange(0, 100, 25)))

print(np.percentile(iris_virginica["PetalLengthCm"],np.arange(0, 100, 25)))

print(np.percentile(iris_versicolor["PetalLengthCm"], np.arange(0, 100, 25)))



print("\n90th Percentiles:")

print(np.percentile(iris_setosa["PetalLengthCm"],90))

print(np.percentile(iris_virginica["PetalLengthCm"],90))

print(np.percentile(iris_versicolor["PetalLengthCm"], 90))



from statsmodels import robust

print ("\nMedian Absolute Deviation")

print(robust.mad(iris_setosa["PetalLengthCm"]))

print(robust.mad(iris_virginica["PetalLengthCm"]))

print(robust.mad(iris_versicolor["PetalLengthCm"]))
# boxplot is a standardized way of displaying the distribution of data based

sns.boxplot(x='Species',y='PetalLengthCm', data=df)

plt.show()

#tells min, max ,25 50 75 
sns.boxplot(x='Species',y='PetalWidthCm', data=df)

plt.show()

sns.boxplot(x='Species',y='SepalWidthCm', data=df)

plt.show()
sns.boxplot(x='Species',y='SepalLengthCm', data=df)

plt.show()
sns.violinplot(x="Species", y="PetalLengthCm", data=df, size=10)

plt.show()
sns.violinplot(x="Species", y="PetalWidthCm", data=df, size=10)

plt.show()
#2D Density plot, contors-plot

sns.jointplot(x="PetalLengthCm", y="PetalWidthCm", data=iris_setosa, kind="kde");

plt.show();
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris_setosa, kind="kde");

plt.show();
df.hist(edgecolor='black', linewidth=1.2)

fig = plt.gcf()

fig.set_size_inches(12, 6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2, 2,1)

sns.violinplot(x='Species', y='PetalLengthCm' , data=df)

plt.subplot(2, 2, 2)

sns.violinplot(x='Species', y='PetalWidthCm', data=df)

plt.subplot(2, 2, 3)

sns.violinplot(x='Species', y='SepalLengthCm', data=df)

plt.subplot(2, 2, 4)

sns.violinplot(x='Species', y='SepalWidthCm', data=df)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
df.shape
plt.figure(figsize=(7,4))

sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')

plt.show()
train, test = train_test_split(df, test_size=0.3)

print(train.shape)

print(test.shape)
train_x = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y = train.Species

test_x = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_y = test.Species
train_x.head()
train_y.head()
model = svm.SVC() #select the algorithm

model.fit(train_x,train_y) # we train the algorithm with the training data and the training output

prediction=model.predict(test_x) #now we pass the testing data to the trained algorithm

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))#now we check the accuracy of the algorithm. 

#we pass the predicted output by the model and the actual output