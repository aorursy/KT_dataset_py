import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import category_encoders as ce #encoding
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #dim red
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 



%matplotlib inline

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
#import seaborn as sns
#import matplotlib.pyplot as plt
sb.set(style="white", color_codes=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
iris = pd.read_csv("../input/iris/Iris.csv") #load the dataset
iris.head()
iris.shape #Shape of dataset
iris.columns # Cloumns present in dataset
# Let's see how many examples we have of each species
iris["Species"].value_counts()
iris.describe() # Describe statistics of the dataset
iris.info() # Informantion of dataset
iris.isnull().sum() #checking for missng values
x1=iris.SepalLengthCm
y1=iris.SepalWidthCm
x2=iris.PetalLengthCm
y2=iris.PetalWidthCm
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
plt.subplot(1,2,1)
plt.scatter(x1,y1) 
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.subplot(1,2,2)
plt.scatter(x2, y2) # We'll plot a scatterplot of the Iris features using .plot
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.tight_layout(pad=5)
# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sb.jointplot(x1, y1, data=iris, size=5)
sb.jointplot(x2, y2, data=iris, size=5)
# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sb.FacetGrid(iris, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
sb.FacetGrid(iris, hue="Species", size=5) \
   .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()
# We can look at an individual feature in Seaborn through a boxplot
sb.boxplot(x="Species", y="PetalLengthCm", data=iris)
# One way we can extend this plot is adding a layer of individual points on top of it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown on top of the previous axes
ax = sb.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sb.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sb.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sb.FacetGrid(iris, hue="Species", size=6) \
   .map(sb.kdeplot, "PetalLengthCm") \
   .add_legend()
# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sb.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
sb.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")
# Another multivariate visualization technique pandas has is parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")
# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")
X = iris.drop(['Species'],axis=1)
y = iris.Species
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=20, stratify=y)
knn = KNeighborsClassifier(7)
knn.fit(X_train,y_train)
print("Train score before PCA",knn.score(X_train,y_train))
print("Test score before PCA",knn.score(X_test,y_test))
from sklearn.decomposition import PCA
pca = PCA()
X_new = pca.fit_transform(X)
pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
explained_variance
pca=PCA(n_components=3)
X_new=pca.fit_transform(X)
X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state=20, stratify=y)
knn_pca = KNeighborsClassifier(7)
knn_pca.fit(X_train_new,y_train)
print("Train score after PCA",knn_pca.score(X_train_new,y_train))
print("Test score after PCA",knn_pca.score(X_test_new,y_test))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)
#set ids as PassengerId and predict survival 
y_test=pd.DataFrame(y_test)
ids = y_test.index
ids
#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'Id' : ids, 'Species': y_pred })
output.to_csv('submission.csv', index=False)