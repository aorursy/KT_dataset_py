# First of all let's import basic libraries and data to get started

import numpy as np

from numpy  import array

import pandas as pd

import matplotlib.pyplot as plt



iris = pd.read_csv("../input/Iris.csv") # load data

iris.drop('Id',axis=1,inplace=True) # Id column is redundant as df already assigns index
# DATA EXPLORATION...



# Before starting pre-processing, basic descriptive statistics can help identify if 

# scaling/normalization of data is needed, or how variance varies between 

# features which is useful information for feature selection or at least understanding which 

# features might be most important for classification



# dataframe.info gives basic information on data integrity (data types and detection of NaN values)

iris.info()



# Next I'm using dataframe.describe function. Works for both object and numeric

iris.describe()



# Means are in the same order of magnitude for all features so scaling might not be beneficial.

# If mean values were of different orders of magnitude, scaling could significantly 

# improve accuracy of a classifier.
# Although not essential for this dataset let's see if scaling provides additional insights.



# First we need to split X, Y into separate sets to feed X to the scaler.

X = iris.drop('Species',1)

Y = iris.Species



# Scaling of X

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print('X_scaled type is',type(X_scaled))



# As output of scaler is np array I'll transform back to df for easier exploration and plotting

X_scaled_df = pd.DataFrame(X_scaled,columns=['s_SepalLength','s_SepalWidth',

                                             's_PetalLength','s_PetalWidth'])

df = pd.concat([X_scaled_df,Y],axis=1)



# Notice x-axis on subplots are all the same for all features (0 to 1) after scaling.

fig = plt.figure(figsize=(12,7))

fig.suptitle('Frequency Distribution of Features by Species ',fontsize=20)



ax1 = fig.add_subplot(221)

df.groupby("Species").s_PetalLength.plot(kind='hist',alpha=0.8,legend=True,title='s_PetalLength')

ax2 = fig.add_subplot(222,sharey=ax1)

df.groupby("Species").s_PetalWidth.plot(kind='hist',alpha=0.8,legend=True,title='s_PetalWidth')

ax3 = fig.add_subplot(223,sharey=ax1)

df.groupby("Species").s_SepalLength.plot(kind='hist',alpha=0.8,legend=True,title='s_SepalLength')

ax4 = fig.add_subplot(224,sharey=ax1)

df.groupby("Species").s_SepalWidth.plot(kind='hist',alpha=0.8,legend=True,title='s_SepalWidth')



plt.tight_layout(pad=4, w_pad=1, h_pad=1.5)

plt.show()

X_scaled_df.describe()
# Another useful visualization I like to use during exploration is a 2D PCA plot.

# It can quickly indicate how easy or difficult the classification problem is.

# This is particularly relevant for high-dimensional datasets.



from sklearn.decomposition import PCA

# spliting input and target

X = iris.drop('Species',1)

Y = iris.Species



pca = PCA(n_components=2).fit_transform(X) # pca output is an array

pca_df = pd.DataFrame(pca,columns=['pca_1','pca_2']) # transforming back to df

pca_Y = pd.concat([pca_df, Y],axis=1)



# The 3 species cluster nicely which is a good indication a classifier can be trained 

# at high accuracy. 

# It is also expected that accuracy of identifying setosa will be higher.



import matplotlib.pyplot as plt

import seaborn as sns

sns.FacetGrid(pca_Y, hue="Species", palette="Set1", size=6).map(plt.scatter, "pca_1", "pca_2").add_legend()

plt.show()
# Let's now build a classifier and evaluate accuracy

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 0)

print("train sample size",x_train.shape,type(x_train))

print("test sample size",x_test.shape,type(x_test))
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support



clf = SVC(kernel = 'linear').fit(x_train,y_train)

clf.predict(x_train)

y_pred = clf.predict(x_test)



# Creates a confusion matrix

cm = confusion_matrix(y_test, y_pred) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['setosa','versicolor','virginica'], 

                     columns = ['setosa','versicolor','virginica'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()


