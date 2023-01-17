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

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

print("Libraries Imported")
file_path = "../input/iris/Iris.csv"



iris_data = pd.read_csv(file_path, index_col="Id")



print("Data Imported")
iris_data.head()
iris_data.shape
iris_data.info()
iris_data.describe()
iris_data.Species.unique()
data_setosa = iris_data.loc[iris_data.Species == "Iris-setosa"]

data_versicolor = iris_data.loc[iris_data.Species == "Iris-versicolor"]

data_virginica = iris_data.loc[iris_data.Species == "Iris-virginica"]
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 3, figsize=(15, 5), sharex=True)

sns.despine(left=True)



#Plot the boxplot

sns.boxplot(x = iris_data['SepalLengthCm'], ax=axes[0])



# Plot the histogram

sns.distplot(a=iris_data['SepalLengthCm'], kde=False, ax=axes[1])



# Plot the density plot

sns.kdeplot(data=iris_data['SepalLengthCm'], shade=True, ax=axes[2])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 3, figsize=(15, 5), sharex=True)

sns.despine(left=True)



#Plot the boxplot

sns.boxplot(x = iris_data['SepalWidthCm'], ax=axes[0])



# Plot the histogram

sns.distplot(a=iris_data['SepalWidthCm'], kde=False, ax=axes[1])



# Plot the density plot

sns.kdeplot(data=iris_data['SepalWidthCm'], shade=True, ax=axes[2])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 3, figsize=(15, 5), sharex=True)

sns.despine(left=True)



#Plot the boxplot

sns.boxplot(x = iris_data['PetalLengthCm'], ax=axes[0])



# Plot the histogram

sns.distplot(a=iris_data['PetalLengthCm'], kde=False, ax=axes[1])



# Plot the density plot

sns.kdeplot(data=iris_data['PetalLengthCm'], shade=True, ax=axes[2])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 3, figsize=(15, 5), sharex=True)

sns.despine(left=True)



#Plot the boxplot

sns.boxplot(x = iris_data['PetalWidthCm'], ax=axes[0])



# Plot the histogram

sns.distplot(a=iris_data['PetalWidthCm'], kde=False, ax=axes[1])



# Plot the density plot

sns.kdeplot(data=iris_data['PetalWidthCm'], shade=True, ax=axes[2])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)





# Histograms for each species

sns.distplot(a=data_setosa['SepalLengthCm'], label="Iris-setosa", kde=False, ax=axes[0])

sns.distplot(a=data_versicolor['SepalLengthCm'], label="Iris-versicolor", kde=False, ax=axes[0])

sns.distplot(a=data_virginica['SepalLengthCm'], label="Iris-virginica", kde=False, ax=axes[0])



# KDE plots for each species

sns.kdeplot(data=data_setosa['SepalLengthCm'], label="Iris-setosa", shade=True, ax=axes[1])

sns.kdeplot(data_versicolor['SepalLengthCm'], label="Iris-versicolor", shade=True, ax=axes[1])

sns.kdeplot(data=data_virginica['SepalLengthCm'], label="Iris-virginica", shade=True, ax=axes[1])





plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Plot the Swarmplot

sns.swarmplot(x=iris_data['Species'], y=iris_data['SepalLengthCm'], ax=axes[0])



# Plot the Boxplot

sns.boxplot(x=iris_data['Species'], y=iris_data['SepalLengthCm'], ax=axes[1])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Histograms for each species

sns.distplot(a=data_setosa['SepalWidthCm'], label="Iris-setosa", kde=False, ax=axes[0])

sns.distplot(a=data_versicolor['SepalWidthCm'], label="Iris-versicolor", kde=False, ax=axes[0])

sns.distplot(a=data_virginica['SepalWidthCm'], label="Iris-virginica", kde=False, ax=axes[0])



# KDE plots for each species

sns.kdeplot(data=data_setosa['SepalWidthCm'], label="Iris-setosa", shade=True, ax=axes[1])

sns.kdeplot(data_versicolor['SepalWidthCm'], label="Iris-versicolor", shade=True, ax=axes[1])

sns.kdeplot(data=data_virginica['SepalWidthCm'], label="Iris-virginica", shade=True, ax=axes[1])





plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Plot the Swarmplot

sns.swarmplot(x=iris_data['Species'], y=iris_data['SepalWidthCm'], ax=axes[0])



# Plot the Boxplot

sns.boxplot(x=iris_data['Species'], y=iris_data['SepalWidthCm'], ax=axes[1])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Histograms for each species

sns.distplot(a=data_setosa['PetalLengthCm'], label="Iris-setosa", kde=False, ax=axes[0])

sns.distplot(a=data_versicolor['PetalLengthCm'], label="Iris-versicolor", kde=False, ax=axes[0])

sns.distplot(a=data_virginica['PetalLengthCm'], label="Iris-virginica", kde=False, ax=axes[0])



# KDE plots for each species

sns.kdeplot(data=data_setosa['PetalLengthCm'], label="Iris-setosa", shade=True, ax=axes[1])

sns.kdeplot(data_versicolor['PetalLengthCm'], label="Iris-versicolor", shade=True, ax=axes[1])

sns.kdeplot(data=data_virginica['PetalLengthCm'], label="Iris-virginica", shade=True, ax=axes[1])





plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Plot the Swarmplot

sns.swarmplot(x=iris_data['Species'], y=iris_data['PetalLengthCm'], ax=axes[0])



# Plot the Boxplot

sns.boxplot(x=iris_data['Species'], y=iris_data['PetalLengthCm'], ax=axes[1])



plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Histograms for each species

sns.distplot(a=data_setosa['PetalWidthCm'], label="Iris-setosa", kde=False, ax=axes[0])

sns.distplot(a=data_versicolor['PetalWidthCm'], label="Iris-versicolor", kde=False, ax=axes[0])

sns.distplot(a=data_virginica['PetalWidthCm'], label="Iris-virginica", kde=False, ax=axes[0])



# KDE plots for each species

sns.kdeplot(data=data_setosa['PetalWidthCm'], label="Iris-setosa", shade=True, ax=axes[1])

sns.kdeplot(data_versicolor['PetalWidthCm'], label="Iris-versicolor", shade=True, ax=axes[1])

sns.kdeplot(data=data_virginica['PetalWidthCm'], label="Iris-virginica", shade=True, ax=axes[1])





plt.tight_layout()
# Set up the matplotlib figure

f, axes = plt.subplots(ncols = 2, figsize=(10, 5), sharex=True)

sns.despine(left=True)



# Plot the Swarmplot

sns.swarmplot(x=iris_data['Species'], y=iris_data['PetalWidthCm'], ax=axes[0])



# Plot the Boxplot

sns.boxplot(x=iris_data['Species'], y=iris_data['PetalWidthCm'], ax=axes[1])



plt.tight_layout()
sns.pairplot(iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']], hue="Species", diag_kind="hist")
from sklearn.model_selection import train_test_split



y = iris_data.Species

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']

X = iris_data[features]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
data = pd.DataFrame(X_train)

data.columns = X.columns

data['label'] = y



f, axes = plt.subplots(ncols = 4, figsize=(20, 5), sharex=True)

sns.despine(left=True)



sns.kdeplot(data=data['SepalLengthCm'], shade=True, ax=axes[0])

sns.kdeplot(data=data['SepalWidthCm'], shade=True, ax=axes[1])

sns.kdeplot(data=data['PetalLengthCm'], shade=True, ax=axes[2])

sns.kdeplot(data=data['PetalWidthCm'], shade=True, ax=axes[3])

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



classifier = RandomForestClassifier(max_depth=2, random_state=0)



classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print('Confusion matrix : \n' + str(confusion_matrix(y_test, y_pred)))

print('Accuracy score : \n' + str(accuracy_score(y_test, y_pred)))
# Train / Test Split

y_2 = iris_data.Species

features = ['SepalLengthCm', 'PetalLengthCm', 'PetalWidthCm']

X_2 = iris_data[features]



X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.1, random_state=0)



# Standardization

sc = StandardScaler()

X_train_2 = sc.fit_transform(X_train_2)

X_test_2 = sc.transform(X_test_2)



# Model Development

classifier_2 = RandomForestClassifier(max_depth=2, random_state=0)



classifier_2.fit(X_train_2, y_train_2)

y_pred_2 = classifier_2.predict(X_test_2)



print('Confusion matrix : \n' + str(confusion_matrix(y_test, y_pred_2)))

print('Accuracy score : \n' + str(accuracy_score(y_test, y_pred_2)))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis(n_components=2)



X_train_lda = lda.fit_transform(X_train, y_train)

X_test_lda = lda.transform(X_test)



explained_variance = lda.explained_variance_ratio_



print(explained_variance)
n_component_1 = 0.99099846

n_component_2 = 0.00900154

percentage_of_variance_explained = n_component_1 + n_component_2

print("Percentage of variance explained = " + str(percentage_of_variance_explained*100))
classifier_lda = RandomForestClassifier(max_depth=2, random_state=0)



classifier_lda.fit(X_train_lda, y_train)

y_pred_lda = classifier_lda.predict(X_test_lda)



print('Confusion matrix : \n' + str(confusion_matrix(y_test, y_pred_lda)))

print('Accuracy score : \n' + str(accuracy_score(y_test, y_pred_lda)))