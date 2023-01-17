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
# necessary import

# for visualization
import seaborn as sns
from matplotlib import pyplot as plt


# for dimension reduction
from sklearn.manifold import TSNE

# ML algorithms
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

# ML algorithms evaluations
from sklearn.metrics import confusion_matrix
print('setup complete')
# the dataset have already been split into train, test set
test_set = "/kaggle/input/mnist-in-csv/mnist_test.csv"
train_set = "/kaggle/input/mnist-in-csv/mnist_train.csv"

# a quirk of mine, to make me immediately aware when the code execution end
# or if it has already been executed
print('setup complete') 
# dump the train dataset into pandas dataframe for easy manipulation
# and plotting (don't touch the test set until when you are finally going to
# test in on the final model)

X_train = pd.read_csv(train_set)
X_test = pd.read_csv(test_set)
X_train.head()
# to check out what we are going to be working with
X_train.info()
# remove target label
y_train = X_train['label'].copy()
X_train.drop('label', axis=1, inplace=True)
print('setup complete')
# you can change the random_image value to visualize any other image
random_image = 15000
some_digit = X_train.iloc[random_image] # select any number, change to select any number
plt.imshow(some_digit.values.reshape(28, 28))
print('label', y_train.iloc[random_image])
# to check if the samples are evenly distributed

plt.bar(y_train.value_counts().index, y_train.value_counts())
# the digit image 5 has the lowest number of instances while image 1 has the highest
# but all within acceptable ranges, so Good to Go
# we will visualize the datasets to gain insights
tsne = TSNE(n_components=2)

# will only visualize only 5000 examples of the datasets
# if you have the time you can fit to all the examples to visualize it 

X_train_tsne = tsne.fit_transform(X_train[:5000])
print('setup complete')
X_train_tsne.shape
colors = ['blue', 'green', 'red', 'orange', 'yellow', 'purple', '#143D9F',
          '#9A99AB', '#79543A', 'black']
y_plot = y_train[:5000] 
plt.figure(figsize=(10, 8))
for i in range(10):
    # select only the indexes of each labeled examples
    index = y_plot == i
    plt.scatter(X_train_tsne[index, 0], X_train_tsne[index, 1],
                c=colors[i], label=f"{i}")
plt.legend()
print('setup complete')
knn = KNeighborsClassifier(n_jobs=-1)
scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
print(f"Scores:\t{scores}\nMean:\t{scores.mean()}\nStd:\t{scores.std()}")
predictions = cross_val_predict(knn, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, predictions)
row_sum = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sum
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
conf_mx
# first let see why 4 is mistaken as 9
a, b = 4, 9
true_positive = X_train[(y_train == a) & (predictions == a)]

# what we really need to look at

# number 4 mistaken as 9
false_negative = X_train[(y_train == a) & (predictions == b)]
# number 9 mistaken as 4
false_positive = X_train[(y_train == b) & (predictions == a)]

# don't need this
true_negative = X_train[(y_train == b) & (predictions == b)]
def plot_mistaken_identities(array):
    dim = int(np.sqrt(len(array))) + 1
    fig, axes = plt.subplots(dim, dim)
    for axe, img in zip(axes.ravel(), array.values):
        axe.imshow(img.reshape(28, 28))
plot_mistaken_identities(false_negative.iloc[:15])
# from the below plot the number 4 that is wrongly classified as 9 are
# those that actually looks like number 9! and are something if we humans
# will have errors in classifying (tag someone who has handwriting that
# makes reading their books a challenge, thank God for word processing SW)
plot_mistaken_identities(false_positive.iloc[30:])
# Jesus! some people need to go back to creche school to relearn how to 
# write numbers, or maybe this number were written by creche students who 
# hasn't yet mastered how to write

# because I see no reason why a 9 should look as exactly as 4, then how
# would your 4 look like then. YOU ARE BREAKING MY MODEL!! and making it look
# dumb