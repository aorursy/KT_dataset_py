import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
import os
path = '../input/'
print(os.listdir(path))
df = pd.read_csv(path + 'train.csv')
# Shape of the dataset
df.shape
# Columns
df.columns
# Target variable = label
df.label.value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,5))
labels = df.label.value_counts()
sns.barplot(labels.index, labels.values )
plt.show()
def print_image(image_num):
    an_image = df.iloc[image_num,1:]
    an_image_matrix = an_image.values.reshape(28,28)
    print('Image Label - {}'.format(df.iloc[image_num,:1].label))
    plt.imshow(an_image_matrix)
    plt.show()
print_image(801)
print_image(800)
X = df.iloc[:,1:]
y = df.iloc[:,:1]
X[X>0] = 1
from sklearn.model_selection import train_test_split
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:1000,:], y.iloc[:1000,:], random_state = 0, train_size=0.7)
classifier = svm.SVC()
classifier.fit(X_train, y_train.values.ravel())
classifier.score(X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:5000,:], y.iloc[:5000,:], random_state = 0, train_size=0.7)
classifier = svm.SVC()
classifier.fit(X_train, y_train.values.ravel())
classifier.score(X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:10000,:], y.iloc[:10000,:], random_state = 0, train_size=0.7)
classifier = svm.SVC(C=10, kernel='rbf', gamma=0.01)
classifier.fit(X_train, y_train.values.ravel())
classifier.score(X_test, y_test)
classifier = svm.SVC(C=7, kernel='rbf', gamma=0.01)
classifier.fit(X_train, y_train.values.ravel())
classifier.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV
def tune_svm_parameters(X, y, cv=5):
    C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    gamma_list = [0.001, 0.01, 0.1, 1]
    kernel_list = ['rbf', 'linear', 'poly']
    param_grid = {'kernel':kernel_list, 'C': C_list, 'gamma' : gamma_list}

    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv)
    grid_search.fit(X, y.values.ravel())

    return grid_search.best_params_
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:1000,:], y.iloc[:1000,:], random_state = 0, train_size=0.7)
tune_svm_parameters(X.iloc[:1000,:], y.iloc[:1000,:])
