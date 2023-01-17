import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score



# Importing the dataset

dataset = pd.read_csv("../input/data.csv")

# Check for NULL values

dataset.info()

dataset.isnull().sum()
# Examining some Features

for i in range(1,100,5):

    plt.scatter(dataset['y'], dataset.iloc[:,i], color = 'red')

    #plt.plot(X, regressor.predict(X), color = 'blue')

    plt.title('Seizure Features')

    plt.xlabel('Brain Function')

    plt.ylabel('Brain Recording')

    plt.show()
'''

Modeling will clearly have the most difficult time differentiating between a seizure and 

    recording of brain area where tumor is located (1 & 2)

All the features have some zero/near-zero readings and will likely reduce accuracy

Used the Naive Bayes classification technique because it can handle nonlinear problems, 

    isn't biased by outliers and can handle a large number of features if necessary

Applied kPCA dimensionality reduction mostly for visualization purposes.

'''



# Creating variables to be used later in feature analysis

d1 = dataset.iloc[:,1:178][dataset['y']==1]

d2 = dataset.iloc[:,1:178][dataset['y']==2]

d3 = dataset.iloc[:,1:178][dataset['y']==3]

d4 = dataset.iloc[:,1:178][dataset['y']==4]

d5 = dataset.iloc[:,1:178][dataset['y']==5]



# Give non-seizure patients zero values (avoided for loop - might check individual instances later)

dataset['y'] = dataset['y'].replace([5], [0]).ravel()

dataset['y'] = dataset['y'].replace([3], [0]).ravel()

dataset['y'] = dataset['y'].replace([4], [0]).ravel()

dataset['y'] = dataset['y'].replace([2], [0]).ravel()



X = dataset.iloc[:, 1:178].values

y = dataset.iloc[:, 179].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Feature Scaling (MUST BE APPLIED IN DIMENSIONALITY REDUCTION)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Applying kPCA (non-linear)

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components = 2, kernel = 'rbf')

X_train = kpca.fit_transform(X_train)

X_test = kpca.transform(X_test)



# Fitting Naive Bayes Classification to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



predictions = [round(value) for value in y_pred]



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, predictions)

print("Area Under the Receiver Operating Characteristic Curve: %.2f%%" % roc_auc)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
# Visualising the Training set results

from matplotlib.colors import ListedColormap

f = plt.figure(figsize=(12, 12))

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()
# Visualising the Test set results

from matplotlib.colors import ListedColormap

f = plt.figure(figsize=(12, 12))

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Test set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()
'''

As expected, the model was able to guess the majority of non-seizure cases and struggled and

    struggled more with positive results.  

Reducing and using only the most significant features from the original dataset before reducing to 2

    dimensions will hopefully provide a jump in accuracy.

'''



# Average value of each feature

d1_avg, d2_avg, d3_avg, d4_avg, d5_avg = [], [], [], [], []

for i in range(0,177):

    d1_avg.append(d1.iloc[:,i].sum()/177)

    d2_avg.append(d2.iloc[:,i].sum()/177)

    d3_avg.append(d3.iloc[:,i].sum()/177)

    d4_avg.append(d4.iloc[:,i].sum()/177)

    d5_avg.append(d5.iloc[:,i].sum()/177)

    

# Difference between seizure feature averages and normal brain averages

d12_dif, d13_dif, d14_dif, d15_dif = [], [], [], []

for d1s, d2s in zip(d1_avg, d2_avg):

    d12_dif.append(d1s-d2s)

for d1s, d3s in zip(d1_avg, d3_avg):

    d13_dif.append(d1s-d3s)

for d1s, d4s in zip(d1_avg, d4_avg):

    d14_dif.append(d1s-d4s)

for d1s, d5s in zip(d1_avg, d5_avg):

    d15_dif.append(d1s-d5s)



# Determine the indices with the largest average difference and likely impact the dependend variable the most

d_ind = []

for d12 in d12_dif:

    if d12 > 150:

        d_ind.append(d12_dif.index(d12))

        

d3_ind = []

for d13 in d13_dif:

    if d13 > 150 and d13_dif.index(d13) not in d_ind:

        d_ind.append(d13_dif.index(d13))

        

d4_ind = []

for d14 in d14_dif:

    if d14 > 150 and d14_dif.index(d14) not in d_ind:

        d_ind.append(d14_dif.index(d14))

        

d5_ind = []

for d15 in d15_dif:

    if d15 > 150 and d15_dif.index(d15) not in d_ind:

        d_ind.append(d15_dif.index(d15))



        
X_top_ind = dataset.iloc[:, d_ind].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_top_ind, y, test_size = 0.3, random_state = 0)



# Feature Scaling (MUST BE APPLIED IN DIMENSIONALITY REDUCTION)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Applying kPCA (non-linear)

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components = 2, kernel = 'rbf')

X_train = kpca.fit_transform(X_train)

X_test = kpca.transform(X_test)



# Fitting Naive Bayes Classification to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



predictions = [round(value) for value in y_pred]



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, predictions)

print("Area Under the Receiver Operating Characteristic Curve: %.2f%%" % roc_auc)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
'''

Isolating and including only the most impactful features resulted in a slight increase in accuracy.

The improvement was primarily from more accurate positive predictions (seizures)

'''
# Visualising the Training set results

from matplotlib.colors import ListedColormap

f = plt.figure(figsize=(12, 12))

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()