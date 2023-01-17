# Loading important classes

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



# Loading the data

data = pd.read_csv(r"/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction", "Age", "Outcome"]

data.info()
data.describe()
data.isnull().sum()
data.eq(0).sum()
data.head()
# QQ plot

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Distributions of Features')

for ax, feature in zip(axes.flatten(),  names):

    sns.distplot(data[feature],  ax=ax)

plt.show()



# QQ plot

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,12))

fig.subplots_adjust(hspace=0.6)

fig.suptitle('Relation between Target and Predictor variables')

for ax, feature in zip(axes.flatten(),  names[:-1]):

    sns.barplot(x = data[feature], y = data['Outcome'], palette = 'Set3',data = data, ax = ax, orient = 'h')

plt.show()



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = 0, strategy = 'mean')

data.iloc[:,1:6] = imputer.fit_transform(data.iloc[:, 1:6])

data.eq(0).sum()
fig, axes = plt.subplots(figsize = (10,10))

sns.heatmap(data.corr(method  = 'spearman'), cmap="YlGnBu", ax = axes,  annot = True)
f, axes = plt.subplots(nrows=1, ncols=3, figsize = (14, 5))

fig.subplots_adjust(hspace=0.6)

fig.suptitle('Relation between Target and Predictor variables')

sns.lineplot(x = data['Age'], y = data['Pregnancies'], hue = data['Outcome'], ax = axes[0])

sns.lineplot(x = data['Insulin'], y = data['Glucose'], hue = data['Outcome'], ax = axes[1])

sns.lineplot(x = data['BMI'], y = data['SkinThickness'], hue = data['Outcome'], ax = axes[2] )

plt.show()

# Splitting the data into test and train

X = data.iloc[:, :-1]

Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)





# k nearest neighbors

# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, Y_train)



# Predicting the Test set results

Y_pred = classifier.predict(X_test)



# Accuracy score

from sklearn.metrics import accuracy_score

score = accuracy_score(Y_test, Y_pred)

print ('Accuracy score = ',score)



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Distributions of Features')

for ax, feature in zip(axes.flatten(),  names):

    sns.distplot(data[feature],  ax=ax)

plt.show()

from sklearn.preprocessing import KBinsDiscretizer  

disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

X_binned = disc.fit_transform(X) 

X_binned
Y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_binned, Y, test_size = 0.25, random_state = 0)





# k nearest neighbors

# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train1, Y_train1)



# Predicting the Test set results

Y_pred1 = classifier.predict(X_test1)



# Accuracy score

from sklearn.metrics import accuracy_score

score1 = accuracy_score(Y_test1, Y_pred1)

print ('Accuracy score = ',score1)
