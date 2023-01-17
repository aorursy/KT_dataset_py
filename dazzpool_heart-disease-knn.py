import pandas as pd
import seaborn as sns
sns.set() #Apply the default seaborn theme
dataset = pd.read_csv('../input/heart-disease-uci/heart.csv') #Loading the dataset
dataset.head()
type(dataset)
dataset.shape
dataset.dtypes
dataset.describe()
dataset.columns
dataset.info()
sns.countplot(dataset['sex'])
sns.countplot(dataset['cp'])
sns.countplot(dataset['cp'], hue='sex', data = dataset)
sns.countplot(dataset['target'], hue='sex', data = dataset)
sns.countplot(dataset['cp'], hue='target', data = dataset)
dataset.isna().sum()
sns.heatmap(dataset.isnull(),cbar = False)
sns.countplot(dataset['age'])
sns.distplot(dataset['age'])
dataset.corr()
sns.heatmap(dataset.corr(), annot = True)
from sklearn.model_selection import train_test_split
X = dataset.drop('target', axis = 1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
from sklearn.neighbors import KNeighborsClassifier
# Creating the model
model = KNeighborsClassifier(n_neighbors=1) #Let's try for one neighbour
model.fit(X_train, y_train) # Fitting the dataset into the model
y_pred = model.predict(X_test)
# To see the confusion matrix
from sklearn.metrics import confusion_matrix
# This is the confusion matrix of the predicted data and the actual data
confusion_matrix(y_test, y_pred)
# To see the accuracy of the model
from sklearn.metrics import accuracy_score
# Accuracy of the model when the total neighbours were one
accuracy_score(y_test,y_pred)
# Now let's what will be the accuracy when we'll be having 2 nearest neighbours
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
# For 3 now,
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
# Now let's check that what is the number of neighbours that fit our model, best.
accuracy = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_pred))
accuracy
import matplotlib.pyplot as plt
plt.plot(range(1,50), accuracy, marker='o')
# This is the highest accuracy we are getting of our model
max(accuracy)
accuracy.index(max(accuracy))
import numpy as np
error_rate = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error_rate.append(np.mean(y_test != y_pred))
plt.plot(range(1,50), error_rate, marker='o')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = dataset.drop('target', axis = 1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
X_train_sc = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)  
y_train = y_train.values
y_train_sc = sc.fit_transform(y_train.reshape(-1,1)) #Reshaping it, otherwise we'll get a 1-D array and it'll also give us an error
y_train_sc_flatten = y_train_sc.flatten() #Flattening it
y_test = y_test.values
y_test_sc = sc.fit_transform(y_test.reshape(-1,1))
y_test_sc_flatten = y_test_sc.flatten()
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
y_train_sc_flatten_encoded = lab_enc.fit_transform(y_train_sc_flatten)
y_test_sc_flatten_encoded = lab_enc.fit_transform(y_test_sc_flatten)
from sklearn.neighbors import KNeighborsClassifier
accuracy_improved = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train_sc, y_train_sc_flatten_encoded)
    y_pred = model.predict(X_test)
    accuracy_improved.append(accuracy_score(y_test_sc_flatten_encoded,y_pred))
accuracy_improved
max(accuracy_improved)