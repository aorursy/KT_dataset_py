# import the relevant packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib.colors import ListedColormap
import seaborn as sns

h = .02  # step size in the mesh
n_neighbors = 12
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# read the data from the CSV file
data = pd.read_csv("../input/creditcard.csv")
data.describe()
data.columns
data.dtypes
data['Class'].value_counts()
data_Class0 = data.loc[data['Class'] == 0]
data_Class1 = data.loc[data['Class'] == 1]

n=(int)(2*data_Class0.shape[0]/data_Class1.shape[0])
data1 = data_Class1.append(data_Class0.sample(n))
data1 = shuffle(data1)

data1['Class'].value_counts()
Y = data1.loc[:,'Class']
X = data1.loc[:,['V11','V12']]

# Normalize the DataSet to reduce chances of increased optimization iteration
X = (X - X.mean())/X.std()
X.describe()
model = KNN(n_neighbors)
model.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X.loc[:, 'V11'].min() - 1, X.loc[:, 'V11'].max() + 1
y_min, y_max = X.loc[:, 'V12'].min() - 1, X.loc[:, 'V12'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize = (20,10))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.get_cmap('GnBu', 2))

# Plot also the training points
plt.scatter(X.loc[:, 'V11'], X.loc[:, 'V12'], c=Y, cmap=plt.cm.get_cmap('GnBu_r', 20),edgecolor='k', s=50)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))

plt.show()
X = data1.loc[:,'V1': 'V28']
X = (X - X.mean())/X.std()
Y = data1.loc[:, 'Class']
model = KNN(n_neighbors)
model.fit(X, Y)
X = data.loc[:,'V1':'V28']
X = (X - X.mean())/X.std()
Z = model.predict(X)
true_class = data.loc[:,'Class']
model.score(X, true_class)
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))
f.suptitle('2 Features at a time')
ax1.scatter(data[['V1']], data[['V2']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax2.scatter(data[['V3']], data[['V4']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax3.scatter(data[['V5']], data[['V6']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax4.scatter(data[['V7']], data[['V8']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax5.scatter(data[['V9']], data[['V10']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax6.scatter(data[['V11']], data[['V12']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))
f.suptitle('2 Features at a time')
ax1.scatter(data[['V13']], data[['V14']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax2.scatter(data[['V15']], data[['V16']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax3.scatter(data[['V17']], data[['V18']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax4.scatter(data[['V19']], data[['V20']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax5.scatter(data[['V21']], data[['V22']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax6.scatter(data[['V23']], data[['V24']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
f, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(20,10))
f.suptitle('2 Features at a time')
ax1.scatter(data[['V25']], data[['V26']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
ax2.scatter(data[['V27']], data[['V28']], c=data[['Class']], cmap=plt.cm.get_cmap('terrain', 20),edgecolor='k', s=50)
X = data1.loc[:,["V2", "V3", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18"]]
Y = data1.loc[:,"Class"]
model = KNN(n_neighbors)
model.fit(X, Y)
model.score(data.loc[:,["V2", "V3", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18"]], data.loc[:,"Class"])