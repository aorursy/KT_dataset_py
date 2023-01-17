import numpy as np 

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("../input/data.csv")

df.head()
# replace "?" to NaN

df.replace("?", np.nan, inplace = True)

df.head(5)
missing_data = df.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")    
list = ['Unnamed: 32','id']

df.drop(list, axis = 1, inplace = True)
df.head()
df.describe().transpose()
df.describe(include=['object'])
B, M = df['diagnosis'].value_counts()

print('Number of Malignant : ', M)

print('Number of Benign: ', B)
df['diagnosis'].value_counts().to_frame()
import seaborn as sns

sns.set(style="darkgrid")

ax = sns.countplot(df.diagnosis,label="Count") 
import matplotlib.pyplot as plt



# Data to plot

labels = 'Benign', 'Malignant'

sizes = df['diagnosis'].value_counts()

colors = ['lightskyblue', 'orange']

explode= [0.4,0]

# Plot

plt.pie(sizes, explode=explode, labels=labels,radius= 1400 ,colors=colors,

autopct='%1.1f%%', shadow=True, startangle=90)



plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(7,7)

plt.show()
width = 12

height = 10



import itertools

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
df.columns
X= df[['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']]

X[0:5]
y=df["diagnosis"].values

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
k = 4

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
yhat = neigh.predict(X_test)

yhat[0:5]
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
Ks = 30

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 