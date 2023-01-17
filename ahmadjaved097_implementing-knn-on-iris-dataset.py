import warnings                          # to hide error messages(if any)
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('../input/Iris.csv')
df.head()
#Removing unnecessary column i.e. Id
df = df.drop(['Id'], axis = 1)

df.shape
df.dtypes
df.isnull().sum()
df['Species'].unique()
df['Species'].value_counts()
df.describe()
sns.countplot(x = 'Species', data = df)
plt.show()
corr = df.corr()
plt.figure(figsize = (10,6))

#Drawing a heatmap to show how various features are correlated

sns.heatmap(corr,annot = True)
plt.yticks(rotation = 45)
plt.show()
sns.pairplot(df, hue = 'Species')
plt.show()
#Scatter plot between petal length and petal witdth
plt.figure(figsize = (10,6))
sns.lmplot(x = 'PetalLengthCm', y = 'PetalWidthCm',data = df, hue = 'Species')
plt.show()
plt.figure(figsize =(10,6) )
sns.lmplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = df, hue = 'Species')
#swarmplot
plt.figure(figsize = (10,6))
sns.swarmplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = df, hue = 'Species')
plt.show()

#Box Plot
plt.figure(figsize =(10,7) )
sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = df)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = np.array(df.iloc[:,0:4])
y = np.array(df['Species'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33, random_state = 42)
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
neighbors = []     #empty list to store the number of neighbors
cv_scores = []     #empty list to score cross validation scores

from sklearn.model_selection import cross_val_score
for i in range(1,51,2):
    neighbors.append(i)
    knn = KNeighborsClassifier(n_neighbors = i)
    
    #Performing 10 fold cross-validation
    
    scores = cross_val_score(knn,X_train,y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())

#Misclassification error rates
MSE = [1-x for x in cv_scores]

#determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is %d ' %optimal_k)
#Plotting misclassification versus k(number of nearest neighbors)

sns.set()
plt.figure(figsize = (10,6))
plt.plot(neighbors,MSE, 'c')
plt.xlabel('Neighbors')
plt.ylabel('Misclassification Error Rate')
plt.title('Misclassification Error Rate vs. Nearest Neighbors')
plt.show()