# Import required libraries and data 
import pandas as pd
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
iris = pd.read_csv("../input/Iris.csv")

# Let's see what data looks like:
iris.head()
# Let's see how many examples we have of each species
print(iris["Species"].value_counts())
print(sns.countplot(iris['Species']))


iris.info() 
iris.drop('Id',axis=1,inplace=True) 

# useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations

# Finally, we can looki at univariate relations in the kdeplot in diagonal,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.pairplot(data=iris,hue='Species',palette='Dark2',)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
iris.boxplot(by="Species", figsize=(12, 6))
plt.figure(figsize=(10,5)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r') 
plt.show()
from sklearn.model_selection import train_test_split
iris.columns
X = iris.drop('Species',axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.svm import SVC
model = SVC(gamma='auto')
model.fit(X_train,y_train)

predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, predictions))
print( confusion_matrix(y_test, predictions))
# Sample data
sample = [6,3,2,1]

# But model.predict does not accept one dimension data and accepts 2D data, therefore, let's convert it:
import numpy as np
sample = np.reshape(sample, (1, 4))
sample

model.predict(sample)
from sklearn.linear_model import LogisticRegression
LogR = LogisticRegression()
LogR.fit(X_train,y_train)
prediction=LogR.predict(X_test)
print('The accuracy of the Logistic Regression is',accuracy_score(prediction,y_test))
from sklearn.tree import DecisionTreeClassifier
DecTree = DecisionTreeClassifier()
DecTree.fit(X_train,y_train)
prediction=DecTree.predict(X_test)
print('The accuracy of the Decision Tree is',accuracy_score(prediction,y_test))
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train,y_train)
prediction=KNN.predict(X_test)
print('The accuracy of the K nearest neighbor is',accuracy_score(prediction,y_test))