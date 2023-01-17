# First I will import the main libraries to manage and plot datas.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

% matplotlib inline 
#This code is to turn more practical when I want to see a graph.

# Now I will read the dataset.
iris = pd.read_csv('../input/Iris.csv')
# Let's see basic layouts about this data.
iris.head()
# First of all, we can improve the title of the columns.
iris.columns = ['ID', 'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species' ]

# Let's if this worked!
iris.head()
iris.info()
# I want to see how much species there are.
iris['Species'].value_counts()
# I will start to build some graph to visualize data and guess what model will be better to use.
sns.pairplot(iris, hue='Species')

# I focus in "Species", because is my goal.
# First I will determinate the "X"and "y"values.

# The independent variables are the features of species: index 1,2,3 and 4.
X = iris.iloc[:,1:5]

# The dependent variable is the names of species: index 5.
y = iris.iloc[:,5]

# PS.: The "ID" is not necessary to my algorithmn.
# Let's check!
X.head()
y.head()
# Now it's time to separate in Train an Test set.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Let's check if the class done correctly the separation.
print(len(X_train))
print('\n')
print(len(y_train))
# that's fine! Now it's time to train in SVM algorithmn.
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
# Now we have to predict using our model.
y_pred = classifier.predict(X_test)
# Let's see with this model works!
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))