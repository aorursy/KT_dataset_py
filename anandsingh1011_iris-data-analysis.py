

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
%matplotlib inline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
plt.style.use("ggplot")

df = pd.read_csv("../input/Iris.csv", header = 0)

df.head()
df.describe()
# visualization of Specie Type by Length
sns.barplot(x="Species", y="SepalLengthCm", data=df)
sns.set(rc={'figure.figsize':(12,16)})

plt.xlabel('Specie Type')
plt.ylabel('Sepal Length in cm')
plt.title('Specie Type by Length')

plt.show()
# Split Data

X = df.iloc[:,0:4].values
Y = df.iloc[:,-1].values

#Where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica
labelencoder_Y=LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# we can see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sns.pairplot(df.drop("Id", axis=1), hue="Species", size=3)
sns.set(rc={'figure.figsize':(12,16)})
plt.show()
df.hist(layout=(3,2),figsize=(12,16))
plt.show()
# Box plots visualization
df.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
plt.show()


# ========================= Create Model with LogisticRegression Algorithms ========================= 
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)

# Plot confusion matrix
#plt.imshow(cm, interpolation='nearest')
#plt.colorbar()
#plt.show()
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)
plt.show()
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=21, stratify=Y)
knn=KNeighborsClassifier(n_neighbors= 8)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)

print("Score : ",knn.score(x_test,y_test))

print('Confusion Matrix :')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
#plt.imshow(cm, interpolation='nearest')
#plt.colorbar()
#plt.show()
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)
plt.show()
