import pandas as pd

iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()
#What is the dataset Size (rows, columns)
iris.shape
# what datatypes we have?
iris.dtypes
# Do we have missing values?
iris.isnull().sum(axis = 0)
# how many species do we have?
iris['Species'].unique()
# How many record of each species?
iris.groupby('Species').size()

# Statistical Summary of each numeric variable
iris.describe()
#Categorical Variable stats
iris['Species'].describe()
from matplotlib import pyplot
iris.hist()
pyplot.show()
# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(iris)
pyplot.show()
# Creating the X dataset, which includes all variable except target variable
X= iris.drop(columns=["Id", "Species"])
X
# Creating the y dataset, which included the target (dependant) variable only
target=  iris.drop(["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], axis=1)
target
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


for name, model in models:
    accuracy = cross_val_score(model, X, target.values.ravel(), scoring='accuracy', cv = 10)
    print("Accuracy of %s: is %.2f percent" % (name ,accuracy.mean()*100))
    
      
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size=0.20, random_state=83)
print ("X_train (rows, columns): ", X_train.shape)
print ("Y_train (rows, columns): ", Y_train.shape)
print("X_test (rows, columns): ", X_test.shape)
print ("Y_test (rows, columns): ", Y_test.shape)

# view the datasets

print ("X_train: ", X_train)
print ("Y_train: ", Y_train)
print("X_test: ", X_test)
print ("Y_test: ", Y_test)
# Make predictions on testing dataset
from sklearn.svm import SVC

model = SVC(gamma='auto')
model.fit(X_train, Y_train.values.ravel())
predictions = pd.DataFrame(model.predict(X_test), columns=['predictions'])
#Compare predictions to actual values
pd.concat([X_test.reset_index(drop='True'), Y_test.reset_index(drop='True'),predictions],axis=1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Accuracy: (True Positives + True Negatives)/(Total observations) (28/30=93.33%)
print("Accuracy Score is:" ,accuracy_score(Y_test, predictions)*100)

# Confusion Matric showing the two observations that we got wrong
print(confusion_matrix(Y_test, predictions))

# Precision=TP/(TP+FP)   Recall=TP/(TP+FP)
# Recall for Iris-virginica = 9/11= 82%
print(classification_report(Y_test, predictions))
