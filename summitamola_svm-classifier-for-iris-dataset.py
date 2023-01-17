#importing dependencies
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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
# Load dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("../input/IRIS Dataset.csv", names=names)

#Shape of the Dataset
print(dataset.shape)

print ("-------------------------------------------------------------------")

#Peek at the Data
print(dataset.head(20))


print ("-------------------------------------------------------------------")

# descriptions
print(dataset.describe())


print ("-------------------------------------------------------------------")

# class distribution
print(dataset.groupby('class').size())
