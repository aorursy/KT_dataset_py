import pandas 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pandas.read_csv("../input/lab02/iris.csv", names=names) # Load dataset
df.head(3) # first three samples
# check the version of scikit-learn
import pandas
import sklearn
print('sklearn version')
print(sklearn.__version__)
# Load dataset
file = "../input/lab02/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pandas.read_csv(file, names=names)
# shape
print(df.shape)
# head
print(df.head(5))
# tail
print(df.tail(5))
# class distribution
print(df.groupby('class').size())
# Plot histograms
import matplotlib.pyplot as plt
df.hist()
plt.show()
#split dataset in features and target variable
X = df.iloc[:,0:4] # we choose all four features
#X = df.iloc[:, 0:2] # we choose first two features as X.  
y = df.iloc[:,4]
#show y
print(y)
from sklearn.model_selection import train_test_split # Import train_test_split function
#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10) # 50% training and 50% test with random seed = 10
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion='entropy')
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from matplotlib import pyplot as plt 
from sklearn import tree
clf.fit(X, y) # classify all data
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, feature_names=df.columns,class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'],filled=True)