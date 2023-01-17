# Importing libraries 
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Loading data
iris = load_iris()
iris.data[0:5], iris.target
# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(generations=8, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print("Accuracy is {}%".format(tpot.score(X_test, y_test)*100))
tpot.export('tpot_iris_pipeline.py')
