# Load libraries
import pandas
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
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
# shape
print(dataset.shape)
# histograms
dataset.hist()
plt.show()
# It looks like perhaps two of the input variables have a Gaussian distribution. 
#This is useful to note as we can use algorithms 
#that can exploit this assumption
# Multivariate plots to better understand the relationships between attributes.
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
#The diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.


#We will split the loaded dataset into two, 80% of which we will use to train our models and 
#20% that we will hold back as a validation dataset.
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


print (len(Y_train),len(Y_validation))
#so our training and validation set is ready with 80:20 ratio
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#We will use 10-fold cross validation to estimate accuracy.
#This will split our dataset into 10 parts, train on 9 and test on 1 
#and repeat for all combinations of train-test splits.

# we will evaluate using  6 different algorithms.
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)#here e use kfold validation for scoring
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)#apply dataset on train data
    results.append(cv_results)
    names.append(name)
    #here we are using supervised method as we are giving what could be the result to given characteristics
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    #here we can see  KNN has the largest estimated accuracy score.
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# The box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy
# Make predictions on validation dataset
# Here we apply our built model on test data set so to see that our model does not overfit and it gives an
# idea whether our model is appropiate for prediction on other datasets
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))