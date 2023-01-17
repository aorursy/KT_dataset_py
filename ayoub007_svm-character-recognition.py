#Importing Libraries :
import matplotlib.pyplot as plt #visulization
from sklearn import datasets #pre-prepared dataset that we can directly use
from sklearn import svm #SupportVectorMachine
#Importing dataset :
#I am using a pre-existing data from the scikit-learn library :
digits = datasets.load_digits()
print(digits.data) #digit.data = features
print(digits.target) #digits.target = actual label (ground truth)
clf = svm.SVC(gamma=0.001, C=100) #SVM Classifier
#Train vector :
X = digits.data[:-10] #take all elements but leave the last 10 for testing

#Test vector :
y = digits.target[:-10] #same, just leave the last 10 elements (to use them for predictions later)
#fitting the classifier :
clf.fit(X, y)

#test on one element :

#classifier.predict has some problems so:
    #use of reshape(-1, 1) when the data has one feature (one column)
    #use of reshape(1, -1) when the data is a sample data
print(clf.predict(digits.data[-5].reshape(1, -1))) #element number 5 from buttom (so it is not in the training set)
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')     
plt.show()
#Other test :
print(clf.predict(digits.data[-9].reshape(1, -1)))
plt.imshow(digits.images[-9], cmap=plt.cm.gray_r, interpolation='nearest')     
plt.show()
#Other test :
print(clf.predict(digits.data[-1].reshape(1, -1)))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')     
plt.show()