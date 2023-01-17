# Imports
import numpy as np

from scipy.stats import norm
X = np.random.randn(100, 3)
y = np.random.rand(100)
y = np.where(y >= 0.5, 1, 0)
X
y
class NativeBayesClassifier(object):
    """
    My implementation of the Gausian Naive Bayes classifier.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y):        
        # Find label classes (unique values)
        self.classes = np.unique(y)
        print("classes:", self.classes)
        
        # Compute probability of each class
        self.class_probs = np.zeros(len(self.classes))
        for cl_index, cl in enumerate(self.classes):
            self.class_probs[cl_index] = (y == self.classes[cl_index]).sum() / y.size
            
        print("class_probs:", self.class_probs)
        
        # Compute mean and stddev of each class
        self.means = []
        self.stds = []
        for cl_index, cl in enumerate(self.classes):
            X_class = X[y == self.classes[cl_index]]
            
            self.means.append(X_class.mean(axis=0))
            self.stds.append(X_class.std(axis=0))
        
    def predict(self, X):
        # Return the class that maximize the likelihood
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
    
    def predict_proba(self, X):
        # Bayes theorem
        # P(C|X) = (P(X|C) n P(C)) / P(X)
        
        # For each class, calculate the probability that X belongs to class
        probs = []
        for cl_index, cl in enumerate(self.classes):
#             X_class = self.X[y == self.classes[cl_index]]

            prob = norm.pdf(
                X, 
                loc=self.means[cl_index], # mean
                scale=self.stds[cl_index] # stddev
            )
#             print("prob.shape: {}".format(prob.shape))
            prob = np.multiply.reduce(prob, axis=1) * self.class_probs[cl_index]
            print("class: {}".format(cl))
#             print("prob: {}".format(prob))
            print("prob.shape: {}".format(prob.shape))
            print("--" * 10)
            probs.append(prob)
            
#         print(probs)
        
        likelihoods = []
        for ps in zip(*probs):
            p_sum = sum(ps)
#             print(p1/p_sum, p2/p_sum)
            likelihoods.append(list(ps/p_sum))
            
        likelihoods = np.array(likelihoods)
#         print("likelihoods:", likelihoods)

        return likelihoods
model = NativeBayesClassifier()
model.fit(X, y)
model.predict(X)
model.predict_proba(X)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, y).predict(X)
gnb.predict_proba(X)
assert np.allclose(model.predict(X), gnb.predict(X)), "Incorrect prediction"
assert np.allclose(model.predict_proba(X), gnb.predict_proba(X)), "Incorrect prediction"
%timeit model.predict(X)
%timeit gnb.predict(X)
