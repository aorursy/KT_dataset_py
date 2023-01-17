import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from scipy.stats import norm

np.random.seed(2)

x0 = np.random.normal(0,1,(100,1))

x1 = np.random.normal(2,1,(10000,1))



y0  = np.zeros((100,1))

y1  = np.ones((10000,1))



x = np.concatenate((x0,x1), axis=0)

y = np.concatenate((y0,y1), axis=0)

sgd_clf = SGDClassifier()

sgd_clf.fit(x, y.ravel())

x_new = np.arange(-5,5,0.01)

y_predict = [sgd_clf.predict([[x_n]]) for x_n in x_new]



plt.scatter(x_new, y_predict), plt.title("Class definition for range of x values")

plt.xlabel('input value')

plt.ylabel('class')

plt.grid(), plt.show()

mu0, std0 = norm.fit(x0)

mu1, std1 = norm.fit(x1)



x_new_instance            = -2

likelihood_of_class0      = norm.pdf(x_new_instance, mu0, std0)

likelihood_of_class1      = norm.pdf(x_new_instance, mu1, std1)

print(likelihood_of_class0, likelihood_of_class1)


tot             = 10000+100

prior_of_class0 = 100/tot

prior_of_class1 = 10000/tot

print('Bayesian probability of ' + str(x_new_instance)+ ' being in class 0')

print(prior_of_class0*likelihood_of_class0/(prior_of_class0*likelihood_of_class0 + prior_of_class1*likelihood_of_class1))
