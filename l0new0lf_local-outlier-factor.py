import numpy as np

import matplotlib.pyplot as plt

np.random.seed(42)



from sklearn.neighbors import LocalOutlierFactor
X = [

       [ -1.1], 

       [  0.2], 

       [101.1], 

       [  0.3]

]
clf = LocalOutlierFactor(n_neighbors=2)

clf.fit_predict(X)
clf.negative_outlier_factor_
# Generate dataset



# x-cood

xs_1    = np.random.normal(-2, 1, 50).reshape(50,1)

ys_1    = np.random.normal(-2, 1, 50).reshape(50,1)

es_xs_1 = np.ones(xs_1.shape[0]).reshape(50,1)



xs_1_out    = np.random.normal(-2, 0.3, 5).reshape(5,1)

ys_1_out    = np.random.normal( 2, 0.3, 5).reshape(5,1)

es_xs_1_out = (np.zeros(xs_1_out.shape[0]) - 1).reshape(5,1)



# y-cood

xs_2    = np.random.normal(3, 0.5, 50).reshape(50,1)

ys_2    = np.random.normal(3, 0.5, 50).reshape(50,1)

es_xs_2 = np.ones(xs_2.shape[0]).reshape(50,1)



xs_2_out = np.random.normal(4, 0.2, 5).reshape(5,1)

ys_2_out = np.random.normal(0, 0.2, 5).reshape(5,1)

es_xs_2_out = (np.zeros(xs_2_out.shape[0]) - 1).reshape(5,1)



xs = np.vstack((xs_1, xs_2, xs_1_out, xs_2_out))

ys = np.vstack((ys_1, ys_2, ys_1_out, ys_2_out))

es = np.vstack((es_xs_1, es_xs_2, es_xs_1_out, es_xs_2_out))



X = np.hstack((xs, ys))

is_outlier = es.flatten()
plt.xlim((-5,5))

plt.ylim((-5,5))

plt.scatter(X[:,0], X[:,1], c=is_outlier)

plt.show()
# default:

K = 20

C = 0.1



# fit the model for outlier detection 

clf = LocalOutlierFactor(n_neighbors=K, contamination=C)
y_pred = clf.fit_predict(X)

n_errors = (y_pred != is_outlier).sum()

X_scores = clf.negative_outlier_factor_
y_pred # remove corresponding xs from dataset to clean outliers
print(f"errors: {n_errors}")

print(f"accuracy: {(1 - (n_errors / 10))*100}") #10: total outliers
plt.title("Local Outlier Factor (LOF)")



# plot all point

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=5., label='Data points')



# plot circles with radius proportional to the outlier scores

radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',

            facecolors='none', label='Outlier scores')



plt.axis('tight')

plt.xlim((-5, 5))

plt.ylim((-5, 5))

plt.xlabel("prediction errors: %d" % (n_errors))

legend = plt.legend(loc='upper left')

legend.legendHandles[0]._sizes = [10]

legend.legendHandles[1]._sizes = [20]

plt.show()