from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean
from numpy import std

def fit_distribution(data):
    mu = mean(data)
    sigma = std(data)
    print(mu, sigma)
    dist = norm(mu, sigma)
    return dist
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
print(Xy0.shape, Xy1.shape)
priory0 = len(Xy0) / len(X) #p(y=0)
priory1 = len(Xy1) / len(X) #p(y=1)
print(priory0, priory1)
distX1y0 = fit_distribution(Xy0[:, 0])
distX2y0 = fit_distribution(Xy0[:, 1])
distX1y1 = fit_distribution(Xy1[:, 0])
distX2y1 = fit_distribution(Xy1[:, 1])
def probability(X, prior, dist1, dist2):
    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])
 
Xpredict, ypredict = X[15], y[15]
prob_0 = probability(Xpredict, priory0, distX1y0, distX2y0)
prob_1 = probability(Xpredict, priory1, distX1y1, distX2y1)
class_label = 0 if prob_0>prob_1 else 1
print(f'predict label:{class_label}')
print(f'True label: y={ypredict}')
