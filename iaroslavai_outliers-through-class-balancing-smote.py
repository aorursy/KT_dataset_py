import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

np.random.seed(1)

# Example disballanced data to be fixed by SMOTE
# Data with positive class points surrounding two 
# clusters of negative data points.
X = np.random.randn(200, 2)
y = np.sum(X ** 2, axis=-1) > 1
I = 2*(np.random.rand(len(X)) > 0.5)-1
X = (X.T+I.T).T

# Make data disballanced
I = np.copy(y)
I[::10] = False
I = ~I
X = X[I]
y = y[I]

print('Positive class instances: %s' % np.sum(y == True))
print('Negative class instances: %s' % np.sum(y == False))

# resample
smote = SMOTE()
Xr, yr = smote.fit_resample(X, y)

# visualize results
def plot_data(X, y, title):
    plt.title(title)
    plt.scatter(X[~y, 0], X[~y, 1])
    plt.scatter(X[y, 0], X[y, 1])
    
plt.subplot(1, 2, 1)
plot_data(X, y, 'Original')
plt.subplot(1, 2, 2)
plot_data(Xr, yr, 'SMOTE')
plt.show()