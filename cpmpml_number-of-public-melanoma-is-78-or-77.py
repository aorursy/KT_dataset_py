import numpy as np # linear algebra
from sklearn.metrics import roc_auc_score
# 1000 samples
n = 1000

# ground truth
y = np.zeros(n)
y[-78:] = 1

# probing prediction with one known positive set to 1, rest set to 0
yhat = np.zeros(n)
yhat[-1] = 1

# score
roc_auc_score(y, yhat)
# 3300 samples
n = 3300

# ground truth
y = np.zeros(n)
y[-78:] = 1

# probing prediction with one known positive set to 1, rest set to 0
yhat = np.zeros(n)
yhat[-1] = 1

# score
roc_auc_score(y, yhat)
# 3300 samples
n = 3300

# ground truth with 77 positive
y = np.zeros(n)
y[-77:] = 1

# probing prediction with one known positive set to 1, rest set to 0
yhat = np.zeros(n)
yhat[-1] = 1

# score
roc_auc_score(y, yhat)
# 3300 samples
n = 3300

# ground truth with 79 positive
y = np.zeros(n)
y[-79:] = 1

# probing prediction with one known positive set to 1, rest set to 0
yhat = np.zeros(n)
yhat[-1] = 1

# score
roc_auc_score(y, yhat)
