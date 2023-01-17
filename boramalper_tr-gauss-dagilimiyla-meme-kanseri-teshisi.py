import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from scipy.stats import multivariate_normal



# Any results you write to the current directory are saved as output.

data_csv = pd.read_csv("../input/data.csv")

data_csv
# Split data into training and test subset.

Dtrn, Dtst = train_test_split(data_csv, test_size=0.4)
# We are only intersted in worst case.

# (I have tried mean earlier, did not perform as well.)

Xtrn = Dtrn.iloc[:, 22:32]

Ytrn = Dtrn.iloc[:, 1]



Xtst = Dtst.iloc[:, 22:32]

Ytst = Dtst.iloc[:, 1]



D = 10
## Parameter Estimation



# Means

BenM = np.mean(Xtrn[Ytrn == 'B'])

MalM = np.mean(Xtrn[Ytrn == 'M'])



# Covariance Matrices

eps = 0.001

BenC = np.cov(Xtrn[Ytrn == 'B'], rowvar=False) + eps * np.eye(D)

MalC = np.cov(Xtrn[Ytrn == 'M'], rowvar=False) + eps * np.eye(D)

Probs = np.zeros([Xtst.shape[0], 2])



Probs[:, 0] = multivariate_normal.pdf(Xtst, BenM, BenC) * (357 / (357 + 212))

Probs[:, 1] = multivariate_normal.pdf(Xtst, MalM, MalC) * (212 / (357 + 212))



Ypreds = ['B' if i == 0 else 'M' for i in list(Probs.argmax(1))]
print("Doğru Teşhisler :", sum(Ypreds == Ytst))

print("Yanlış Tehşisler:", len(Ypreds) - sum(Ypreds == Ytst))

print("Doğruluk Payı   : {:.2f}%".format(sum(Ypreds == Ytst) / len(Ypreds) * 100))