import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
# gen some rvs

X1 = np.arange(0, 1, 0.01)

X2 = 3*X1 + 4                   # linearly dependant on X1 (increases w/ X1) 

X3 = -0.3*X1 + 4                # linearly dependant on X1 (decreases w/ X1)

X4 = np.random.uniform(0,1,100) # independant of X1
def covar(X1, X2):

    """𝑋1  and  𝑋2  are 1-D continous distributions of rvs"""

    if len(X1) != len(X2): raise("Number of samples must be same!")

    

    mean_X1 = np.mean(X1)

    mean_X2 = np.mean(X2)

    

    return np.sum( (X1-mean_X1) * (X2-mean_X2) ) / len(X1) 
# find respective covars

covarX1X2 = covar(X1,X2)

covarX1X3 = covar(X1,X3)

covarX1X4 = covar(X1,X4)



print(f"covar(X1,X2): {covarX1X2:.2f} \ncovar(X1,X3): {covarX1X3:.2f} \ncovar(X1,X4): {covarX1X4:.2f}")
fig, axarr = plt.subplots(1, 3)

fig.set_size_inches(15,4)



axarr[0].scatter(X1, X2)

axarr[0].title.set_text(f"X1-X2 -> positive covariance: {covarX1X2:.2f}")

axarr[0].set_xlabel("X1")

axarr[0].set_ylabel("X2")

axarr[0].legend(["x-axis: X1 y-axis: X2"])



axarr[1].scatter(X1, X3)

axarr[1].title.set_text(f"X1-X3 -> negative covariance: {covarX1X3:.2f}")

axarr[1].set_xlabel("X1")

axarr[1].set_ylabel("X3")

axarr[1].legend(["x-axis: X1 y-axis: X3"])



axarr[2].scatter(X1, X4)

axarr[2].title.set_text(f"X1-X4 -> zero covariance: {covarX1X4:.2f}")

axarr[2].set_xlabel("X1")

axarr[2].set_ylabel("X4")

axarr[2].legend(["x-axis: X1 y-axis: X4"])



plt.legend()

plt.show()
def pcc(X1, X2):

    """𝑋1  and  𝑋2  are 1-D continous distributions of rvs"""

    if len(X1) != len(X2): raise("Number of samples must be same!")    

    factor = np.std(X1) * np.std(X2)

    return covar(X1, X2) / factor
# find respective coavrs

pccX1X2 = pcc(X1,X2)

pccX1X3 = pcc(X1,X3)

pccX1X4 = pcc(X1,X4)



print(f"pcc(X1,X2): {pccX1X2:.2f} \npcc(X1,X3): {pccX1X3:.2f} \npcc(X1,X4): {pccX1X4:.2f}")
# X5 -> less linearly dependant (increases slowly with X1)

ERRORS = np.random.uniform(-0.3,0.3, len(X1))

X5 =  3*X1 + 4 + ERRORS



pccX1X5 = pcc(X1,X5)

print(f"pcc(X1,X5): {pccX1X5}")
fig, axarr = plt.subplots(1,2)

fig.set_size_inches(15,4)



axarr[0].scatter(X1, X2)

axarr[0].title.set_text(f"stronger +ve covariance \nPCC(X1-X2) = {pccX1X2:.2f}")

axarr[0].set_xlabel("X1")

axarr[0].set_ylabel("X2")

axarr[0].legend(["x-axis: X1 y-axis: X2"])



axarr[1].scatter(X1, X5)

axarr[1].title.set_text(f"weaker +ve covariance \nPCC(X1-X5) = {pccX1X5:.2f}")

axarr[1].set_xlabel("X1")

axarr[1].set_ylabel("X5")

axarr[1].legend(["x-axis: X1 y-axis: X5"])



plt.show()
def to_ranks(rvs):

    """ returns ranks instead of raw rvs """

    ordered_idxs = rvs.argsort()

    ranks = np.empty_like(ordered_idxs)

    ranks[ordered_idxs] = np.arange(len(rvs))

    return ranks



def srcc(X1, X2):

    """𝑋1  and  𝑋2  are 1-D continous distributions of rvs"""

    if len(X1) != len(X2): raise("Number of samples must be same!")  

        

    X1_to_ranks = to_ranks(X1)

    X2_to_ranks = to_ranks(X2)    

    

    return pcc(X1_to_ranks, X2_to_ranks)
# generate monotonic function

monotoneX2 = np.array([4,5,5.1,5.2,7.01,8.95])

monotoneX1 = np.arange(0, len(monotoneX2), 1)



plt.scatter(monotoneX1, monotoneX2)

plt.plot(monotoneX1, monotoneX2)

plt.title("Monotonic funtion")

plt.xlabel("monotoneX1")

plt.ylabel("monotoneX2")

plt.show()
print("="*80, f"\nPCC(monotoneX1, monotoneX2): \t{pcc(monotoneX1, monotoneX2)}"   , "\n"+"="*80)

print(f"SRCC(monotoneX1, monotoneX2): \t{srcc(monotoneX1, monotoneX2)}" , "\n"+"="*80)
# generate outlier data

feat1 = np.append(X1, np.random.uniform(0.8,1.0,10))

feat2 = np.append(X5, np.random.uniform(4.0,4.5,10))



# plot

plt.scatter(feat1, feat2)



plt.title("Features w/ outliers")

plt.xlabel("feat1")

plt.ylabel("feat2")

plt.show()
print("="*80, f"\nPCC(feat1, feat2): \t{pcc(feat1, feat2):.3f}"   , "\n"+"="*80)

print(f"SRCC(feat1, feat2): \t{srcc(feat1, feat2):.3f}" , "\n"+"="*80)
print(f"PCC(X1,X2): {pcc(X1, X2):.3f} \tSRCC(X1,X2): {srcc(X1, X2):.3f}")

print(f"PCC(X1,X5): {pcc(X1, X5):.3f} \tSRCC(X1,X5): {srcc(X1, X5):.3f}")