import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

NUM_SAMPLES = 100000 

def randn_skew_fast(N, alpha=0.0, loc=0.0, scale=1.0): 
    sigma = alpha/np.sqrt(1.0 + alpha**2) 
    u0 = np.random.randn(N) 
    v = np.random.randn(N) 
    u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale 
    u1[u0 < 0] *= -1 
    u1 = u1 + loc 
    return u1 


    
plt.subplots(figsize=(12,4))
p1 = randn_skew_fast(NUM_SAMPLES, 2,10)
sns.distplot(p1)
p2 = randn_skew_fast(NUM_SAMPLES, 3,10)
sns.distplot(p2)
p3 = randn_skew_fast(NUM_SAMPLES, 5,10)
sns.distplot(p3)
p = randn_skew_fast(NUM_SAMPLES, 0,10)
sns.distplot(p)
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
aslist = axes.ravel()
stats.probplot(p1, plot=aslist[0])
stats.probplot(p2, plot=aslist[1])
stats.probplot(p3, plot=aslist[2])
plt.show()
sqrt_p1 = np.sqrt(p1-6)
sqrt_p2 = np.sqrt(p2-8)
sqrt_p3 = np.sqrt(p3-9)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
aslist = axes.ravel()
stats.probplot(sqrt_p1, plot=aslist[0])
stats.probplot(sqrt_p2, plot=aslist[1])
stats.probplot(sqrt_p3, plot=aslist[2])
plt.show()
log_p1 = np.log(p1-3)
log_p2 = np.log(p2-7)
log_p3 = np.log(p3-8)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
aslist = axes.ravel()
stats.probplot(log_p1, plot=aslist[0])
stats.probplot(log_p2, plot=aslist[1])
stats.probplot(log_p3, plot=aslist[2])
plt.show()
sqrt_p1 = 1/p1
sqrt_p2 = 1/p2
sqrt_p3 = 1/p3

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
aslist = axes.ravel()
stats.probplot(sqrt_p1, plot=aslist[0])
stats.probplot(sqrt_p2, plot=aslist[1])
stats.probplot(sqrt_p3, plot=aslist[2])
plt.show()
plt.subplots(figsize=(12,4))
p4 = randn_skew_fast(NUM_SAMPLES, -2,10)
sns.distplot(p4)
p5 = randn_skew_fast(NUM_SAMPLES, -3,10)
sns.distplot(p5)
p6 = randn_skew_fast(NUM_SAMPLES, -5,10)
sns.distplot(p6)
p = randn_skew_fast(NUM_SAMPLES, 0,10)
sns.distplot(p)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
aslist = axes.ravel()
stats.probplot(p4, plot=aslist[0])
stats.probplot(p5, plot=aslist[1])
stats.probplot(p6, plot=aslist[2])
plt.show()
log_p4 = np.log(13-p4)
log_p5 = np.log(13-p5)
log_p6 = np.log(13-p6)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
aslist = axes.ravel()
stats.probplot(log_p4, plot=aslist[0])
stats.probplot(log_p5, plot=aslist[1])
stats.probplot(log_p6, plot=aslist[2])
plt.show()
