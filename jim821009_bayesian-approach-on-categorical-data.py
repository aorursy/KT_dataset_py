import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
%matplotlib inline
df = pd.read_excel("Dispatchers_Background_Data-1.xls")[:443]
disorder = df[df["Diagnosed_Sleep_disorder"] == 1.0]
disorder.shape
disorder.describe()
NoDisorder = df[df["Diagnosed_Sleep_disorder"] == 2.0]
NoDisorder.shape
NoDisorder.describe()
plt.hist(df["Avg_Work_Hrs_Week"])
plt.hist(NoDisorder["Avg_Work_Hrs_Week"])
plt.show()
plt.hist(disorder["Avg_Work_Hrs_Week"])
plt.show()
# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(df)
ser1 = df["Marital_Status"]
ser2 = df["Childrendependents"]
np.multiply(ser1, ser2)
df_target = df[["Job_pressure", "Marital_Status", "Childrendependents"]]
df_target.insert(3, "Marital*Children",np.multiply(ser1, ser2) , True) 
df_target.dropna(axis=0, inplace=True)
def helperRank(x):
    '''
    x: numpy array
    '''
    res = []
    x_s = np.sort(np.unique(x))
    for i in range(len(x)):
        item = x.iloc[i]
        if item in x_s:
            res.append(list(x_s).index(item))
    return res    
from numpy.linalg import inv
from numpy.linalg import cholesky
from numpy.random import uniform as runif
from scipy.stats import uniform
from scipy.stats import norm


y = df_target["Job_pressure"].astype(int).reset_index(drop=True)
X = df_target[["Marital_Status", "Childrendependents", "Marital*Children"]].reset_index(drop=True)
ranks = helperRank(y)
urank = np.sort(np.unique(ranks))
n, p = X.shape[0], X.shape[1]
ixx = inv(X.T @ X)
V = ixx * (n/(n+1))
cholV = cholesky(V)

### setup
beta = np.zeros(p)
z = scipy.stats.norm.ppf(y.rank(method="first")/(n+1))
g = np.repeat(0, len(urank)-1)
K = len(urank)

betas = np.zeros(shape=(10000, 3))
Zs = np.zeros(shape=(10000, n))
ac = 0
mu = np.zeros(K-1)
sigma = np.repeat(100, K-1)
S = 10000
for s in range(S):
    #update g
    for k in range(1, 4): # 1, 2, 3
        a = max(z[y==k]) # 1, 2, 3
        b = min(z[y==k+1]) # 2, 3, 4
        low = norm.cdf((a-mu[k-1])/sigma[k-1] )
        high = norm.cdf((b-mu[k-1])/sigma[k-1] )
        u = uniform.rvs(loc=low, scale=high-low)
        g[k-1] = mu[k-1] + sigma[k-1] * norm.ppf(u)
    
    # update beta
    E = V @ ( X.T @ z)
    beta = cholV @ norm.rvs(size=3) + E
    
    # update z
    for i in range(440):
        ez = beta.T @ X.iloc[i]
        a = -np.infty
        b = np.infty
        if y.iloc[i]-1 >= 1:
            a = g[y.iloc[i]-1-1]
        if y.iloc[i] < 4:
            b = g[y.iloc[i]-1]
        u = uniform.rvs(loc=norm.cdf(a-ez), scale=norm.cdf(b-ez)-norm.cdf(a-ez))
        z[i] = ez + norm.ppf(u)
    Zs[s] = z
    betas[s] = beta
betas.shape
plt.hist(betas[500:, 2], bins=40)
plt.show()
plt.hist(betas[500:, 1], bins=40)
plt.show()
plt.hist(betas[500:, 0], bins=40)
plt.show()
Zs.shape
y_values_masked = np.ma.masked_where(y == 1 , y)
area3 = np.ma.masked_where(y != 3, y)
area1 = np.ma.masked_where(y != 1, y)
area2 = np.ma.masked_where(y != 2, y)
area4 = np.ma.masked_where(y != 4, y)


plt.scatter(X["Childrendependents"], y=Zs[1000], s=area1 * 10, label="No Stress", c='g')
plt.scatter(X["Childrendependents"], y=Zs[1000], s=area2 * 10, label="Little Stress", c='b')
plt.scatter(X["Childrendependents"], y=Zs[1000], s=area3 * 10, label="Stressful", c='y')
plt.scatter(X["Childrendependents"], y=Zs[1000], s=area4 * 7, label='Very Stressful', c='r')
plt.legend()
plt.show()
area3 = np.ma.masked_where(y != 3, y)
area1 = np.ma.masked_where(y != 1, y)
area2 = np.ma.masked_where(y != 2, y)
area4 = np.ma.masked_where(y != 4, y)

plt.scatter(X["Marital_Status"], y=Zs[1000], s=area1 * 10, label="No Stress", c='g')
plt.scatter(X["Marital_Status"], y=Zs[1000], s=area2 * 10, label="Little Stress", c='b')
plt.scatter(X["Marital_Status"], y=Zs[1000], s=area3 * 10, label="Stressful", c='y')
plt.scatter(X["Marital_Status"], y=Zs[1000], s=area4 * 7, label='Very Stressful', c='r')
plt.legend()
plt.show()
area3 = np.ma.masked_where(y != 3, y)
area1 = np.ma.masked_where(y != 1, y)
area2 = np.ma.masked_where(y != 2, y)
area4 = np.ma.masked_where(y != 4, y)

plt.scatter(X["Marital*Children"], y=Zs[1000], s=area1 * 10, label="No Stress", c='g')
plt.scatter(X["Marital*Children"], y=Zs[1000], s=area2 * 10, label="Little Stress", c='b')
plt.scatter(X["Marital*Children"], y=Zs[1000], s=area3 * 10, label="Stressful", c='y')
plt.scatter(X["Marital*Children"], y=Zs[1000], s=area4 * 7, label='Very Stressful', c='r')
plt.legend()
plt.show()