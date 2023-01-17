!pip install copulae
!pip install scipy==1.2 --upgrade
import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import copulae
os.listdir("../input")
data = pd.read_excel("../input/CAC 40.xlsx")
data.head()
data_clot = data.set_index(["JOUR","NOM"],append=False)["CLOT"]

data_clot.head()
data_clot = data_clot.unstack(level="NOM")

data_clot.head()
ca_clot = data_clot["Credit Agricole"]

bnp_clot = data_clot["Bnp Paribas"]
ca_returns = np.log(ca_clot/ca_clot.shift()).dropna()

bnp_returns = np.log(bnp_clot/bnp_clot.shift()).dropna()
plt.plot(ca_returns)

plt.title("Rendements du Crédit Agricole")

plt.xticks(rotation=45)

plt.show()



plt.plot(bnp_returns)

plt.title("Rendements de BNP Paribas")

plt.xticks(rotation=45)

plt.show()
h= sns.jointplot(x= ca_returns,y=bnp_returns,kind='scatter')

h.set_axis_labels('CA', 'BNP', fontsize=16);



h= sns.jointplot(x= ca_returns,y=bnp_returns,kind='kde',xlim=[-0.05,0.05],ylim=[-0.05,0.05])

h.set_axis_labels('CA', 'BNP', fontsize=16)

plt.title("Zoom")

plt.show()
from statsmodels.distributions.empirical_distribution import ECDF



ca_cdf = ECDF(ca_returns)

bnp_cdf = ECDF(bnp_returns)
h= sns.jointplot(x= ca_cdf(ca_returns),y=bnp_cdf(bnp_returns),kind='scatter')

h.set_axis_labels('CA', 'BNP', fontsize=16)



plt.show()
from scipy import stats



returns = pd.concat([ca_returns,bnp_returns],axis=1)

mv_norm = stats.multivariate_normal(mean= np.mean(returns,axis=0),cov= returns.corr())

mv_norm_spl = mv_norm.rvs(returns.shape[0]*100)
h= sns.jointplot(x= mv_norm_spl[:,0],y= mv_norm_spl[:,1],kind='hex')

h.set_axis_labels('CA', 'BNP', fontsize=16);



plt.show()
norm = stats.norm()

mv_norm_spl_unif = norm.cdf(mv_norm_spl)



h = sns.jointplot(mv_norm_spl_unif[:, 0], mv_norm_spl_unif[:, 1], kind='hex')

h.set_axis_labels('CA', 'BNP', fontsize=16)



plt.show()
c1 = copulae.archimedean.GumbelCopula(dim=2)

c1.fit(data=returns)

c1.summary()
c1_cdf = c1.random(n=returns.shape[0]*100)

h = sns.jointplot(c1_cdf[:, 0],c1_cdf[:, 1], kind='hex')

h.set_axis_labels('CA', 'BNP', fontsize=16);
from scipy.stats import kurtosis, skew



pf_reel = returns.sum(axis=1)

plt.hist(pf_reel,bins=50)

plt.xlim([-0.4,0.2])

plt.title("Portefeuille réel")

plt.show()



print("Skewness: {} \nKurtosis: {}".format(skew(pf_reel),kurtosis(pf_reel)))
m1 = stats.norm(loc=ca_returns.mean(),scale=ca_returns.std())

m2 = stats.norm(loc=bnp_returns.mean(),scale=bnp_returns.std())



ca_returns_m1 = m1.ppf(c1_cdf[:, 0])

bnp_returns_m2 = m2.ppf(c1_cdf[:, 1])



h = sns.jointplot(ca_returns_m1,bnp_returns_m2, kind='hex')

h.set_axis_labels('CA', 'BNP', fontsize=16)



plt.show()
returns_c1 = pd.DataFrame(np.vstack([ca_returns_m1,bnp_returns_m2]).T)

pf_c1 = returns_c1.sum(axis=1)

plt.hist(pf_c1,bins=50)

plt.xlim([-0.4,0.2])

plt.title("Portefeuille simulé")

plt.show()





print("Skewness: {} \nKurtosis: {}".format(skew(pf_c1),kurtosis(pf_c1)))
def vars(percentile,pf_c):

    var_reel = -1 * np.percentile(pf_reel, percentile)

    var_c = -1 * np.percentile(pf_c, percentile)

    return var_reel,var_c
VaRs = pd.DataFrame([vars(i,pf_c1) for i in np.linspace(0.5,10,(10/0.5))],

                    columns=['réel','simulé'],

                    index= 100 - np.linspace(0.5,10,(10/0.5)))



plt.plot(VaRs['réel'],label='Portefeuille réel')

plt.plot(VaRs['simulé'],label='Portefeuille simulé')

plt.title('VaR selon le risque (%)')

plt.xticks(100 - np.linspace(0.5,10,(10/0.5)))

plt.xticks(rotation=45)

plt.legend()

plt.show()
c2 = copulae.archimedean.ClaytonCopula(dim=2)

c2.fit(data=returns)

c2.summary()
c2_cdf = c2.random(n=returns.shape[0]*100)

h = sns.jointplot(c2_cdf[:, 0],c2_cdf[:, 1], kind='hex')

h.set_axis_labels('CA', 'BNP', fontsize=16);
l1 = stats.t.fit(ca_returns)

l2 = stats.t.fit(bnp_returns)



l1 = stats.t(l1[0],l1[1],l1[2])

l2 = stats.t(l2[0],l2[1],l2[2])



ca_returns_l1 = l1.ppf(c2_cdf[:, 0])

bnp_returns_l2 = l2.ppf(c2_cdf[:, 1])



h = sns.jointplot(ca_returns_l1,bnp_returns_l2, kind='kde')

h.set_axis_labels('CA', 'BNP', fontsize=16)

h.ax_marg_x.set_xlim(-0.1,0.1)

h.ax_marg_y.set_ylim(-0.1,0.1)

plt.show()
returns_c2 = pd.DataFrame(np.vstack([ca_returns_l1,bnp_returns_l2]).T)

pf_c2 = returns_c2.sum(axis=1)

plt.hist(pf_c2,bins=50)

plt.xlim([-0.4,0.2])

plt.title("Portefeuille simulé")

plt.show()





print("Skewness: {} \nKurtosis: {}".format(skew(pf_c2),kurtosis(pf_c2)))
VaRs = pd.DataFrame([vars(i,pf_c2) for i in np.linspace(0.5,10,(10/0.5))],

                    columns=['réel','simulé'],

                    index= 100 - np.linspace(0.5,10,(10/0.5)))



plt.plot(VaRs['réel'],label='Portefeuille réel')

plt.plot(VaRs['simulé'],label='Portefeuille simulé')

plt.title('VaR selon le risque (%)')

plt.xticks(100 - np.linspace(0.5,10,(10/0.5)))

plt.xticks(rotation=45)

plt.legend()

plt.show()