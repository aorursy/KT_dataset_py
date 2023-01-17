import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns


from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib inline
plt.style.use('seaborn-white')
adv = pd.read_csv('../input/Advertising.csv', usecols=[1,2,3,4])
adv.info()
adv.describe()
adv.head()
sns.regplot(adv.TV, adv.Sales, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(0,310)
plt.ylim(ymin=0);
estTV = smf.ols('Sales ~ TV', adv).fit()
estTV.summary().tables[1]
estTV = smf.ols('Sales ~ TV', adv).fit()
estTV.summary().tables[2]
estRD = smf.ols('Sales ~ Radio', adv).fit()
estRD.summary().tables[1]
estRD = smf.ols('Sales ~ Radio', adv).fit()
estRD.summary().tables[2]
estNP = smf.ols('Sales ~ Newspaper', adv).fit()
estNP.summary().tables[1]
estNP = smf.ols('Sales ~ Newspaper', adv).fit()
estNP.summary().tables[2]
estAll = smf.ols('Sales ~ TV + Radio + Newspaper', adv).fit()
estAll.summary()
adv.corr()


auto = pd.read_csv('../input/Auto.csv', na_values='?').dropna();
auto.head()
auto.info()
auto.describe()
sns.regplot(auto.acceleration, auto.mpg, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.show()
sns.regplot(auto.displacement, auto.mpg, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.show()
sns.regplot(auto.horsepower, auto.mpg, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.show()
s = sns.PairGrid(auto)
s.map(plt.scatter)
sns.regplot(auto.horsepower, auto.mpg, order=1, ci=None)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
auto = pd.read_csv('../input/Auto.csv', na_values='?').dropna();
auto.head()
auto.info()
auto.describe()
auto.corr()
f, ax = plt.subplots(figsize=(10, 8))
corr = auto.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


