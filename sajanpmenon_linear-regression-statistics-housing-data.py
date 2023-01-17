import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 

from scipy.stats import zscore

%matplotlib inline


df = pd.read_csv("../input/housing-linear-regression/Housing.csv")

df.head()
df.head()
df.sample(5)
df.shape
df.info()
df.get_dtype_counts()
df.describe().T
sns.boxplot(y=df['area'])

plt.show()
sns.distplot(df['area'])

plt.show
q3=df['area'].quantile(0.75)

q1=df['area'].quantile(0.25)

iqr=q3-q1

df=df[(df['area']>q1-1.5*iqr)  &   (df['area']<=q3+1.5*iqr)]
sns.boxplot(y=df['area'])

plt.show()
sns.distplot(df['area'])

plt.show
sns.boxplot(y=df['price'])

plt.show()
sns.distplot(df['price'])

plt.show
df['lprice']=np.log(df['price'])
df.head()
sns.boxplot(y=df['lprice'])

plt.show()
sns.distplot(df['lprice'])

plt.show
sns.scatterplot(x='area',y='price',data=df)

plt.show()
sns.lmplot(x='area',y='price',hue='mainroad',col='mainroad',data=df)

plt.show()
sns.barplot(x='furnishingstatus',y='price',data=df)

plt.show()
sns.violinplot(x='furnishingstatus', y='price', hue='mainroad',

               split=True, data=df)

plt.show()
df.info()
df['bedrooms'].value_counts()
df['parking'].value_counts()
#independent features 'bathrooms','stories','parking' have to be converted to categorical as the range in them are vry less.

df[['bathrooms','stories','parking']]=df[['bathrooms','stories','parking']].astype('object')
df.info()
cols=list(df.select_dtypes('object'))

cols
df=pd.get_dummies(df,columns=cols,drop_first=True)
df.info()
y=df['lprice']

x=df.drop(['lprice','price'],axis=1)
X=x.assign(const=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=1)
import statsmodels.api as sm

ols_train=sm.OLS(y_train,x_train).fit()





import warnings as warnings

warnings.filterwarnings('ignore')
ols_train.summary()
ols_test=sm.OLS(y_test,x_test).fit()

ols_test.summary()
plt.figure(figsize=(15,10))

corr = x_train.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr, mask=mask, linewidth=0.5, annot=True)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=[variance_inflation_factor(X.values,i) for i in range (X.shape[1])]

#pd.DataFrame({'vif':vif,'index':X.columns}).T

pd.DataFrame(vif,index=X.columns).T
# to check the possibility of interaction we will apply polynomial feature engineering:
from sklearn.preprocessing import PolynomialFeatures



#X = X.drop('const', axis=1)



pf = PolynomialFeatures()

Xp = pf.fit_transform(X)

cols = pf.get_feature_names(X.columns)



Xp = pd.DataFrame(Xp, columns=cols)

colsxp = Xp.columns
X.shape
Xp.head()
ys=y.values
xp_train,xp_test,ys_train,ys_test=train_test_split(Xp,ys,test_size=0.33,random_state=1)
ols_p=sm.OLS(ys_train,xp_train).fit()

ols_p.summary()
ols_p=sm.OLS(ys_test,xp_test).fit()

ols_p.summary()
pmax=1

cols = list(Xp.columns)

X_1 = Xp.copy()



while (len(cols)>0):

    p = []

    X_1 = X_1[cols]

    model = sm.OLS(ys, X_1).fit()

    p = model.pvalues

    pmax = max(p)

    

    if (pmax>0.05):

        cols = list(p.drop(p[p==pmax].index).index)

        #print(cols)

    else:

        break

        

selected = cols

print(selected)

print(len(selected))
Xps=Xp[selected]

Xps.shape
yps=ys.copy()
xps_train,xps_test,yps_train,yps_test=train_test_split(Xps,yps,test_size=0.33,random_state=1)
ols_xps=sm.OLS(yps_train,xps_train).fit()

ols_xps.summary()
ols_xps=sm.OLS(yps_test,xps_test).fit()

ols_xps.summary()