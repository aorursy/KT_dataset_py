import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

train = pd.read_csv("../input/train.csv") 
#test = pd.read_csv("../input/test.csv") 

#remove static data,
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)
        
#print(remove)
#for x in remove:
#    print(x,train[x].describe()) # yes these are all just 0s
train.drop(remove, axis=1, inplace=True)

#fit a rf to the data
from sklearn.ensemble import RandomForestClassifier

X = train.drop(['ID','TARGET'], axis=1)
y = train['TARGET']
clf = RandomForestClassifier().fit(X,y)
#find the important features
imp = clf.feature_importances_
best = sorted(imp, reverse=True)
for x in best[:10]:
    print (x, X.columns[np.where(imp == x)].values[0])
most_important = X.columns[np.argmax(imp)]
X[most_important].describe()
# 116 values in column var3 are -999999
# var3 is suspected to be the nationality of the customer
# -999999 would mean that the nationality of the customer is unknown
#train['var3'].hist()
train.loc[train.var3==-999999].shape
# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape
#train.var3.max()
#train.var3.min()
train.var3.hist(bins=100)

train[train.var3!=2].var3.hist(bins=100)
# var38 is important according to XGBOOST
# see https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping/files
# Also RFC thinks var38 is important
# see https://www.kaggle.com/tks0123456789/santander-customer-satisfaction/data-exploration/notebook
# so far I have not seen a guess what var38 may be about
train.var38.describe()
# How is var38 looking when customer is unhappy ?
train.loc[train['TARGET']==1, 'var38'].describe()
# Histogram for var 38 is not normal distributed
fig, ax = plt.subplots()
train.var38.hist(ax=ax, bins=1000, bottom=0.1)
ax.set_yscale('log')

# Histogram for var 38 is not normal distributed
fig, ax = plt.subplots()
train.loc[train['TARGET']==1, 'var38'].hist(ax=ax, bins=1000, bottom=0.1)
#ax.set_yscale('log')
# Histogram for var 38 is not normal distributed
fig, ax = plt.subplots()
train.loc[train['TARGET']==0, 'var38'].hist(ax=ax, bins=1000, bottom=0.1)
#ax.set_yscale('log')
train.var38.hist(bins=1000)
# where is the spike between 11 and 12  in the log plot ?
train.var38.map(np.log).mode()
# What are the most common values for var38 ?
train.var38.value_counts()
# what is we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()
# Look at the distribution
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100)
dist38 = train.loc[~np.isclose(train.var38, 117310.979016), 'var38']
fig, ax = plt.subplots()
d = dist38.map(np.log);
from scipy.stats import norm
mu, std = norm.fit(d);
plt.hist(d, bins=25, normed=True, alpha=0.6, color='g')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'b', linewidth=2)
# Above plot suggest we split up var38 into two variables
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0
#Check for nan's
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())
train['var15'].describe()
#Looks more normal, plot the histogram
train['var15'].hist(bins=100)
fig, ax = plt.subplots()
#d = train['var15'].map(np.log);
d = train['var15']
from scipy.stats import norm
mu, std = norm.fit(d);
n, bins, patches = plt.hist(d, bins=100, normed=True, alpha=0.6, color='g')
p = norm.pdf(bins, mu, std)
plt.plot(bins, p, 'b', linewidth=2)
d = train['var15']#.map(np.log)
from scipy.stats import expon
floc_d = 21
mu, std = expon.fit(d, floc=floc_d);
plt.hist(d, bins=100, normed=True, alpha=0.6, color='g')

xmin, xmax = plt.xlim()
print (xmin, xmax)
#xmin, xmax = plt.xlim(d)
x = np.linspace(xmin, xmax, 100)
p = expon.pdf(x, mu, std)
plt.plot(x, p, 'b', linewidth=2)
# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
plt.title('Unhappy customers are slightly older')
plt.show()
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "var38", "var15") \
   .add_legend()
sns.FacetGrid(train, hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120]) # Age must be positive ;-)
# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "logvar38", "var15") \
   .add_legend()
plt.ylim([0,120])
# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6) \
   .map(sns.kdeplot, "var15") \
   .add_legend()
X = train.iloc[:,:-1]
y = train.TARGET

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# First select features based on chi2 and f_classif
p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)
# Make a dataframe with the selected features and the target variable
X_sel = train[features+['TARGET']]
# var38 (important for XGB and RFC is not selected but var36 is. Let's explore
X_sel['var36'].value_counts()
# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "var36") \
   .add_legend()
# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend()

sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10) \
   .map(plt.scatter, "var36", "logvar38") \
   .add_legend()
# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6) \
   .map(sns.kdeplot, "logvar38") \
   .add_legend()
sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde")
train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6))
# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET")
features
radviz(train[features], "TARGET")
sns.pairplot(train[features], hue="TARGET", size=2, diag_kind="kde")