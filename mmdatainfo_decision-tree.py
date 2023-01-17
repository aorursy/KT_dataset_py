# Scikit-learn

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.dummy import DummyRegressor, DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn import datasets

# Other libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
# Use entropy equation

def entropy(p,*args):

    H = p*np.log2(p);

    for i in args:

        H += 0 if i == 0 else i*np.log2(i)

    return -H
machine1 = entropy(0.25,0.25,0.25,0.25)

machine2 = entropy(0.5,0.125,0.125,0.25)

print("Machine 1 entropy = ",machine1)

print("Machine 2 entropy = ",machine2)
# Full data set with two classes

print("Entropy of the data set: {:.5f}".format(entropy(24/49,25/49)))

# Orange box

print("Enropy within the orange box: {:.5f} = lower than whole data set as we have ordered data".format(

    entropy(4/25,21/25)))

# Blue box

print("Enropy within the blue box: {:.5f} = even less disorder because only 3 diamonds".format(

    entropy(21/24,3/24)))

print("= higher probability to predict random selection correctly")

# Entropy of the partition

print("Entropy after parition: {:.5f}".format(25/49*entropy(4/25,21/25) + 24/49*entropy(21/24,3/24)))

# Information gain

print("Information gain (=maximized in decision tree) after paritioning to two color-based boxes: {:.5f}".format(

      entropy(24/49,25/49) - (25/49*entropy(4/25,21/25) + 24/49*entropy(21/24,3/24))))
iris = datasets.load_iris()

# This is extra step that can be omitted but Pandas DataFrame contains some powerfull features

df = pd.DataFrame(iris.data,columns=iris.feature_names)

df = df.assign(target=iris.target)
# Compute selected stats

dfinfo = pd.DataFrame(df.dtypes,columns=["dtypes"])

for (m,n) in zip([df.count(),df.isna().sum()],["count","isna"]):

    dfinfo = dfinfo.merge(pd.DataFrame(m,columns=[n]),right_index=True,left_index=True,how="inner");

# Add to `describe` output

dfinfo.T.append(df.describe())
plt.figure(figsize=(9,2))

for (i,v) in enumerate(df.columns):

    plt.subplot(1,df.shape[1],i+1);

    plt.hist(df.iloc[:,i],bins="sqrt")

    plt.title(df.columns[i],fontsize=9);
df.corr().round(2).style.background_gradient(cmap="viridis")
X = df.drop("target",axis=1).values

y = df["target"].values;
dt = DecisionTreeClassifier(criterion="entropy");

rf = RandomForestClassifier(criterion="entropy",n_estimators=300);
# check how split affects the score

for i in range(0,5):

    Xtr, Xte, ytr, yte = train_test_split(X, y, 

                            train_size = 0.67, test_size =0.33, stratify=y,random_state=i*10);

    dt.fit(Xtr,ytr);

    rf.fit(Xtr,ytr);

    print("Accuracy score 1 vs random forest ensamble (random state = {}):\t {:.2f} vs {:.2f}".format(

             i*10,accuracy_score(yte,dt.predict(Xte)),

             accuracy_score(yte,rf.predict(Xte))))
df = pd.read_csv("../input/test-data/california_housing.csv").drop(columns=["Unnamed: 0"],errors='ignore')
# Compute selected stats

dfinfo = pd.DataFrame(df.dtypes,columns=["dtypes"])

for (m,n) in zip([df.count(),df.isna().sum()],["count","isna"]):

    dfinfo = dfinfo.merge(pd.DataFrame(m,columns=[n]),right_index=True,left_index=True,how="inner");

# Add to `describe` output

dfinfo.T.append(df.describe())
df.corr().round(2).style.background_gradient(cmap="viridis")
plt.figure(figsize=(9,4))

for (i,v) in enumerate(df.columns):

    plt.subplot(2,5,i+1);

    plt.hist(df.iloc[:,i],50,density=True)

    plt.legend([df.columns[i]],fontsize=6);
X = df.drop(["target","AveOccup"],axis=1).values;

y = df.target.values
[X_train,X_test,y_train,y_test] = train_test_split(X,y,train_size=0.67,test_size=0.33,

                                                   random_state=123);
dt = DecisionTreeRegressor();

rf = RandomForestRegressor(n_estimators=300);

du = DummyRegressor(strategy="mean")
for i in [dt,rf,du]:

    i.fit(X_train,y_train)
for (n,m) in zip(["simple tree ","random forest","dummy / mean"],[dt,rf,du]):

    print("RMSE ",n," train =\t {:.1f} *1000$".format(

        100*np.sqrt(mean_squared_error(y_test,m.predict(X_test)))))