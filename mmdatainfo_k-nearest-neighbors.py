# Scikit-learn

from sklearn import datasets

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.metrics import classification_report, mean_squared_error

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA

# other libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
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
scale = StandardScaler(with_mean=True,with_std=True);

Xo = scale.fit_transform(df.drop(["target"],axis=1).values);
pca = PCA(n_components=0.99)# or set n_components="mle"

X = pca.fit_transform(Xo)

print("Nr. of features after PCA = {} (input = {})".format(X.shape[1],Xo.shape[1]))
# encode target values (is not necessary for IRIS but still:-)

y = LabelEncoder().fit_transform(df["target"].values);

# Split 2/3 to 1/3 train to test respectively

[X_train,X_test,y_train,y_test] = train_test_split(X,y,train_size = 0.67,test_size = 0.33,

                                                   stratify=y,random_state=123);
model = KNeighborsClassifier(algorithm="auto");

parameters = {"n_neighbors":[1,3,5],

              "weights":["uniform","distance"]}

model_optim = GridSearchCV(model, parameters, cv=5,scoring="accuracy");
model_optim.fit(X_train,y_train)
model_optim.best_estimator_
for (i,x,y) in zip(["Train","Test"],[X_train,X_test],[y_train,y_test]):

    print("Classification kNN",i," report:\n",classification_report(y,model_optim.predict(x)))
for i in ["most_frequent","uniform"]:

    dummy = DummyClassifier(strategy=i).fit(X_train,y_train);

    print("Classification ",i," test report:",classification_report(y_test,dummy.predict(X_test)))
# house = datasets.fetch_california_housing()

# df = pd.DataFrame(house.data,columns=house.feature_names)

# df = df.assign(target=house.target)
df = pd.read_csv("../input/test-data/california_housing.csv").drop(columns=["Unnamed: 0"],errors='ignore')
# Compute selected stats

dfinfo = pd.DataFrame(df.dtypes,columns=["dtypes"])

for (m,n) in zip([df.count(),df.isna().sum()],["count","isna"]):

    dfinfo = dfinfo.merge(pd.DataFrame(m,columns=[n]),right_index=True,left_index=True,how="inner");

# Add to `describe` output

dfinfo.T.append(df.describe())
plt.figure(figsize=(9,4))

for (i,v) in enumerate(df.columns):

    plt.subplot(2,5,i+1);

    plt.hist(df.iloc[:,i],50,density=True)

    plt.legend([df.columns[i]],fontsize=6);
df.corr().round(2).style.background_gradient(cmap="viridis")
df = df.drop(["AveOccup"],axis=1)
X = StandardScaler().fit_transform(df.drop("target",axis=1).values);

y = df.target.values
X = PCA(n_components="mle").fit_transform(X)

print("Nr. of features after reduction = {} (input = {})".format(X.shape[1],df.shape[1]))
[X_train,X_test,y_train,y_test] = train_test_split(X,y,train_size=0.67,test_size=0.33,

                                                   random_state=123);
knn = KNeighborsRegressor();

parameters = {"n_neighbors":[1,3,5,7,9],

              "weights":["uniform","distance"]}

knn_reg = GridSearchCV(knn, parameters, cv=5, scoring="neg_mean_squared_error");
knn_reg.fit(X_train,y_train)
knn_reg.best_estimator_
print("Regression kNN (test) RMSE \t= {:.0f} *1000$".format(

    100*np.sqrt(mean_squared_error(knn_reg.predict(X_test),y_test))))
for i in ["mean","median"]:

    dummy = DummyRegressor(strategy=i).fit(X_train,y_train);

    print("Regression ",i,"(test) RMSE \t= {:.0f} *1000$".format(

        100*np.sqrt(mean_squared_error(y_test,dummy.predict(X_test)))))