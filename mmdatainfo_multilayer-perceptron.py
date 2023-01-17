# Scikit-learn

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.linear_model import LinearRegression, RANSACRegressor

from sklearn import datasets

from sklearn.metrics import mean_squared_error, r2_score

# Other libraries

import numpy as np

import pandas as pd

import scipy.stats as sps

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# Use vector drawing inside jupyter notebook and Set matplotlib default axis font size

%config InlineBackend.figure_format = "svg"

plt.rcParams.update({'font.size': 8})
X = np.array([[0.0, 0.0],[1.0, 0.0],[1.0, 1.0],[0.0, 1.0]]);

y = np.array([0, 1, 0, 1]);

# Plot the data

plt.figure(figsize=(1.5,1.5))

plt.scatter(X[:,0],X[:,1],c=y,marker="s")

plt.axis("equal");

plt.xlabel("feature 1");plt.ylabel("feature 2");

plt.title("Learning input");
clf = MLPClassifier(hidden_layer_sizes=(2,2), # theoritically (2) could be sufficient

                    solver="lbfgs",random_state=123,

                    activation="tanh",

                    tol=1e-9)

clf.fit(X,y);

clf.predict(X)
# Evaluation grid

xi,yi = np.meshgrid(np.arange(-0.5,1.5,step=0.1),

                    np.arange(-0.5,1.5,step=0.1));

X_test = np.concatenate((xi.reshape(xi.size,1),yi.reshape(yi.size,1)),axis=1)

y_test = clf.predict(X_test);
# Plot original + grid

plt.figure(figsize=(1.5,1.5))

plt.scatter(X[:,0],X[:,1],c=y,marker="s")

plt.scatter(X_test[:,0],X_test[:,1],c=y_test,marker=".")

plt.axis("equal");

plt.xlabel("feature 1");plt.ylabel("feature 2");  

plt.title("MLP output");
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

    plt.hist(df.iloc[:,i],40,density=True)

    plt.legend([df.columns[i]],fontsize=6);
# Scale to mean=0, std=1

# Here, the data is scaled prior train-test split. 

# In real applications, first split and scale afterwards, to simulate real-world scenario where we do not have the test set!

scx, scy = StandardScaler(), StandardScaler();

X = scx.fit_transform(df.drop(["target"],axis=1).values);

y = scy.fit_transform(df.target.values.reshape(df.shape[0],1)).flatten()



# Prepare a function for convenient label inverse-transformation

def scyit(yin):

    return scy.inverse_transform(yin.reshape(yin.size,1)).flatten();
[X_train,X_test,y_train,y_test] = train_test_split(X,y,train_size=0.67,test_size=0.33,

                                                   random_state=123);
mlp = MLPRegressor(hidden_layer_sizes=(10,5),solver="lbfgs",

                   tol=1e-6,activation="logistic")

parameters = {"alpha":10.0**-np.arange(1, 7)};

mlp_cv = GridSearchCV(mlp,param_grid=parameters,scoring="neg_mean_squared_error",

                      cv=10,return_train_score=True).fit(X_train,y_train)
mlp_cv.best_estimator_
dfout = pd.DataFrame(mlp_cv.cv_results_)[["param_alpha","mean_train_score","std_train_score"]];

for i in ["mean_train_score","std_train_score"]:

    dfout[i] =  dfout[i].round(3);

dfout.T
mlp_es = MLPRegressor(hidden_layer_sizes=(10,5),solver="adam",

                    tol=1e-6,activation="logistic",early_stopping=True,

                    validation_fraction=0.2,max_iter=2000).fit(X_train,y_train)
mlp_es_i = [];

for i in range(0,10):

    mlp_es.fit(X_train,y_train);

    mlp_es_i = np.append(mlp_es_i,

                mean_squared_error(y_train,mlp_es.predict(X_train)))
print("Scaled mean MSE = {:.3f}+/-{:.3f}".format(

        np.mean(mlp_es_i),np.std(mlp_es_i)))

print("All scaled MSE = {}".format(np.round(mlp_es_i,3)))
for (t,x,y) in zip(["Train","Test"],[X_train,X_test],[y_train,y_test]):

    for (n,m) in zip(["CV","ES"],[mlp_cv,mlp_es]):

        print(n," RMSE ",t," =\t {:.1f} *1000$".format(

            100*np.sqrt(mean_squared_error(scyit(y),scyit(m.predict(x))))))
plt.figure(figsize=(5,4.5))

i = 1;

for (t,x,y) in zip(["Train","Test"],[X_train,X_test],[y_train,y_test]):

    for (n,m) in zip(["CV","ES"],[mlp_cv,mlp_es]):

        plt.subplot(2,2,i);i+=1;

        yp = scyit(y)-scyit(m.predict(x));

        plt.hist(yp,40,density=True)

        # fit and show normal distribution

        param_fit = sps.norm.fit(yp)

        plt.plot(np.linspace(-4,4,40),

                 sps.norm.pdf(np.linspace(-4,4,40),loc=param_fit[0],scale=param_fit[1]),"r-")

        plt.title("Histogram: {}-{} set".format(n,t));plt.ylabel("PDF")

        if i < 4:

            plt.gca().set_xticklabels([])

        else:

            plt.xlabel("error")
plt.figure(figsize=(6,6))

i = 1;

for (t,x,y) in zip(["Train","Test"],[X_train,X_test],[y_train,y_test]):

    for (n,m) in zip(["CV","ES"],[mlp_cv,mlp_es]):

        plt.subplot(2,2,i);

        # target & prediction for plotting

        yp = m.predict(x);

        xp = y.reshape(y.size,1); # just to fit linear line

        # Plot input data. Due to the large data set, plot every 14th data point

        # this is done here only to make the export jupyter html file smaller. Normally plot all!!

        plt.plot(y[::14],yp[::14],"b.",markersize=1.5) 

        # Corresponding linear-fit line

        lm = LinearRegression().fit(xp,yp)

        plt.plot(y,lm.predict(xp),"r-") 

        # Same as above but robust fit

        lm = RANSACRegressor(LinearRegression()).fit(xp,yp)

        plt.plot(y,lm.predict(xp),"m-")

        # add perfect fit line

        plt.plot([np.min(y),np.max(y)],[np.min(y),np.max(y)],"k--");

        # Title (including R^2 value), Labels, and limits

        plt.title("{}-{} set $R^2$={:.2f}".format(n,t,r2_score(y,yp)));

        plt.xlim([np.min(y),np.max(y)]);

        plt.ylim([np.min(y),np.max(y)]);

        plt.axis("equal")

        plt.ylabel("prediction")

        if i < 3:

            plt.gca().set_xticklabels([])

        else:

            plt.xlabel("target")

        if i == 1:

            plt.legend(["input","fit","robust","perfect"])

        i+=1; # next sub-plot