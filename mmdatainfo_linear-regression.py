# Scikit-learn

from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, KFold, train_test_split

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.feature_selection import SelectFromModel, RFECV

from sklearn.decomposition import PCA

from sklearn.dummy import DummyRegressor

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor, HuberRegressor, SGDRegressor

from sklearn.metrics import mean_squared_error

# Other libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Statistical/hypothesis tests

from scipy.stats import normaltest, jarque_bera, anderson, shapiro
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
def genpolydata(X):

    return 10.0 + 3.0 * X[:,0] - 2.0 * X[:,1] + 0.1 * (X[:,0]**2) + 0.4 * (X[:,1]**2);
# regular grid converted to features matrix

x0, y0 = np.meshgrid(np.linspace(0.0,200.0,num=30), np.linspace(-100,100,num=20))



# Convert to feature matrix

X = np.concatenate((np.asanyarray(x0).reshape(x0.size,1),

                    np.asanyarray(y0).reshape(y0.size,1)),axis=1);

y = genpolydata(X);
np.random.seed(0)

y += np.random.randn(y.size)*100; # noise

for i in range(0,y.size):

    if i < 5:

        y[i] = 10000; # outlier
fig = plt.figure(figsize=(8,4));

ax = fig.add_subplot(111, projection='3d');

ax.plot_surface(x0,y0,genpolydata(X).reshape(x0.shape));

ax.scatter(X[:,0],X[:,1],y,c="r",marker=".");

plt.title("Input data (with and withou noise)");
kf = KFold(n_splits=5,random_state=1)

Xs = PolynomialFeatures(include_bias=False,degree=2).fit_transform(X);

Xt = RFECV(LinearRegression(),cv=kf).fit(Xs,y).transform(Xs)
Xt[100,:]
Xs[100,:]
plt.figure(figsize=(9,2))

i = 1;

for train_ind,test_ind in kf.split(X):

    plt.subplot(1,5,i);i+=1

    plt.hist(y[train_ind],20);

    plt.hist(y[test_ind],20);
print("Fit results for all sub-sets (RMSE should be around 100, see input noise):\n")

for (n,m) in zip(["Normal    ","Robust-Iter","Robust-Huber"],

        [LinearRegression(),RANSACRegressor(LinearRegression()),HuberRegressor()]):

    error = cross_val_score(m,Xs,y,cv=kf,scoring="neg_mean_squared_error");

    print("{} \tcross-validation RMSE errors: \t{}".format(

        n,[np.round(i,2) for i in np.sqrt(np.abs(error))]))

    
for (n,m) in zip(["Ridge   ","Lasso   "],

                 [Ridge(max_iter=10000),

                  Lasso(max_iter = 10000)]):

    gscv = GridSearchCV(m,{"alpha":[10**i for i in range(-4,6)]},cv=kf,

                            scoring="neg_mean_squared_error",

                            return_train_score=True).fit(Xs,y);

    error = cross_val_score(

        gscv.best_estimator_,Xs,y,cv=kf,scoring="neg_mean_squared_error");

    print("{} \tcross-validation RMSE errors: \t{}".format(

        n,[np.round(i,2) for i in np.sqrt(np.abs(error))]))
np.random.seed(0)

def sgdmodel(x,a,b,noise=False):

    if noise==True:

        return a + b*x + np.random.randn(x.size)/10

    else:

        return a + b*x
x = np.arange(1,100)

y = sgdmodel(x,10,0.01,noise=True)
# Set learn rate & error change tolerance

ypsilon = 0.0001;

e_tol = 1e-12;
def comploss(y_given,x,a,b):

    return np.sum((y_given - sgdmodel(x,a,b))**2.0/x.size)



def derivative_a(y_given,x,a,b):

    return np.sum(-2.0/x.size * (y_given-sgdmodel(x,a,b)))



def derivative_b(y_given,x,a,b):

    return np.sum(-2.0/x.size * x * (y_given-sgdmodel(x,a,b)))



def estim_values(a_est,b_est):

    err_change = 100.;

    e_est = 100.;

    i = 1;

    while err_change > e_tol:

        # new a,b parameters

        a_est = a_est - ypsilon*derivative_a(y,x,a_est,b_est);

        b_est = b_est - ypsilon*derivative_b(y,x,a_est,b_est);

        # estimate new error and change of error

        e_curr = comploss(y,x,a_est,b_est);

        err_change = np.abs(e_est-e_curr);

        e_est = e_curr;

        i += 1;

        # break to avoid too many iterations

        if i > 1000000:

            break

            

    print("Batch: number of iterations: ",i);

    return a_est,b_est
a_est,b_est = estim_values(0.0,0.0) # the computation will take some time

print("Batch: constant = {:.2f}, slope = {:.4f}".format(a_est,b_est))
# Set learn rate & error change tolerance

ypsilon = 0.00001;

e_tol = 1e-13;
def derivative_a_i(y,x,a,b):

    return -2.0 * (y-sgdmodel(x,a,b));

def derivative_b_i(y,x,a,b):

    return -2.0 *x *(y-sgdmodel(x,a,b));

    

def estim_values_stoch(a_est,b_est):

    err_change = 100.;

    e_est = 100.;

    i = 1;

    # re-shuffle data

    xi,yi = shuffle(x,y,random_state=123);

    while err_change > e_tol:

        # new a,b parameters

        for j in range(0,x.size):

            a_est = a_est - ypsilon*derivative_a_i(yi[j],xi[j],a_est,b_est);

            b_est = b_est - ypsilon*derivative_b_i(yi[j],xi[j],a_est,b_est);

        # estimate new error and change of error

        e_curr = comploss(y,x,a_est,b_est);

        err_change = abs(e_est-e_curr);

        e_est = e_curr;

        i += 1;

        # break to avoid too many iterations

        if i > 1000000:

            break

            

    print("Batch: number of iterations: ",i);

    return a_est,b_est
# the computation will take some time

a_est2,b_est2 = estim_values_stoch(0.0,0.0) 

print("Batch: constant = {:.2f}, slope = {:.4f}".format(a_est2,b_est2))
# sgd = SGDRegressor(fit_intercept=False, penalty="none",learning_rate = "constant",

#                    eta0=ypsilon, max_iter=1000000,tol=e_tol,shuffle=True).fit(x.reshape(x.size,1),y)

lsq = LinearRegression(fit_intercept=True).fit(x.reshape(x.size,1),y)
plt.figure(figsize=(5,2.5))

plt.plot(x,y,"k.")

plt.plot(x,lsq.predict(x.reshape(x.size,1)),"r")

plt.plot(x,sgdmodel(x,a_est,b_est),"g--")

plt.plot(x,sgdmodel(x,a_est2,b_est2),"b-")

plt.legend(["input","LSQ","Batch","SGD"]);

plt.title("Gradient-based estimation");
# house = datasets.fetch_california_housing()

# df = pd.DataFrame(house.data,columns=house.feature_names)

# df = df.assign(target=house.target)
!ls ../input/test-data/california_housing.csv
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
# Here, the data is scaled prior train-test split. 

# In real applications, first split and scale afterwards to simulate real-world scenario where we do not have the test set!

X = StandardScaler().fit_transform(df.drop(["target","AveOccup"],axis=1).values);

X = PCA(n_components="mle").fit_transform(X)

y = df.target.values
[X_train,X_test,y_train,y_test] = train_test_split(X,y,train_size=0.67,test_size=0.33,

                                                   random_state=123);
lr = LinearRegression()

lr_dummy = DummyRegressor(strategy="mean")

parameters = {"alpha":[10**i for i in range(-3,3)],

              "solver":["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]};

lr_reg_opt = GridSearchCV(Ridge(),parameters,cv=5)
for i in [lr,lr_reg_opt,lr_dummy]:

    i.fit(X_train,y_train);
for (n,m) in zip(["simple regression ","ridge/cv regression","dummy / mean value"],

                 [lr,lr_reg_opt,lr_dummy]):

    print("RMSE ",n," train =\t {:.1f} *1000$".format(

        100*np.sqrt(mean_squared_error(y_test,m.predict(X_test)))))

plt.figure(figsize=(8,2))

i = 1;

for (n,m) in zip(["simple regression ", "ridge/cv regression","dummy / mean value"],

                 [lr,lr_reg_opt,lr_dummy]):

    plt.subplot(1,4,i); i += 1;

    plt.hist(y_test-m.predict(X_test),30,density=True)

    plt.title(n)
for (n,m) in zip(["simple ","ridge/cv","dummy"],

                 [lr,lr_reg_opt,lr_dummy]):

    for (tn,tf) in zip(["DAgostino and Pearson","Jarque - Bera", "Shapiro - Wilk",

                            "Anderson - Darling"],

                       [normaltest,jarque_bera,shapiro,anderson]):

        if tf == anderson:

            stat_val,crit_val,p_val = tf((y_train-m.predict(X_train)).reshape(X_train.shape[0]))

            print("{}:{} statistic = {:.3f}, critical = {}, p_val = {}\t".format(n,tn,

                                                                            stat_val,crit_val[2:],p_val[2:]));



        else:

            stat_val,p_val = tf(y_train-m.predict(X_train))

            print("{}:{} p-value = {} (reject for small)".format(n,tn,p_val));