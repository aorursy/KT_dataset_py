# Scikit-learn

from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import mean_squared_error

# Other libraries

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# Use vector drawing inside jupyter notebook & matplotlib default axis font size

%config InlineBackend.figure_format = "svg"

plt.rcParams.update({'font.size': 9})
# Generate data using 3 degree polynomial (degree 6 will be fitted)

X_all = np.arange(1,6,step=0.05).reshape(100,1) # small numbers=no scaling in this case

y_all = np.polyval([0.03,0.064,-2.0,6],X_all[:,0])

# add noise

np.random.seed(0)

y_noise = y_all + np.random.randn(y_all.size)/5



# Inbalanced split to train/test set

X_train, X_test, y_train, y_test = [],[],[],[];

for i in range(0,100):

    if i in [1,3,4,5,0,45,46,47,74,75,76,77,78,79,95,96,97,98,99]:

        X_train,y_train = np.append(X_train,X_all[i]),np.append(y_train,y_noise[i])

    else:

        X_test,y_test = np.append(X_test,X_all[i]),np.append(y_test,y_noise[i])
# set degree of polynomial to be fitted

pf = PolynomialFeatures(degree=6,include_bias=True);

X_train_p = pf.fit_transform(X_train.reshape(X_train.size,1))

X_test_p = pf.fit_transform(X_test.reshape(X_test.size,1))

# Train

lsq = LinearRegression(fit_intercept=False).fit(X_train_p,y_train)

# Use fixed regularization strengt for this case

lsq_reg = Ridge(alpha=0.01,fit_intercept =False).fit(X_train_p,y_train)
print("Root-mean-square error based on whole train/test sets")

for (m,n) in zip([lsq,lsq_reg],["Ordinary LSQ","Regularized"]):

    print(n," \ttrain = {:.2f}, test = {:.2f}".format(

        np.sqrt(mean_squared_error(y_train,m.predict(X_train_p))),

        np.sqrt(mean_squared_error(y_test,m.predict(X_test_p)))))
print("Cross validated Root-mean-square error")

for (m,n) in zip([lsq,lsq_reg],["Ordinary LSQ","Regularized"]):

    temp = -cross_val_score(m,X_train_p,y_train,cv=5,scoring="neg_mean_squared_error");

    print(n," test \t= ",end="")

    for i in temp:

        print("{:.2f}".format(np.sqrt(i)),end=", " if i != temp[-1] else "\n")
plt.figure(figsize=(4,2.5));

plt.plot(X_all,y_all,"b-",linewidth=0.8);

plt.plot(X_train,y_train,"bo");

plt.plot(X_test,lsq.predict(X_test_p),"k-");

plt.plot(X_test,lsq_reg.predict(X_test_p),"r-");

plt.title("Regularization effect");

plt.legend(["\"True\"/no noise","Train set","Ordinary LSQ","Regularized"]);

plt.xlabel("feature (X)",fontsize=10);

plt.ylabel("label (y)",fontsize=10);
lsq_cv = GridSearchCV(Ridge(fit_intercept=False),cv=10,

                      param_grid = {"alpha":np.arange(0.002,0.01,step=0.002)},

                      scoring="neg_mean_squared_error").fit(X_train_p,y_train)
plt.figure(figsize=(4,2.5));

plt.plot(X_all,y_all,"b-",linewidth=0.8);

plt.plot(X_train,y_train,"bo");

plt.plot(X_test,lsq.predict(X_test_p),"k-");

plt.plot(X_test,lsq_cv.predict(X_test_p),"r-");

plt.title("CV for regularization strength");

plt.legend(["\"True\"/no noise","Train set","Ordinary LSQ","CV-regularized"]);

plt.xlabel("feature (X)",fontsize=10);

plt.ylabel("label (y)",fontsize=10);
lsq_cv.best_estimator_ 