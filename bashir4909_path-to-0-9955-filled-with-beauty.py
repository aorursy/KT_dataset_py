# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-Learn modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline

# others
from skimage.measure import block_reduce
%matplotlib inline
_ = plt.style.use('ggplot')
_ = plt.style.available
molecule_data = pd.read_csv('../input/roboBohr.csv', header=0, dtype=np.float64, usecols=list(range(1,1276))+[1277])
molecule_data.head()
pre_X = molecule_data.drop(columns=['Eat'])
zero_mask_X = (pre_X==0)
print("{0:.2f} % of cells were actually padded zero"
      .format(100.0 * zero_mask_X.values.flatten().sum() / (pre_X.shape[0]*pre_X.shape[1])))
print("--- --- --- ")
print("Turning them into np.nan")
pre_X[zero_mask_X] = np.nan
print("DONE!")
print("--- --- --- ")
X = StandardScaler().fit_transform(pre_X)
print("Scaling finished, slice of new feature data")
print(X[:8])
print('--- --- --- ')
print('Target values:')
y = molecule_data['Eat'].values
print(y)
# let us learn some stats and info about dataset
print("There are {} entries with {} features".format(X.shape[0], X.shape[1]))
print("--- --- --- ")
print("The statistical information about each feature (column)")
molecule_stats = pd.DataFrame(X).describe()
print(molecule_stats)
feature_indices = [0,1,2,300,500,700,-3,-2,-1]
chosen_features = ([ pd.DataFrame(X[:,i]).dropna().values for i in feature_indices])
### PLOTTING time ###
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_yscale('linear')
ax.set_xlabel('Value of feature')
ax.set_ylabel('Feature: top-rightmost, bottom-leftmost')
ax.tick_params(bottom=False, left=False, labelleft=False)
# print(chosen_features)
_ = ax.boxplot(chosen_features, showmeans=True,showcaps=True, showfliers=True,vert=False)
np.random.seed(0)
dist_gauss = np.random.normal(size=1000)
dist_gamma = np.random.gamma(shape=1.5, scale=1.5,size=1000)
dist_exp = np.random.exponential(size=1000)

fig = plt.figure(figsize=(10,5))
with plt.xkcd():
    plt.rcParams.update({'font.size':'10'})
    ax_gauss = fig.add_subplot(131)
    ax_gauss.tick_params(left=False, labelleft=False)
    ax_gamma = fig.add_subplot(132)
    ax_gamma.tick_params(left=False, labelleft=False)
    ax_exp = fig.add_subplot(133)
    ax_exp.tick_params(left=False, labelleft=False)
    ax_gauss.hist(dist_gauss,density=True,bins=17)
    ax_gauss.set_title("Normal(Gauss) distribution")
    ax_gamma.hist(dist_gamma,density=True,bins=17)
    ax_gamma.set_title("Gamma distribution")
    ax_exp.hist(dist_exp,density=True,bins=17)
    ax_exp.set_title("Exponential distribution")

    
##############################
### Put missing values back ### 
##############################

X[zero_mask_X] = 0

###########
### PCA ###
###########
# try different number of components and see how well they explain variance
N_PCA=50
p = PCA(n_components=N_PCA).fit(X)
ns = list(range(N_PCA))

plt.figure()
plt.plot(ns, [ p.explained_variance_ratio_[n] for n in ns], 
         'r+', label="Explained variance - single feature")
plt.plot(ns, [ p.explained_variance_ratio_.cumsum()[n] for n in ns], 
         'b*', label="Explained variance - cumulative")
_ = plt.legend()
# from analyzing the graph we can see that about 25 components
# are enough to explain 96% variation in data. We can keep them

# Another thing that grinds my gears
# how well does new PCs explain energy levels?
X_reduced = p.transform(X)[:,:25]
plt.style.use('grayscale')
fig = plt.figure(figsize=(10,10))
axs = fig.subplots(5,5,sharex=True,sharey=True)
axs = np.array(axs).flatten()
for ax in axs[[0,5,10,15,20]]:
    ax.set_ylabel("Energy Level")
for i in range(25):
    ax = axs[i]
    ax.scatter(X_reduced[:,i],y,s=0.1, alpha=0.2)
    ax.set_xlabel("PC-{}".format(i+1), labelpad=2)
    ax.tick_params(left=False, bottom=False)
plt.style.use('ggplot')
plt.set_cmap("magma")
plt.figure(figsize=(8,8))
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
_ = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y,s=5,alpha=0.2)
##################
### KERNEL PCA ###
##################

mask_random = np.random.randint(0,X.shape[0],size=2000)
kp = KernelPCA(n_components=100,kernel='sigmoid', gamma=0.5, max_iter=250).fit(X[mask_random])
print("PCA is trained on kernel")
print('--- --- --- ')
X_reduced = kp.transform(X)[:,:50]
plt.style.use('grayscale')
fig = plt.figure(figsize=(16,8))
axs = fig.subplots(5,10,sharex=True, sharey=True)
axs = np.array(axs).flatten()
for i in range(50):
    ax = axs[i]
    ax.scatter(X_reduced[:,i],y,s=0.1, alpha=0.2)
    ax.tick_params(left=False, bottom=False)
plt.style.use('ggplot')
plt.set_cmap("magma")
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
_ = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y,s=5,alpha=0.2)

plt.subplot(122)
plt.xlabel("PC 2")
plt.ylabel("PC 3")
plt.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
_ = plt.scatter(X_reduced[:,1], X_reduced[:,2], c=y,s=5,alpha=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_reduced[:,:2], y)
R_clf = Ridge(alpha=10).fit(X_train, y_train)
R_clf.score(X_test, y_test)
# pretty embarassing, maybe try quadratic regression?
R_clf = Ridge(alpha=100).fit(PolynomialFeatures(degree=3).fit_transform(X_train), y_train)
R_clf.score(PolynomialFeatures(degree=3).fit_transform(X_test), y_test)
# much better but not promising, KNN still better
# how about Decision Tree and a bit more components
X_train, X_test, y_train, y_test = train_test_split(X_reduced[:,:10], y)
dt_clf = DecisionTreeRegressor(max_depth=25).fit(X_train, y_train)
dt_clf.score(X_test, y_test)
#NICE!!! well maybe Linear models (with poly features) are good fit too
R_clf = Ridge(alpha=0.05).fit(PolynomialFeatures(degree=3).fit_transform(X_train), y_train)
R_clf.score(PolynomialFeatures(degree=3).fit_transform(X_test), y_test)
# Not that bad, Ridge would probably have better generalization, however trees have another trick up their sleeve
# RANDOM FORESTS, this is basically a way to overcome overfitting and have better generalization
forest_reg = RandomForestRegressor(n_estimators=100,max_depth=30).fit(X_train, y_train)
forest_reg.score(X_test, y_test)
# Seems like we are pushing it to limits and hitting wall, there is another Ensemble model for trees -> ExtremeTrees!!!
ex_tree = ExtraTreesRegressor(n_estimators=100, max_depth=22).fit(X_train, y_train)
ex_tree.score(X_test, y_test)
# do not forget to seperate validation and test data
X_inter, X_test, y_inter, y_test = train_test_split(X,y,test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_inter,y_inter,test_size=0.2)
# KIDDING! I will do everythin manually, changing every parameter by hand to get good accuracy
N_BOOTSRAP = 2000
GAMMA = 0.0005
scores = []
np.random.seed(0)
for i in range(5):
    X_train, X_val, y_train, y_val = train_test_split(X_inter,y_inter,test_size=0.2)
    kp = (KernelPCA(n_components=10, kernel='sigmoid', gamma=GAMMA, max_iter=250)
          .fit(X_train[np.random.randint(0,X_train.shape[0],size=N_BOOTSRAP)]))
    X_train_reduce = kp.transform(X_train)
    X_val_reduce = kp.transform(X_val)
    ex_regr = ExtraTreesRegressor(n_estimators=100, max_depth=25).fit(X_train_reduce, y_train)
    r2_score = ex_regr.score(X_val_reduce, y_val)
    scores.append(r2_score)
print("Score on 5 different cross-vals:\n{}\n".format(scores))
print("Average {}".format(np.mean(scores)))
%%time
kp = (KernelPCA(n_components=10, kernel='sigmoid', gamma=0.0005, max_iter=250)
       .fit(X_inter[np.random.randint(0,X_inter.shape[0],size=2000)]))
X_train_reduce = kp.transform(X_inter)
X_test_reduce = kp.transform(X_test)
ex_regr = ExtraTreesRegressor(n_estimators=100, max_depth=25).fit(X_train_reduce, y_inter)
ex_regr.score(X_test_reduce, y_test)
