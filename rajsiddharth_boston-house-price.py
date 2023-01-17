# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
# repository website.
# Load dataset
filename = '../input/boston-house-prices/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)
# shape
print(dataset.shape)
# types
print(dataset.dtypes)
# We can see that all of the attributes are numeric, mostly real values (float) and some have
# been interpreted as integers (int).
# head
print(dataset.head(20))
# We can confirm that the scales for the attributes are all over the place because of the differing
# units. We may benefit from some transforms later on.
# descriptions
set_option('precision', 3)
#print(dataset.describe(percentiles=[.1,.2,.9]))
print(dataset.describe())
# We now have a better feeling for how different the attributes are. The min and max values
# as well are the means vary a lot. We are likely going to get better results by rescaling the data
# in some way.
# correlation
set_option('precision', 3)
dataset.corr(method='pearson')
# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()
# We can see that some attributes may have an exponential distribution, such as CRIM, ZN,
# AGE and B. We can see that others may have a bimodal distribution such as RAD and TAX.
# density
pyplot.figure(figsize = (10,12))
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True,
fontsize=1)
pyplot.show()
# This perhaps adds more evidence to our suspicion about possible exponential and bimodal
# distributions. It also looks like NOX, RM and LSTAT may be skewed Gaussian distributions, which
# might be helpful later with transforms.
# box and whisker plots

dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False,
fontsize=8)
pyplot.show()
#  This helps point out the skew in many distributions so much so that data looks like outliers
# (e.g. beyond the whisker of the plots).
#drawing scatter plot and ,, increasing the size for better view
#fig_size = pyplot.rcParams["figure.figsize"]
#fig_size[0]= 12
#fig_size[1]= 9
#pyplot.rcParams["figure.figsize"]= fig_size
pyplot.figure(figsize=(12,9))

scatter_matrix(dataset)
pyplot.show()
# We can see that some of the higher correlated attributes do show good structure in their
# relationship. Not linear, but nice predictable curved relationships.
# correlation matrix
fig = pyplot.figure(figsize=(12,9))
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
# The dark YELLOW color shows positive correlation whereas the dark blue color shows negative
# correlation. We can also see some dark YELLOW and dark blue that suggest candidates for removal
# to better improve accuracy of models later on.
dataset['CHAS'].value_counts()
dataset['CHAS'] = dataset['CHAS'].map({1:'tract' , 0:'notract' })
dataset.head()
dataset['RAD'].value_counts()
dataset['RAD'] =dataset['RAD'].map({4:'4town' , 3:'4town',2:'4town',1:'4town' , 5:'8town', 6:'8town' ,7:'8town',8:'8town',24:'24town'})
dataset['RAD'].value_counts()
dataset.head()
# Split-out validation dataset
#array = dataset.values
#X = array[:,0:13]
#Y = array[:,13]
validation_size = 0.20
seed = 7

X = dataset.drop('MEDV' , axis = 1)
X.head()
y = dataset.MEDV
y[1:5]
X.dtypes
import pandas as pd
#X = pd.get_dummies(X)
#X.head()
# Split-out validation dataset
#array = dataset.values
#X = array[:,0:13]
#Y = array[:,13]
#validation_size = 0.20
#seed = 7

cat_ix = X.select_dtypes(include=['object', 'bool']).columns
num_ix = X.select_dtypes(include=['int64', 'float64']).columns
# one hot encode cat features only
ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
X = ct.fit_transform(X)
# label encode the target variable to have the classes 0 and 1
X[0:5 ,:]
X_DataFrame = pd.DataFrame(X)
X_DataFrame.head()
#dataset.dtypes
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,
test_size=validation_size, random_state=seed)
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'
import warnings 
warnings.filterwarnings('ignore')
# Spot-Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('SGD', SGDRegressor()))
# The algorithms all use default tuning parameters. Let's compare the algorithms. We will
# display the mean and standard deviation of MSE for each algorithm as we calculate it and
# collect the results for use later.
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = KFold(n_splits=num_folds, random_state=seed)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
model.fit(X_train, Y_train)
model.predict(X_train)
# It looks like LR has the lowest MSE, followed closely by CART.
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# We can see similar distributions for the regression algorithms and perhaps a tighter 
# distribution of scores for CART.
# The differing scales of the data is probably hurting the skill of all of the algorithms and
# perhaps more so for SVR and KNN. In the next section we will look at running the same
# algorithms using a standardized copy of the data.
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
model = LinearRegression()
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
features.append(('rfe', RFE(model, 3)))
feature_union = FeatureUnion(features)
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),('LR',
 LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union) ,('LASSO',
 Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),('EN',
 ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),
                                         ('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union)
                                          ,('CART',
 DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),
                                         ('SVR', SVR())])))
pipelines.append(('ScaledSGD', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),
                                         ('SGD', SGDRegressor())])))




results = []

names = []

for name, model in pipelines:
 kfold = KFold(n_splits=num_folds, random_state=seed)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# We can see that KNN has both a tight distribution of error and has the lowest score.
feature_union.fit(X_train ,Y_train)
from sklearn.linear_model import LinearRegression
# KNN Algorithm tuning
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)

#k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
#param_grid = dict(n_neighbors=k_values)
#model = KNeighborsRegressor()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)
from sklearn.decomposition import PCA

LR = LinearRegression()
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
features.append(('rfe', RFE(LR, 3)))
feature_union = FeatureUnion(features)

Scaler =StandardScaler()
KNN = KNeighborsRegressor()

#pipelines = Pipeline(steps =[('KNN',KNeighborsRegressor())])
model = Pipeline(steps=[('Scaler', Scaler),("Feature_Union",feature_union), ('KNN', KNN)])

k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
#param_grid = dict(n_neighbors=k_values)
param_grid = dict(KNN__n_neighbors=k_values)
#model = KNeighborsRegressor()

kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(X_train, Y_train)
# We can display the mean and standard deviation scores as well as the best performing value
# for k below.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
 print("%f (%f) with: %r" % (mean, stdev, param))
# You can see that the best for k (n neighbors) is 3 providing a mean squared error of
# -18.172137, the best so far.
grid.predict(rescaledX)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

from sklearn.pipeline import FeatureUnion
LR = LinearRegression()
ftu =FeatureUnion([('SKB' ,SelectKBest(k=7)) , ('pca' , PCA(n_components=3)),('rfe',RFE(LR ,5))])
rescaled_T = ftu.fit_transform(rescaledX ,Y_train)

model = KNeighborsRegressor(n_neighbors =3)
model.fit(rescaled_T, Y_train)
predictions = model.predict(rescaled_T)
mean_squared_error( predictions,Y_train)
scaler = StandardScaler().fit(X_validation)
rescaledX_t = scaler.transform(X_validation)


from sklearn.pipeline import FeatureUnion
LR = LinearRegression()
ftu =FeatureUnion([('SKB' ,SelectKBest(k=7)) , ('pca' , PCA(n_components=3)),('rfe',RFE(LR ,5))])
rescaled_test = ftu.fit_transform(rescaledX_t ,Y_validation)

predictions = grid.predict(rescaled_test)
print(mean_squared_error(Y_validation, predictions))
# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),('AB',
 AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),('GBM',
 GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),('RF',
 RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('feature_union', feature_union),('ET',
 ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
 kfold = KFold(n_splits=num_folds, random_state=seed)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
# Running the example calculates the mean squared error for each method using the default
# parameters. We can see that we're generally getting better scores than our linear and nonlinear
# algorithms in previous sections.
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# It looks like Gradient Boosting has a better mean score, it also looks like Extra Trees has a
# similar distribution and perhaps a better median score.
# We can probably do better, given that the ensemble techniques used the default parameters.
# In the next section we will look at tuning the Gradient Boosting to further lift the performance.

from sklearn.decomposition import PCA

LR = LinearRegression()
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
features.append(('rfe', RFE(LR, 3)))
feature_union = FeatureUnion(features)

Scaler =StandardScaler()
GB = GradientBoostingRegressor(random_state=seed)

#pipelines = Pipeline(steps =[('KNN',KNeighborsRegressor())])
model = Pipeline(steps=[('Scaler', Scaler),("Feature_Union",feature_union), ('GB', GB)])

param_grid = dict(GB__n_estimators=numpy.array([50,100,150,200,250,300,350,400]))

kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(X_train, Y_train)
# As before, we can summarize the best configuration and get an idea of how performance
# changed with each different configuration.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
 print("%f (%f) with: %r" % (mean, stdev, param))
# We can see that the best configuration was n estimators=400 resulting in a mean squared
# error of -9.356471, about 0.65 units better than the untuned method.
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)


from sklearn.pipeline import FeatureUnion
LR = LinearRegression()
ftu =FeatureUnion([('SKB' ,SelectKBest(k=7)) , ('pca' , PCA(n_components=3)),('rfe',RFE(LR ,5))])
rescaled_T = ftu.fit_transform(rescaledX ,Y_train)

model = GradientBoostingRegressor(random_state=seed, n_estimators=300)
model.fit(rescaled_T, Y_train)
scaler = StandardScaler().fit(X_validation)
rescaledX_t = scaler.transform(X_validation)


from sklearn.pipeline import FeatureUnion
LR = LinearRegression()
ftu =FeatureUnion([('SKB' ,SelectKBest(k=7)) , ('pca' , PCA(n_components=3)),('rfe',RFE(LR ,5))])
rescaled_test = ftu.fit_transform(rescaledX_t ,Y_validation)

predictions = model.predict(rescaled_test)
print(mean_squared_error(Y_validation, predictions))
predictions = grid.predict(X_validation)
print(mean_squared_error(Y_validation, predictions))

# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
 print("%f (%f) with: %r" % (mean, stdev, param))
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=seed, n_estimators=300)
model.fit(rescaledX, Y_train)

# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
print(model)
print(predictions)
# We can see that the estimated mean squared error is 11.8, close to our estimate of -9.3.
# Regression Project: Boston House Prices

# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)

# Summarize Data

# Descriptive statistics
# shape
print(dataset.shape)
# types
print(dataset.dtypes)
# head
print(dataset.head(20))
# descriptions, change precision to 2 places
set_option('precision', 1)
print(dataset.describe())
# correlation
set_option('precision', 2)
print(dataset.corr(method='pearson'))


# Data visualizations

# histograms
dataset.hist()
pyplot.show()
# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
pyplot.show()
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# Prepare Data

# Split-out validation dataset
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Evaluate Algorithms
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Make predictions on validation dataset

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))



