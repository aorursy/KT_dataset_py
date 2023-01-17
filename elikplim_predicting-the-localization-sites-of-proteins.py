import numpy

import pandas

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier





import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix





from sklearn import cross_validation

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error







from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import np_utils

#from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold

from keras.constraints import maxnorm



# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)

# load dataset

dataframe = pandas.read_csv("../input/ecoli.csv", delim_whitespace=True)



# Assign names to Columns

dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']



dataframe = dataframe.drop('seq_name', axis=1)



# Encode Data

dataframe.site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'),(1,2,3,4,5,6,7,8), inplace=True)

print("Head:", dataframe.head())
print("Statistical Description:", dataframe.describe())
print("Shape:", dataframe.shape)



print("Data Types:", dataframe.dtypes)
print("Correlation:", dataframe.corr(method='pearson'))
dataset = dataframe.values





X = dataset[:,0:7]

Y = dataset[:,7] 
#Feature Selection

model = LogisticRegression()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y)



print("Number of Features: ", fit.n_features_)

print("Selected Features: ", fit.support_)

print("Feature Ranking: ", fit.ranking_) 



plt.hist((dataframe.site))

dataframe.plot(kind='density', subplots=True, layout=(3,4), sharex=False, sharey=False)
dataframe.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,7,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(dataframe.columns)

ax.set_yticklabels(dataframe.columns)



num_instances = len(X)



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('L_SVM', LinearSVC()))

models.append(('ETC', ExtraTreesClassifier()))

models.append(('RFC', RandomForestClassifier()))



# Evaluations

results = []

names = []



for name, model in models:

    # Fit the model

    model.fit(X, Y)

    

    predictions = model.predict(X)

    

    # Evaluate the model

    kfold = cross_validation.KFold(n=num_instances, n_folds=10, random_state=seed)

    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
   

#boxplot algorithm Comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Define 10-fold Cross Valdation Test Harness

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []

for train, test in kfold.split(X, Y):



    # create model

    model = Sequential()

    model.add(Dense(20, input_dim=7, init='uniform', activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(10, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(5, init='uniform', activation='relu'))

    model.add(Dense(1, init='uniform', activation='relu'))



    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



    # Fit the model

    model.fit(X[train], Y[train], epochs=200, batch_size=10, verbose=0)



    # Evaluate the model

    scores = model.evaluate(X[test], Y[test], verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


