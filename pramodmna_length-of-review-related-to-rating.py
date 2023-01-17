import pandas

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import numpy
dataset=pandas.read_csv('../input/reviews.csv')
dataset.head(5)
dataset.info()
ds=dataset.drop(['Unnamed: 0','id','username','location','quote','page','reviewnospace','date'],axis=1)
ds.head(10)
op=ds.titleopinion
op
op.head(10)
ds['new'] = [x.replace(' ', '') for x in op]

ds=ds.drop(['titleopinion'],axis=1)
ds.head(10)


opp=ds.new
ds['newnew']=[len(i) for i in opp]
ds.head(10)
data=ds.drop(['new','userop'],axis=1)
data.head(10)

cols = data.columns.tolist()

cols = [cols[1]] + cols[0:1] + cols[2:]

data = data[cols]
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()
data.head(1)
scatter_matrix(data)

plt.show()
# Split-out validation dataset

array = data.values

X = array[:,0]

Y = array[:,1]

X=X.reshape((9188,1))

Y=Y.reshape((9188,1))

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)
num_folds = 7

num_instances = len(X_train)

seed = 7

scoring = 'accuracy'



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)