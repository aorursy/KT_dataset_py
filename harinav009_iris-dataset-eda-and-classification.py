import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
file = "../input/Iris.csv"
dataset = pd.read_csv(file,sep=",")
type(dataset)
dataset.head()
data = dataset.iloc[:,1:6]
data.describe()
data.shape
data.groupby('Species').size()
# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
from matplotlib.ticker import NullFormatter 
fig = plt.figure(figsize=(10, 10))
fig.suptitle('SepalLengthCm')
sns.FacetGrid(data, hue="Species", size=6) \
   .map(sns.kdeplot, "SepalLengthCm") \
   .add_legend()
plt.grid(True)
#fig.suptitle('PetalWidthCm')
#ax = fig.add_subplot(222)
#sns.FacetGrid(data, hue="Species", size=6) \
 #  .map(sns.kdeplot, "SepalWidthCm") \
 #  .add_legend()
#plt.grid(True)
#fig.suptitle('PetalLengthCm')
#ax = fig.add_subplot(223)
#sns.FacetGrid(data, hue="Species", size=6) \
#   .map(sns.kdeplot, "PetalLengthCm") \
#   .add_legend()
#plt.grid(True)
#fig.suptitle('PetalWidthCm')
#ax = fig.add_subplot(224)
#sns.FacetGrid(data, hue="Species", size=6) \
#   .map(sns.kdeplot, "PetalWidthCm") \
#   .add_legend()
#plt.grid(True)
#plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.5)
plt.show()
# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species

sns.FacetGrid(data, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
# We can look at an individual feature in Seaborn through a boxplot
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
sns.boxplot(data=data)
plt.show()
from matplotlib.ticker import NullFormatter 
fig = plt.figure(figsize=(10, 10))
fig.suptitle('SepalLengthCm')
ax = fig.add_subplot(221)
sns.boxplot(x="Species", y="SepalLengthCm", data=data)
plt.grid(True)
fig.suptitle('SepalWidthCm')
ax = fig.add_subplot(222)
sns.boxplot(x="Species", y="SepalWidthCm", data=data)
plt.grid(True)
fig.suptitle('PetalLengthCm')
ax = fig.add_subplot(223)
sns.boxplot(x="Species", y="PetalLengthCm", data=data)
plt.grid(True)
fig.suptitle('PetalWidthCm')
ax = fig.add_subplot(224)
sns.boxplot(x="Species", y="PetalWidthCm", data=data)
plt.grid(True)
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.5)
plt.show()
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(data=data,)

from matplotlib.ticker import NullFormatter 
fig = plt.figure(figsize=(10, 10))
fig.suptitle('SepalLengthCm')
ax = fig.add_subplot(221)
sns.violinplot(x="Species", y="SepalLengthCm", data=data, size=120)
plt.grid(True)
fig.suptitle('SepalWidthCm')
ax = fig.add_subplot(222)
sns.violinplot(x="Species", y="SepalWidthCm", data=data, size=120)
plt.grid(True)
fig.suptitle('PetalLengthCm')
ax = fig.add_subplot(223)
sns.violinplot(x="Species", y="PetalLengthCm", data=data, size=120)
plt.grid(True)
fig.suptitle('PetalWidthCm')
ax = fig.add_subplot(224)
sns.violinplot(x="Species", y="PetalWidthCm", data=data, size=120)
plt.grid(True)
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.5)
plt.show()
data.hist(figsize=(12,12))
scatter_matrix(data, figsize=(12,12))
plt.show()
 #Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sns.pairplot(data,hue="Species", size=3)
# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
sns.pairplot(data, hue="Species", size=3, diag_kind="kde")

from pandas.tools.plotting import radviz
radviz(data, "Species")


alldata = data.values
X = alldata[:,0:4] 
Y = alldata[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


#Test
seed = 7
scoring = 'accuracy'
# Various algos
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
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
sns.violinplot(data=results, size=120)
ax.set_xticklabels(names)
plt.show()
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))