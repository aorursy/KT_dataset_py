# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
#format plot style

sns.set(style="white", color_codes=True)

%matplotlib inline
df = pd.read_csv('../input/Iris.csv', header = 0)
df.columns = ['Id',"sepal_length", 'sepal_width', 'petal_length', 'petal_width', 'Class']
df = df[["sepal_length", 'sepal_width', 'petal_length', 'petal_width', 'Class']]
df.head()
df.shape
df.describe()
print(df.groupby('Class').size())
#Univariate Plot

ax = sns.boxplot(data=df)

ax = sns.stripplot(data=df, jitter=True,edgecolor="gray")
df.hist()

plt.show()
#Multivariate plots

sns.pairplot(df, hue = 'Class', size =3,diag_kind = 'kde')
from pandas.plotting import andrews_curves

andrews_curves(df, 'Class')
from pandas.plotting import parallel_coordinates

parallel_coordinates(df, 'Class')
from pandas.plotting import radviz

radviz(df, 'Class')
corr = df.corr()

corr

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr, annot = True, square=True, ax=ax)
from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import RFE



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
#Validation Set

array = df.values

x = array[:,0:4]

y = array[:,4]

validation_size = 0.20

seed = 7

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y = le.fit_transform(y)

#Check encoding

le.transform(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
seed = 7

scoring = 'accuracy'
# Isolate Data, class labels and column values

names = df.columns.values



# Build the model

rfc = RandomForestClassifier()



# Fit the model

rfc.fit(x, y)



# Print the results

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse=True))
importance = rfc.feature_importances_



# Sort the feature importances 

sorted_importances = np.argsort(importance)



# Insert padding

padding = np.arange(len(names)-1) + 0.5



# Plot the data

plt.barh(padding, importance[sorted_importances], align='center')



# Customize the plot

plt.yticks(padding, names[sorted_importances])

plt.xlabel("Relative Importance")

plt.title("Variable Importance")



# Show the plot

plt.show()
#3D Plot of features with higher importance score



fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-170, azim=150)



# Plot the training points

ax.scatter(x[:, 0], x[:, 2],x[:,3], c=Y, cmap=plt.cm.Set1,edgecolor='k', s=40)

ax.set_xlabel('Sepal length')

ax.set_ylabel('Petal length')

ax.set_zlabel('Petal width')



ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])
model = LogisticRegression()

rfe = RFE(model, 3)

rfe = rfe.fit(x_train,y_train)

print(rfe.support_)

print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names), reverse=True))
#3D Plot of features with higher importance score



fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-170, azim=140)



# Plot the training points

ax.scatter(x[:, 1], x[:, 2],x[:,3], c=Y, cmap=plt.cm.Set1,edgecolor='k', s=40)

ax.set_xlabel('Sepal width')

ax.set_ylabel('Petal length')

ax.set_zlabel('Petal width')



ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])
# Make predictions on validation dataset - Random Forest Classifier

rfc_predictions = rfc.predict(x_validation)

print(accuracy_score(y_validation,rfc_predictions))

print(confusion_matrix(y_validation,rfc_predictions))

print(classification_report(y_validation,rfc_predictions))
# Make predictions on validation dataset - Logistic Regression

lr = LogisticRegression()

lr.fit(x_train, y_train)

lr_predictions = lr.predict(x_validation)

print(accuracy_score(y_validation, lr_predictions))

print(confusion_matrix(y_validation, lr_predictions))

print(classification_report(y_validation, lr_predictions))
# Make predictions on validation dataset - KNN

knn = KNeighborsClassifier() #default n_neighbors = 5 i.e., classification based on 5 nearest neighbors

knn.fit(x_train, y_train)

knn_predictions = knn.predict(x_validation)

print(accuracy_score(y_validation, knn_predictions))

print(confusion_matrix(y_validation, knn_predictions))

print(classification_report(y_validation, knn_predictions))
# Make predictions on validation dataset - Linear Discriminent Analysis

lda = LinearDiscriminantAnalysis()

lda.fit(x_train, y_train)

lda_predictions = lda.predict(x_validation)

print(accuracy_score(y_validation, lda_predictions))

print(confusion_matrix(y_validation, lda_predictions))

print(classification_report(y_validation, lda_predictions))
# Make predictions on validation dataset - Decision Tree Classifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

dt_predictions = dt.predict(x_validation)

print(accuracy_score(y_validation, dt_predictions))

print(confusion_matrix(y_validation, dt_predictions))

print(classification_report(y_validation, dt_predictions))
# Make predictions on validation dataset - Naive Bayes

nb = GaussianNB()

nb.fit(x_train, y_train)

nb_predictions = nb.predict(x_validation)

print(accuracy_score(y_validation, nb_predictions))

print(confusion_matrix(y_validation, nb_predictions))

print(classification_report(y_validation, nb_predictions))
# Make predictions on validation dataset - SVM

svm = SVC()

svm.fit(x_train,y_train)

svm_predictions = svm.predict(x_validation)

print(accuracy_score(y_validation,svm_predictions))

print(confusion_matrix(y_validation,svm_predictions))

print(classification_report(y_validation,svm_predictions))
a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(x_train,y_train)

    prediction=model.predict(x_validation)

    a=a.append(pd.Series(accuracy_score(y_validation,prediction)))

plt.plot(a_index, a)

plt.xticks(x)
from sklearn.model_selection import GridSearchCV

parameters = {'C':[10,1,0.1,0.01], 'solver':('newton-cg', 'lbfgs', 'sag')}

model= GridSearchCV(LogisticRegression(), parameters)

model.fit(x_train,y_train)

#print(model.best_score_)

#print(model.best_estimator_)

print(model.best_params_)

lrGCV=pd.DataFrame(model.cv_results_)

lrGCV[['param_C','param_solver','mean_test_score']]
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

    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
import pickle

output = open('classifier.pkl', 'wb')

pickle.dump(rfc, output)

output.close()
f = open('classifier.pkl', 'rb')

classifier = pickle.load(f)

f.close()
classifier.score(x_validation,y_validation)