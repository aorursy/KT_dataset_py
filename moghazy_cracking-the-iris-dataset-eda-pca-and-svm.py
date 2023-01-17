%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





import pandas as pd



# mapping the 3 target classes names to numeric 

# values to be able to use them later in our model

#since our model only accepts numeric numbers

X = pd.read_csv("../input/Iris.csv")



# To map the strings you need to use map(dict) function 

# from the dataframe, This function accepts a dictionary 

# with the values to be replaced as Keys and the values 

# that would replace them as the values for those keys 

z = {'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3 }



X['Species'] = X['Species'].map(z)

print ("Number of data points ::", X.shape[0])

print("Number of features ::", X.shape[1])
X.head()
import matplotlib.pyplot as plt

import numpy as np

# X.species holds the classes that we have after mapping which are {1, 2, 3}

classes = np.array(list(X.Species.values))



# Now we will use matplotlib to plot the classes with the axis x and y representing two of the features that we have

# We do that to see if some features contribute to the seperablity of the dataset more than the others

def plotRelation(first_feature, sec_feature):

    

    plt.scatter(first_feature, sec_feature, c = classes, s=10)

    plt.xlabel(first_feature.name)

    plt.ylabel(sec_feature.name)

    

f = plt.figure(figsize=(25,20))

f.add_subplot(331)

plotRelation(X.SepalLengthCm, X.SepalWidthCm)

f.add_subplot(332)

plotRelation(X.PetalLengthCm, X.PetalWidthCm)

f.add_subplot(333)

plotRelation(X.SepalLengthCm, X.PetalLengthCm)

f.add_subplot(334)

plotRelation(X.SepalLengthCm, X.PetalWidthCm)

f.add_subplot(335)

plotRelation(X.SepalWidthCm, X.PetalLengthCm)

f.add_subplot(336)

plotRelation(X.SepalWidthCm, X.PetalWidthCm)
import matplotlib.pyplot as plt

import numpy as np

# X.species holds the classes that we have after mapping which are {1, 2, 3}

classes = np.array(list(X.Species.values))



# Now we will use matplotlib to plot the classes with the axis x and y representing two of the features that we have

# We do that to see if some features contribute to the seperablity of the dataset more than the others

import seaborn as sns



Exploration_columns = X.drop('Id' ,   axis = 1)

sns.pairplot(Exploration_columns, hue = "Species")
import seaborn as sns



# Here we use the seaborn library to visualize the correlation matrix

# The correlation matrix shows how much are the features and the target correlated

# This gives us some hints about the feature importance

import matplotlib.pyplot as plt





corr = X.corr()

f, ax = plt.subplots(figsize=(15, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)
f = plt.figure(figsize=(25,10))

f.add_subplot(221)

X.SepalWidthCm.hist()

f.add_subplot(222)

X.SepalLengthCm.hist()

f.add_subplot(223)

X.PetalLengthCm.hist()

f.add_subplot(224)

X.PetalWidthCm.hist()
f = plt.figure(figsize=(25,10))

f.add_subplot(221)

sns.boxplot(x=X['PetalWidthCm'])

f.add_subplot(222)

sns.boxplot(x=X['PetalLengthCm'])

f.add_subplot(223)

sns.boxplot(x=X['SepalLengthCm'])

f.add_subplot(224)

sns.boxplot(x=X['SepalWidthCm'])





sns.boxplot(x=X['PetalWidthCm'])
from scipy import stats

import numpy as np



z = np.abs(stats.zscore(X))



zee = (np.where(z > 2.5))[1]



print("number of data examples greater than 3 standard deviations = %i " % len(zee))
data_delete = X[(z >= 2.5)]

data_delete.drop_duplicates(keep='first', inplace=True)

unique, count= np.unique(data_delete["Species"], return_counts=True)

print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )
X = X[(z <= 2.5)]
# Removing the label from the training data

y = X['Species']

X = X.drop(["Species"], axis = 1)



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
fig = plt.figure(1, figsize=(16, 9))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(X_scaled)



ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()

print("The number of features in the new subspace is " ,X_reduced.shape[1])
import plotly.graph_objects as go



fig = go.Figure(data=[go.Scatter3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2], mode='markers', marker=dict( size=4, color=y, colorscale= "Portland", opacity=0.))])



fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

                        X_reduced, y, test_size=0.2, random_state=42)
from sklearn.svm import LinearSVC



clf = LinearSVC(penalty='l2', loss='squared_hinge',

                dual=True, tol=0.0001, C=100, multi_class='ovr',

                fit_intercept=True, intercept_scaling=1, class_weight=None,verbose=0

                , random_state=0, max_iter=1000)

clf.fit(X_train,y_train)



print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train)))



print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import numpy as np

    

c = np.logspace(start = -15, stop = 1000, base = 1.02)

param_grid = {'C': c}





grid = GridSearchCV(clf, param_grid =param_grid, cv=3, n_jobs=-1, scoring='accuracy')

grid.fit(X_train, y_train)

  

print("The best parameters are %s with a score of %0.0f" % (grid.best_params_, grid.best_score_ * 100 ))

print( "Best estimator accuracy on test set {:.2f} ".format(grid.best_estimator_.score(X_test, y_test) * 100 ) )
from sklearn.svm import SVC



clf_SVC = SVC(C=100.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 

          probability=False, tol=0.001, cache_size=200, class_weight=None, 

          verbose=0, max_iter=-1, decision_function_shape="ovr", random_state = 0)

clf_SVC.fit(X_train,y_train)



print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import numpy as np

    

c_SVC = np.logspace(start = 0, stop = 10, num = 100, base = 2 , dtype = 'float64')

print( 'the generated array of c values')

print ( c_SVC )

param_grid_S = {'C': c_SVC}







print("\n Array of means \n")

clf = GridSearchCV(clf_SVC, param_grid =param_grid_S, cv=20 , scoring='accuracy')

clf.fit(X_train, y_train)

means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']

print(means)



y_true, y_pred = y_test, clf.predict(X_test)

print( '\nClassification report\n' )

print(classification_report(y_true, y_pred))
