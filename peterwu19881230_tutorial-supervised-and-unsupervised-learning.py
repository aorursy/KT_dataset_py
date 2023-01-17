import pandas as pd

import numpy as np

from sklearn import datasets

import matplotlib.pyplot as plt



iris = datasets.load_iris()

X = pd.DataFrame(iris.data) #Dataframe: 150 rows x 4 columns

y = iris.target #numpy.ndarray. class= 0,1,2
# logtistic regression 

#ref for one-vs-rest logtistic regression: https://chrisalbon.com/machine_learning/logistic_regression/one-vs-rest_logistic_regression/

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score



cv=5

model=LogisticRegression(solver='newton-cg',multi_class='ovr',penalty='l2')



grid_search = GridSearchCV(model, param_grid={},cv=cv,scoring='accuracy') 

grid_search.fit(X, y) 

print('Accuracy= ',grid_search.best_score_)
#Decision tree

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import precision_score

import matplotlib.pyplot as plt



cv=5

model=DecisionTreeClassifier()



grid_search = GridSearchCV(model, param_grid={},cv=cv,scoring='accuracy') 

grid_search.fit(X, y) 

print('Accuracy= ',grid_search.best_score_)



fig=plot_tree(grid_search.best_estimator_)





from matplotlib.pylab import rcParams

##set up the parameters

rcParams['figure.figsize'] = 10,10

plt.show
#function for models that need hyper-parameter tuning

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



def my_inner_cv(X,y,model,cv,param_grid,test_size,random_state):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state,stratify=y)



  grid_search = GridSearchCV(model, param_grid=param_grid,cv=cv,iid=False) 

  grid_search.fit(X_train, y_train)



  accuracy=accuracy_score(y_test,grid_search.best_estimator_.predict(X_test))

  

  return([grid_search.best_estimator_,grid_search.best_params_,accuracy])
#Random forest

from sklearn.ensemble import RandomForestClassifier



random_state=101

test_size=0.2

max_depth = list(range(1,11))

criterion=['entropy','gini']

n_estimators=list(range(1,10)) #this can possibly be set to a much higher value but it would take lots of time

param_grid={'criterion':criterion,'max_depth':max_depth,'n_estimators':n_estimators}

cv=5



model=RandomForestClassifier()

result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)



print('Accuracy= ',result[2])
#Boosting (gradient boosting)

from sklearn.ensemble import GradientBoostingClassifier



random_state=101

test_size=0.2

n_estimators = [5,10,15,20,50,100,200,400]

max_iter=1000



param_grid={'n_estimators':n_estimators}

cv=5

model=GradientBoostingClassifier()



result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)

print('Accuracy= ',result[2])
#SVM

from sklearn.svm import SVC



#==============

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) #avoid the "future warning" that I think is not important"

#==============



random_state=101

test_size=0.2

ds = [1,2,3,4]

Cs = [0.001, 0.01, 0.1, 1, 10,100,1000] #ref: https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0

kernels=['linear', 'poly', 'rbf', 'sigmoid'] # I removed 'precomputed' kernel because it only accepts data that looke like: (n_samples, n_samples). Ref: https://stackoverflow.com/questions/36306555/scikit-learn-grid-search-with-svm-regression/36309526

param_grid={'degree':ds,'C':Cs, 'kernel':kernels}

cv=5

model=SVC()



result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)

print('Accuracy= ',result[2])
#Neuro net 

from keras.layers import Dense 

from keras.models import Sequential

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



test_size = 0.2

random_state=101

epochs=100

optimizer='sgd'

loss='categorical_crossentropy'

metrics=['accuracy']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state,stratify=y) #not sure if stratify=y works when there are more than 2 classes for the labels



#The no. of layers and no. of nodes for each layer can be freely customized

model = Sequential()

model.add(Dense(10,activation='relu',input_dim=X.shape[1])) #X.shape[1] should equal the number of features

model.add(Dense(10,activation='relu'))

model.add(Dense(len(np.unique(y_train)),activation='softmax')) #the number of the output layer should equal the number of unique outcomes (response variables)

        

model.compile(optimizer=optimizer,loss=loss,metrics=metrics)



model.fit(X_train, to_categorical(y_train),epochs=epochs,verbose=0)



y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)



accuracy=accuracy_score(y_test,y_pred)

print('Accuracy= ',accuracy)
#Hierarchical clustering

import scipy.cluster.hierarchy as shc



plt.figure(figsize=(20, 10))

plt.title("Iris Dendograms")

dend = shc.dendrogram(shc.linkage(X, method='ward'),labels=y)
#PCA

##ref: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

import matplotlib.pyplot as plt

import matplotlib.cm as mplcm

import matplotlib.colors as colors

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents)



finalDf = pd.concat([principalDf, pd.DataFrame(y)], axis = 1)

finalDf.columns = ['principal component 1', 'principal component 2','target']





fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



NUM_COLORS = len(np.unique(np.array(y)))

cm = plt.get_cmap('gist_rainbow')

cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)

scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)





targets = np.unique(y) 



for target in targets:

  indicesToKeep = (finalDf['target'] == target)

  ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], s = 50, alpha=0.3)

ax.legend(targets)

ax.grid()
# t-SNE

##ref: https://medium.com/@sourajit16.02.93/tsne-t-distributed-stochastic-neighborhood-embedding-state-of-the-art-c2b4b875b7da

from sklearn.manifold import TSNE 





# Module for standardization

from sklearn.preprocessing import StandardScaler

#Get the standardized data

standardized_data = StandardScaler().fit_transform(X)



model = TSNE(n_components=2) #n_components means the lower dimension



low_dim_data = pd.DataFrame(model.fit_transform(standardized_data))





finalDf = pd.concat([low_dim_data, pd.DataFrame(y)], axis = 1)

finalDf.columns = ['Dim 1', 'Dim 2','target']





fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Dim 1', fontsize = 15)

ax.set_ylabel('Dim 2', fontsize = 15)

ax.set_title('2 component tSNE', fontsize = 20)

targets = np.unique(y)



from itertools import cycle

cycol = cycle('bgrcmk')



for target in targets:

  indicesToKeep = (finalDf['target'] == target)

  ax.scatter(finalDf.loc[indicesToKeep, 'Dim 1'], finalDf.loc[indicesToKeep, 'Dim 2'], s = 50, color=next(cycol), alpha=0.3)

ax.legend(targets)

ax.grid()
!pip install minisom

# self-organizing map

##ref: https://rubikscode.net/2018/08/27/implementing-self-organizing-maps-with-python-and-tensorflow/

##ref: https://pypi.org/project/MiniSom/

##ref: https://github.com/JustGlowing/minisom

##ref: https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html

#!pip install minisom

from minisom import MiniSom

from numpy import genfromtxt,array,linalg,zeros,apply_along_axis



data = X

# normalization to unity of each pattern in the data

data = apply_along_axis(lambda x: x/linalg.norm(x),1,data)



from minisom import MiniSom

### Initialization and training ###

som = MiniSom(7,7,X.shape[1],sigma=1.0,learning_rate=0.5)

som.random_weights_init(data)

som.train_random(data,100) # training with 100 iterations



from pylab import plot,axis,show,pcolor,colorbar,bone

bone()

pcolor(som.distance_map().T) # distance map as background

colorbar()



# use different colors and markers for each label

markers = ['o','s','D']

colors = ['r','g','b']

for cnt,xx in enumerate(data):

 w = som.winner(xx) # getting the winner

 # palce a marker on the winning position for the sample xx

 plot(w[0]+.5,w[1]+.5,markers[y[cnt]],markerfacecolor='None',

   markeredgecolor=colors[y[cnt]],markersize=12,markeredgewidth=2)

show() # show the figure