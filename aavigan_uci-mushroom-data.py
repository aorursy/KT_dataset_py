#import modules

import pandas as pd

import numpy as np



#read csv to pandas dataframe

mushrooms = pd.read_csv("../input/mushrooms.csv")



#create dummy variables

mushrooms = pd.get_dummies(mushrooms)



#subset data into dependent and independent variables x,y

LABELS = ['class_e', 'class_p']

FEATURES = [a  for a in mushrooms.columns if a not in LABELS ]

y = mushrooms[LABELS[0]]

x= mushrooms[FEATURES]



mushrooms.head()
#import modules

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE



#create model

model = TSNE(learning_rate = 100)



#fit model

transformed = model.fit_transform(x.values)

xs = transformed[:,0]

ys = transformed[:,1]

df_trans = pd.DataFrame({'xs':xs, 'ys':ys})



#create plots

plt.scatter(df_trans.loc[y==0]['xs'], df_trans.loc[y ==0]['ys'], c= 'tab:green')

plt.scatter(df_trans.loc[y ==1]['xs'], df_trans.loc[y ==1]['ys'], c= 'tab:blue')

plt.legend(loc ='lower left', labels = ['p', 'e'])



plt.show()
#import modules

from sklearn.linear_model import Lasso



#create model, set alpha

lasso = Lasso(alpha = 0.1)



#fit model, access coef_ attribute: lasso_coef

lasso_coef = lasso.fit(x,y).coef_



#create plot

_=plt.plot(range(len(x.columns)), lasso_coef)

plt.show()
#import modules

from sklearn.linear_model import Lasso



#create model, set alpha

lasso = Lasso(alpha = 0.001)



#fit model, access coef_ attribute: lasso_coef

lasso_coef = lasso.fit(x,y).coef_



#create plot

_=plt.plot(range(len(x.columns)), lasso_coef)

plt.show()
#import modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report



#create parameter grid for values of k

parameters = {'knn__n_neighbors': np.arange(1,10)}



#instatiate pipeline with KNNClassifier: pl

pl = Pipeline([('knn', KNeighborsClassifier())])



#split data into test and training data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)



#instatiate GridsearchCV:cv

cv = GridSearchCV(pl, param_grid= parameters, cv = 3)



#fit model to training data

cv.fit(X_train, y_train)



#predict test data: y_pred

y_pred = cv.predict(X_test)

#print performance metrics

print (cv.best_params_)

print(classification_report(y_test, y_pred))
#import modules

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.tree import export_graphviz





#instantiate DecisionTreeClassifier:tree

tree = DecisionTreeClassifier()



#split data into test and training data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)



#fit data

tree.fit(X_train, y_train)



#predict test data:y_pred

y_pred=tree.predict(X_test)



#print performance metrics

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))



import graphviz



#export a graphic representation of tree to file

dot_data =export_graphviz(tree, out_file = None, feature_names =x.columns, class_names = ['edible', 'poisonous'])

graph = graphviz.Source(dot_data)

graph
#import modules

from sklearn.cluster import KMeans

from sklearn import metrics



#instantaiate lists

xp=[]

yp=[]



# fit KMeans model for n_clusters 2:30

for i in range(2,30):

    model = KMeans(n_clusters = i, random_state = 42)

    model.fit(x.values)

    labels = model.predict(x)

    scores =metrics.homogeneity_completeness_v_measure(y, labels)

    xp.append(i)

    yp.append(scores[0])

plt.plot(xp,yp)

plt.title('Kmeans Clustering')

plt.xlabel(s = 'Number of Clusters')

plt.ylabel(s = 'Homogeneity Score')

plt.show()
#import modules

from scipy.cluster.hierarchy import linkage,  fcluster



#instantiate linkage with 'average'

merging = linkage(x, method = 'average')



#create empty lists

xp=[]

yp=[]



#modify distance threshold of clustering and append cluster number and homogeneity scores to xp, yp

for i in range(30):

    labels = fcluster(merging,6-i*.1, criterion= 'distance')

    scores =metrics.homogeneity_completeness_v_measure(y, labels)

    xp.append(len(np.unique(labels)))

    yp.append(scores[0])



#plot number of cluster vs homogeneity score

plt.plot(xp,yp)



plt.title('Hierarchical Clustering - Average')

plt.xlabel(s = 'Number of Clusters')

plt.ylabel(s = 'Homogeneity Score')



plt.show()
#instantiate linkage with 'centroid'

merging = linkage(x, method = 'centroid')



#create empty lists

xp=[]

yp=[]



#modify distance threshold of clustering and append cluster number and homogeneity scores to xp, yp

for i in range(40):

    labels = fcluster(merging,6-i*.1, criterion= 'distance')

    scores =metrics.homogeneity_completeness_v_measure(y, labels)

    xp.append(len(np.unique(labels)))

    yp.append(scores[0])



#plot number of cluster vs homogeneity score

plt.plot(xp,yp)



plt.title('Hierarchical Clustering - Centroid')

plt.xlabel(s = 'Number of Clusters')

plt.ylabel(s = 'Homogeneity Score')

plt.show()
#create species labels using optimal clustering parameters

species = fcluster(merging,2.3, criterion= 'distance')



#transform mushroom data using-tsne

model = TSNE(learning_rate = 100)

transformed = model.fit_transform(x.values)

xs = transformed[:,0]

ys = transformed[:,1]

df_trans = pd.DataFrame({'xs':xs, 'ys':ys})



#determine centroid locations of t-sne clusters

cpx =[]

cpy=[]



for i in range(1,24):

    xi = df_trans.loc[(species ==i)]['xs'].mean()

    yi = df_trans.loc[(species ==i)]['ys'].mean()

    cpx.append(xi)

    cpy.append(yi)



#plot edible and poisonous samples with different colors

plt.scatter(df_trans.loc[y==0]['xs'], df_trans.loc[y ==0]['ys'], c= 'tab:green' )

plt.scatter(df_trans.loc[y ==1]['xs'], df_trans.loc[y ==1]['ys'], c= 'tab:blue')



#annotate clusters centroids for each species

for i in range(1,24):

    plt.annotate(s = str(i), xy = (cpx[i-1], cpy[i-1]), xytext = (-4,-4), textcoords = 'offset points')



#insert legend with labels 'p', 'e'

plt.legend(loc ='lower left', labels = ['p', 'e'])

plt.show()