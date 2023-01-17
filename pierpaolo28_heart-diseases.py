# from IPython.display import HTML

# HTML('''

# <script>

#   function code_toggle() {

#     if (code_shown){

#       $('div.input').hide('500');

#       $('#toggleButton').val('Show Code')

#     } else {

#       $('div.input').show('500');

#       $('#toggleButton').val('Hide Code')

#     }

#     code_shown = !code_shown

#   }



#   $( document ).ready(function(){

#     code_shown=true;

#   });

# </script>

# <form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Hide Code"></form>''')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv("../input/heart.csv")

#df = df.drop('Unnamed: 0', axis=1)

print(df.head())

print(df.shape)

print(df.columns)
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(x = 'target',data = df,hue = 'sex')

plt.title("Male (1) vs Female (0) affected by Heart Diseases")
plt.figure(figsize = (20,12))

sns.countplot(x = 'age',hue = 'target',data = df)

plt.title("People affected by heart deseases vs age", fontsize=20)

plt.legend(["Healthy","Ill"], fontsize=20)
plt.figure(figsize = (10,7))

plt.scatter(x = 'age',y = 'thalach', c='target',data = df)

plt.xlabel('Age')

plt.ylabel('Max heart rate')

plt.title('Heart rate vs Age')
import seaborn as sns



corr=df.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



X = df.drop(['target'], axis = 1).values

Y = df['target']



X = StandardScaler().fit_transform(X)



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
# Preprocessing :

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report,confusion_matrix

from itertools import product



# Classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn import tree

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
trainedmodel = LogisticRegression().fit(X_Train,Y_Train)

predictions =trainedmodel.predict(X_Test)

print(confusion_matrix(Y_Test,predictions))

print(classification_report(Y_Test,predictions))
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)

predictionforest = trainedforest.predict(X_Test)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))
trainedsvm = svm.LinearSVC().fit(X_Train, Y_Train)

predictionsvm = trainedsvm.predict(X_Test)

print(confusion_matrix(Y_Test,predictionsvm))

print(classification_report(Y_Test,predictionsvm))
trainedtree = tree.DecisionTreeClassifier().fit(X_Train, Y_Train)

predictionstree = trainedtree.predict(X_Test)

print(confusion_matrix(Y_Test,predictionstree))

print(classification_report(Y_Test,predictionstree))
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz



data = export_graphviz(trainedtree,out_file=None,feature_names=df.drop(['target'], axis = 1).columns,

                       class_names=['0', '1'],  

                       filled=True, rounded=True,  

                       max_depth=2,

                       special_characters=True)

graph = graphviz.Source(data)

graph
trainedlda = LinearDiscriminantAnalysis().fit(X_Train, Y_Train)

predictionlda = trainedlda.predict(X_Test)

print(confusion_matrix(Y_Test,predictionlda))

print(classification_report(Y_Test,predictionlda))
trainednb = GaussianNB().fit(X_Train, Y_Train)

predictionnb = trainednb.predict(X_Test)

print(confusion_matrix(Y_Test,predictionnb))

print(classification_report(Y_Test,predictionnb))
from xgboost import XGBClassifier

from xgboost import plot_tree

import matplotlib.pyplot as plt

model = XGBClassifier()



# Train

model.fit(X_Train, Y_Train)



plot_tree(model)

plt.figure(figsize = (50,55))

plt.show()
from itertools import product

import itertools



predictions =model.predict(X_Test)

print(confusion_matrix(Y_Test,predictions))

print(classification_report(Y_Test,predictions))



# Thanks to: https://www.kaggle.com/tejainece/data-visualization-and-machine-learning-algorithms

def plot_confusion_matrix(cm, classes=["0", "1"], title="",

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title('Confusion matrix ' +title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



cm_plot = confusion_matrix(Y_Test,predictions)



plt.figure()

plot_confusion_matrix(cm_plot, title = 'XGBClassifier')
pca = PCA(n_components=2,svd_solver='full')

X_pca = pca.fit_transform(X)

# print(pca.explained_variance_)



X_reduced, X_test_reduced, Y_Train, Y_Test = train_test_split(X_pca, Y, test_size = 0.30, random_state = 101)



# pca = PCA(n_components=2,svd_solver='full')

# X_reduced = pca.fit_transform(X_Train)

#X_reduced = TSNE(n_components=2).fit_transform(X_Train, Y_Train)



trainednb = GaussianNB().fit(X_reduced, Y_Train)

trainedsvm = svm.LinearSVC().fit(X_reduced, Y_Train)

trainedforest = RandomForestClassifier(n_estimators=700).fit(X_reduced,Y_Train)

trainedmodel = LogisticRegression().fit(X_reduced,Y_Train)



# pca = PCA(n_components=2,svd_solver='full')

# X_test_reduced = pca.fit_transform(X_Test)

#X_test_reduced = TSNE(n_components=2).fit_transform(X_Test, Y_Test)



print('Naive Bayes')

predictionnb = trainednb.predict(X_test_reduced)

print(confusion_matrix(Y_Test,predictionnb))

print(classification_report(Y_Test,predictionnb))



print('SVM')

predictionsvm = trainedsvm.predict(X_test_reduced)

print(confusion_matrix(Y_Test,predictionsvm))

print(classification_report(Y_Test,predictionsvm))



print('Random Forest')

predictionforest = trainedforest.predict(X_test_reduced)

print(confusion_matrix(Y_Test,predictionforest))

print(classification_report(Y_Test,predictionforest))



print('Logistic Regression')

predictions =trainedmodel.predict(X_test_reduced)

print(confusion_matrix(Y_Test,predictions))

print(classification_report(Y_Test,predictions))
reduced_data = X_reduced



trainednb = GaussianNB().fit(reduced_data, Y_Train)

trainedsvm = svm.LinearSVC().fit(reduced_data, Y_Train)

trainedforest = RandomForestClassifier(n_estimators=700).fit(reduced_data,Y_Train)

trainedmodel = LogisticRegression().fit(reduced_data,Y_Train)



# Thanks to: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html



x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1

y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))



f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))



for idx, clf, tt in zip(product([0, 1], [0, 1]),

                        [trainednb, trainedsvm, trainedforest, trainedmodel],

                        ['Naive Bayes Classifier', 'SVM',

                         'Random Forest', 'Logistic Regression']):



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    



    axarr[idx[0], idx[1]].contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)

    axarr[idx[0], idx[1]].scatter(reduced_data[:, 0], reduced_data[:, 1], c=Y_Train,

                                  s=20, edgecolor='k')

    axarr[idx[0], idx[1]].set_title(tt)



plt.show()
# Load libraries

from sklearn import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# Create an LDA that will reduce the data down to 1 feature

lda = LinearDiscriminantAnalysis(n_components=2)



# run an LDA and use it to transform the features

X_lda = lda.fit(X, Y).transform(X)



# Print the number of features

print('Original number of features:', X.shape[1])

print('Reduced number of features:', X_lda.shape[1])



## View the ratio of explained variance

print(lda.explained_variance_ratio_)



X_reduced, X_test_reduced, Y_Train, Y_Test = train_test_split(X_lda, Y, test_size = 0.30, random_state = 101)



trainednb = GaussianNB().fit(X_reduced, Y_Train)

trainedsvm = svm.LinearSVC().fit(X_reduced, Y_Train)



print('Naive Bayes')

predictionnb = trainednb.predict(X_test_reduced)

print(confusion_matrix(Y_Test,predictionnb))

print(classification_report(Y_Test,predictionnb))



print('SVM')

predictionsvm = trainedsvm.predict(X_test_reduced)

print(confusion_matrix(Y_Test,predictionsvm))

print(classification_report(Y_Test,predictionsvm))
from sklearn.manifold import TSNE

import time



time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
plt.figure(figsize=(6,5))

sns.scatterplot(

    x=tsne_results[:,0], y=tsne_results[:,1],

    hue=Y,

    palette=sns.color_palette("hls", 2),

    data=df,

    legend="full",

    alpha=0.3

)
pca = PCA(n_components=2,svd_solver='full')

X_pca = pca.fit_transform(X)

# print(pca.explained_variance_)



# print('Original number of features:', X.shape[1])

# print('Reduced number of features:', X_lda.shape[1])

print(pca.explained_variance_ratio_)



X_reduced, X_test_reduced, Y_Train, Y_Test = train_test_split(X_pca, Y, test_size = 0.30, random_state = 101)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced)
kpredictions = kmeans.predict(X_test_reduced)

print(confusion_matrix(Y_Test,kpredictions))

print(classification_report(Y_Test,kpredictions))
plt.scatter(X_test_reduced[kpredictions ==0,0], X_test_reduced[kpredictions == 0,1], s=100, c='red')

plt.scatter(X_test_reduced[kpredictions ==1,0], X_test_reduced[kpredictions == 1,1], s=100, c='black')
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering



# create dendrogram

dendrogram = sch.dendrogram(sch.linkage(X_reduced, method='ward'))

# create clusters

hc = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')

# save clusters for chart

hierarchicalpredictions = hc.fit_predict(X_test_reduced)
plt.scatter(X_test_reduced[hierarchicalpredictions ==0,0], X_test_reduced[hierarchicalpredictions == 0,1], s=100, c='red')

plt.scatter(X_test_reduced[hierarchicalpredictions ==1,0], X_test_reduced[hierarchicalpredictions == 1,1], s=100, c='black')