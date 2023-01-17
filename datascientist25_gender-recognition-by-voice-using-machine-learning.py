%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/voicegender/voice.csv')
df.columns
df.shape
df.dtypes
df.head(3)
df.isnull().values.any()
# Distribution of target varibles
colors = ['pink','Lightblue']
data_y = df[df.columns[-1]]
plt.pie(data_y.value_counts(),colors=colors,labels=['female','male'])
plt.axis('equal')
print (df['label'].value_counts())
# Box plot see comparision in labels by other features
df.boxplot(column = 'meanfreq',by='label',grid=False)
# plot correlation matrix (here i used the headmape using seaborn)
correlation =df.corr()
sns.heatmap(correlation)
plt.show()
# train_test_split is responsible to split the data into (Train and Test)
from sklearn.model_selection import train_test_split
X = df[df.columns[:-1]].values
#y = df['lable']
y = df[df.columns[-1]].values
# We will divide the data into 70-30% into train and test data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30)
import warnings
warnings.filterwarnings("ignore")
# Random forest
# Importing Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
rand_forest = RandomForestClassifier()
rand_forest.fit(Xtrain, ytrain)
y_pred = rand_forest.predict(Xtest)

#import matrix to calcualte accuracy
from sklearn import metrics, neighbors
from sklearn.metrics import accuracy_score
print(metrics.accuracy_score(ytest, y_pred))
# Import confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, y_pred))
# 10-fold cross validation
# Importing Bausian Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
CVFirst = GaussianNB()
CVFirst = CVFirst.fit(Xtrain, ytrain)
test_result = cross_val_score(CVFirst, X, y, cv=10, scoring='accuracy')
print('Accuracy obtained from 10-fold cross validation is:',test_result.mean())
male_funFreq_outlier_index = df[((df['meanfun'] < 0.085) | (df['meanfun'] > 0.180)) &
                               (df['label'] == 'male')].index
female_funFreq_outlier_index = df[((df['meanfun'] < 0.165)  | (df['meanfun'] > 0.255)) &
                                 (df['label'] == 'female')].index
index_to_remove = list(male_funFreq_outlier_index) + list(female_funFreq_outlier_index)
len(index_to_remove)
data_x = df[df.columns[0:20]].copy()
data2 = data_x.drop(['kurt','centroid','dfrange'],axis=1).copy()
data2.head(3)
data2 = data2.drop(index_to_remove,axis=0)

#y = df['lable']
data_y = pd.Series(y).drop(index_to_remove,axis=0)
Xtrain, Xtest, ytrain, ytest = train_test_split(data2, data_y, test_size=0.30 )
clf1 = RandomForestClassifier()
clf1.fit(Xtrain, ytrain)
y_pred = clf1.predict(Xtest)
print(metrics.accuracy_score(ytest, y_pred))
# Importing Decision Trees Classifier
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier()
clf2.fit(Xtrain, ytrain)
y_predict = clf2.predict(Xtest)
print(metrics.accuracy_score(ytest, y_predict))
clf3 = GaussianNB()
clf3 = clf3.fit(Xtrain, ytrain)
y_predd = clf3.predict(Xtest)
print(metrics.accuracy_score(ytest,y_predd))
# Importing linear Regression classifier
from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression()
clf4.fit(Xtrain,ytrain)
y_predict4 = clf4.predict(Xtest)
print(metrics.accuracy_score(ytest,y_predict4))
# cross validation with same classifire as first time 
test_result = cross_val_score(clf3, data2, data_y, cv=10, scoring='accuracy')
print('Accuracy obtained from 10-flod cross validation is:',test_result.mean())
# cros validation on the best result
test_result = cross_val_score(clf2, data2, data_y, cv=10,scoring = 'accuracy')
print('Accuracy obtained from 10-fold validation is:',test_result.mean())
import pylab as pl
labels = ['female', 'male']
cm = confusion_matrix(ytest,y_pred,labels)  #ypred for RandomForest
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax =ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()
from sklearn.metrics import classification_report
print(classification_report(ytest, y_pred))
# Distribution of male and female
sns.FacetGrid(df, hue='label',size=5).map(sns.kdeplot,"meanfun").add_legend()
plt.show()
#Since we're doing flat-clustering , our task is a bit easier since we can tell the machine that we want it category

from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")

data_x = np.array(df[['meanfreq','meanfun']])
kmeans = KMeans(n_clusters= 2)
kmeans.fit(data_x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#print(centroids)
#print(labels) # 0-male, 1-Female( the machine has assigned on its own.)

colors = ["g.","b."]  #green = male

for i in range(len(data_x)):
    plt.plot(data_x[i][0], data_x[i][1], colors[labels[i]], markersize = 10)

    
plt.scatter(centroids[:,0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.ylabel('meanfun')
plt.xlabel('meanfun')

plt.show()