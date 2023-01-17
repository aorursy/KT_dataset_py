# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import sklearn.model_selection

import matplotlib.pyplot as plt

import seaborn as sb

import warnings

warnings.filterwarnings('ignore')  # change once to ignore when publishing 

from sklearn.metrics import accuracy_score



rituximab = pd.read_csv("../input/rituximab.csv")

rit=rituximab.copy()

rit.columns=['FSCH','SSCH','FL1H','FL2H','FL3H','FL1A','FL1W','Time','Gate']

rit.head(5)



rit1=rit.drop('Time',axis=1)

rit1=rit1[rit1['Gate']!=-1]

rit1.loc[:, "Gate"] = rit1.loc[:, "Gate"].map({1: 0, 2: 1})

print("The dataset contains ",rit1.isnull().sum().sum()," null values")

print('The first 10 elements of the cleaned dataset are shown below')

rit1.head(10)



from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(rit1, test_size=0.2, random_state=42)

train_labels=train_set.copy()['Gate']

test_labels=test_set.copy()['Gate']

test_no_labels=test_set.copy().drop('Gate',axis=1)

print('The train set has ',train_set.shape[0],' data points')

print('The test set has ',test_set.shape[0],' data points')









train_set.hist(bins=25, figsize=(20,15))



plt.show()

ax=sb.pairplot(train_set,hue='Gate',plot_kws={'alpha': 0.3})

ax.fig.set_size_inches(20, 20);

fig,ax =plt.subplots()

fig.set_size_inches(11.7, 8.27)

ax = sb.scatterplot(x="FSCH", y="SSCH", hue="Gate",  data=train_set,alpha=0.3)

ax.set_title("Scattering Parameters");
fig,ax =plt.subplots()

fig.set_size_inches(11.7, 8.27)

ax = sb.scatterplot(x="FL1H",y=1, hue="Gate",  data=train_set,alpha=0.3)

ax.set_title("FL1H Parameter");



ax.set_yticklabels([]);




#correlation matrix

corrmat = train_set.corr()

f, ax = plt.subplots(figsize=(12, 9))

sb.heatmap(corrmat, vmax=.8, square=True,cmap='YlGnBu');



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

train_no_labels=train_set.drop('Gate',axis=1)

principalComponents = pca.fit_transform(train_no_labels)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['pc1', 'pc2','pc3'])

finalDf = pd.concat([principalDf, train_set['Gate']], axis = 1)



s=pca.explained_variance_ratio_.sum()

print('The first 3 principal components explain',round(s*100,2),'% of the variance of the data')
















ax=sb.pairplot(finalDf,hue='Gate',plot_kws={'alpha': 0.3},vars=['pc1','pc2','pc3'])

ax.fig.set_size_inches(20, 20)







from sklearn.cluster import KMeans

Kmean = KMeans(n_clusters=2)

Kmean.fit(train_no_labels)

kmlabels=pd.DataFrame(Kmean.labels_)

tkl=train_set.copy()

kmlabels = kmlabels.set_index(train_set.index)

tkl['km']=kmlabels

# Running k means on separate occasions may yield different values for the labels, in this case 0 or 1

# The following is a probably bad way of setting the output of the k means labelling to be the one that I want so the graph labels are correct





def g(row):

    if (row['km'] == 0):

        val = 1

    elif (row['km']==1):

        val = 0

   

    return val



if((tkl['km'].iloc[0]!=0) and (tkl['km'].iloc[1]!=0)):

    tkl['km']=tkl.apply(g,axis=1)



def f(row):

    if (row['Gate'] == row['km'] and row['Gate']==0):

        val = 'Gate 0'

    elif (row['Gate'] == row['km'] and row['Gate']==1):

        val = 'Gate 1'

    elif row['Gate'] != row['km']:

        val = 'Misclassified'

    

    return val



tkl['colorlabel'] = tkl.apply(f, axis=1)



tkl.head()

print('The Scatter plot of the K Means clustering is shown below\n The green points correspond to points that were assigned to the wrong class')
ax=sb.pairplot(tkl,hue='colorlabel',plot_kws={'alpha': 0.3},vars=['FSCH', 'SSCH', 'FL1H', 'FL2H', 'FL3H', 'FL1A', 'FL1W','Gate'])

ax.fig.set_size_inches(20, 20);

ax=sb.pairplot(data=tkl,hue='colorlabel',vars=['FL1H'])

ax.fig.set_size_inches(12, 8);

from sklearn.metrics import adjusted_rand_score

adjrand=adjusted_rand_score(tkl['Gate'],tkl['km'])

rand=(tkl['colorlabel'].value_counts()[0]+tkl['colorlabel'].value_counts()[1])/(tkl['colorlabel'].value_counts().sum())

print('The Rand Index is ',round(rand,4))

print('The Adjusted Rand Index is ',round(adjrand,4))


from pandas.plotting import radviz

fig = plt.figure( )

fig.set_size_inches(12,10)



tr=tkl.drop(['Gate','colorlabel'],axis=1)

tr.head()

rad_viz = pd.plotting.radviz(tr, 'km',color=['blue','orange'],alpha=0.5)



from sklearn.cluster import AgglomerativeClustering

s=AgglomerativeClustering(n_clusters=2,linkage='single')

s.fit(train_no_labels)

rs=adjusted_rand_score(tkl['Gate'],s.labels_)

print('Adjusted Rand Index for Hierarchical Clustering with single linkage is ',round(rs,4))



a=AgglomerativeClustering(n_clusters=2,linkage='average')

a.fit(train_no_labels)

ra=adjusted_rand_score(tkl['Gate'],a.labels_)

print('Adjusted Rand Index for Hierarchical Clustering with average linkage is ',round(ra,4))



c=AgglomerativeClustering(n_clusters=2,linkage='complete')

c.fit(train_no_labels)

rc=adjusted_rand_score(tkl['Gate'],c.labels_)

print('Adjusted Rand Index for Hierarchical Clustering with complete linkage is ',round(rc,4))



w=AgglomerativeClustering(n_clusters=2,linkage='ward')

w.fit(train_no_labels)

rw=adjusted_rand_score(tkl['Gate'],w.labels_)

print('Adjusted Rand Index for Hierarchical Clustering with ward linkage is ',round(rw,4))
label=train_set.copy().pop('Gate')

lut = dict(zip(label.unique(), ['blue','orange']))

row_colors = label.map(lut)



sb.set(rc={'figure.figsize':(12,8)})

g = sb.clustermap(train_no_labels, col_cluster=False,method='ward',row_colors=row_colors)

plt.title('Hierarchical Clustering with ward linkage');
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

param_grid={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}

knn=KNeighborsClassifier()

grid_search=GridSearchCV(knn,param_grid,cv=5,scoring='accuracy',return_train_score=True)

grid_search.fit(train_no_labels,train_labels);
c=grid_search.cv_results_

print ('10 K Nearest Neighbors classifiers were trained on 5 stratified subsets of the data and evaluted on the part that was not used for training\n')

print('Evaluating K Nearest Neighbor Classifier predictions...\n')

print ('The mean accuracy score for the 10 classifiers on the 5 folds are shown below\n')



for mean_score,params in zip(c['mean_test_score'],c['params']):

    print (round(mean_score,5),params)

print ('The optimal model has mean score',round(c['mean_test_score'][4],4))
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

from sklearn.linear_model import SGDClassifier

log_clf=SGDClassifier(loss='log',random_state=42)

cv=cross_validate(log_clf,train_no_labels,train_labels,scoring='accuracy',cv=StratifiedKFold(5))

s=0



print('Evaluating Logistic Regression Model predictions...\n')

for i,x in enumerate(cv['test_score']):

    s+=x

    print('The accuracy of the logistic classifier on fold',i+1,'is',round(x,5))



print('\nThe mean of the accuracy of the model on the 5 folds is',round(s/5,4))



    



from sklearn.tree import DecisionTreeClassifier

param_grid={'max_depth':[2,4,6,8,10]}

tree=DecisionTreeClassifier()

grid_search=GridSearchCV(tree,param_grid,cv=5,scoring='accuracy',return_train_score=True)

grid_search.fit(train_no_labels,train_labels);
c=grid_search.cv_results_

print ('5 Decision Tree Classifiers were trained on 5 stratified subsets of the data and evaluted on the part that was not used for training\n')

print('Evaluating Decison Tree Classifier predictions...\n')

print ('The mean accuracy score for the 5 classifiers on the 5 folds are shown below\n')



for mean_score,params in zip(c['mean_test_score'],c['params']):

    print (round(mean_score,5),params)

print('\nThe optimal depth of the decision tree with the settings chosen is',grid_search.best_params_['max_depth'])

import tensorflow as ts

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Activation, Dense





model = Sequential()

model.add(Dense(12, input_dim=7, activation='relu'))



model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(train_no_labels, train_labels, epochs=100, batch_size=10,verbose=0)

print('Training Multi Layer Perceptron...')

model.evaluate(train_no_labels,train_labels)

print('Evaluating model on test set')

y_pred=model.predict_classes(test_no_labels)

mlp_score=accuracy_score(test_labels,y_pred)



print('The Accuracy of the Multi Layer Perceptron on the test set is',round(mlp_score,4))
from sklearn.ensemble import RandomForestClassifier

forest_clf=RandomForestClassifier(n_estimators=100,max_depth=4);

forest_clf.fit(train_no_labels,train_labels);

print('Evaluating Random Forest Classifier predictions on the training set...\n')

print('The accuracy of the Random Forest Classifier with 100 trees on the training set is',round(forest_clf.score(train_no_labels,train_labels),4))







from sklearn.ensemble import VotingClassifier

voting_clf=VotingClassifier(estimators=[('for',forest_clf),('log',SGDClassifier(loss='log',random_state=42)),('knn',KNeighborsClassifier(n_neighbors=5))],voting='hard')

voting_clf.fit(train_no_labels,train_labels);

print('Evaluating Voting Classifier predictions on the training set...\n')

print('The accuracy of the Voting Classifier on the training set is',round(voting_clf.score(train_no_labels,train_labels),4))








y_test = np.asarray(test_labels)







knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(train_no_labels,train_labels)





print('Evaluating models on test set...\n')





    



knn.fit(train_no_labels,train_labels)

kpred_labels=knn.predict(test_no_labels)

kmis= np.where(y_test != knn.predict(test_no_labels))





print('Accuracy score for KNeighbors Classifier on test set is',round(100*accuracy_score(test_labels,kpred_labels),4),'%')



log_clf.fit(train_no_labels,train_labels)

lpred_labels=log_clf.predict(test_no_labels)

lmis= np.where(y_test != log_clf.predict(test_no_labels))



print('Accuracy score for Logistic Regression model on test set is',round(100*accuracy_score(test_labels,lpred_labels),4),'%')







print('Accuracy score for Multi Layer Perceptron on test set is',round(100*mlp_score,4),'%')







voting_clf.fit(train_no_labels,train_labels)

vpred_labels=voting_clf.predict(test_no_labels)

vmis= np.where(y_test != voting_clf.predict(test_no_labels))



print('Accuracy score for Voting Classifier on test set is',round(100*accuracy_score(test_labels,vpred_labels),4),'%')



forest_clf.fit(train_no_labels,train_labels)

fpred_labels=forest_clf.predict(test_no_labels)

fmis= np.where(y_test != forest_clf.predict(test_no_labels))





print('Accuracy score for Random Forest Classifier on test set is',round(accuracy_score(test_labels,fpred_labels)*100,4),'%')





misclassified=set()

misclassified.update(vmis[0])

misclassified.update(kmis[0])

misclassified.update(fmis[0])

misclassified   #This set contains indices of misclassified elements in y_test

                #Must retrieve indices from test_labels

    

misclassified_t_indices=set()

for val in misclassified:

    misclassified_t_indices.add(test_labels.index[val])

    



print('Indices of misclassified points by the models are',misclassified_t_indices,'\n')     #These are the indices where the cell is misclassified by the f,k,v models in test_set





def get_sigma_fl1h(idx):

    gate=rit1[rit1.index==idx]['Gate'].values[0]

    classified_gate=1-gate

    

    mean=rit1[rit1['Gate']==gate]['FL1H'].mean()

    classified_mean=rit1[rit1['Gate']==classified_gate]['FL1H'].mean()

    

    std=rit1[rit1['Gate']==gate]['FL1H'].std()

    cls_std=rit1[rit1['Gate']==classified_gate]['FL1H'].std()

    val=rit1.loc[idx]['FL1H']

    return (val-mean)/std,(val-classified_mean)/cls_std



for i,val in enumerate(misclassified_t_indices):

    print ('Misclassified data point',i+1,'is',get_sigma_fl1h(val)[0],'standard deviations away from the mean of the FL1H parameter of it\'s correct gate\n')

    print ('Misclassified data point',i+1,'is',get_sigma_fl1h(val)[1],'standard deviations away from the mean of the FL1H parameter of it\'s classified gate\n')