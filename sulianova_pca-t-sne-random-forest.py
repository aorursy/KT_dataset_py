# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import decomposition

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv("../input/data.csv", index_col = 'id')

df.drop('Unnamed: 32',axis = 1 ,inplace = True)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B':0})

X = df.drop('diagnosis',axis = 1)
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)



pca = decomposition.PCA(n_components=2)

X_pca_scaled = pca.fit_transform(X_scaled)



print('Projecting %d-dimensional data to 2D' % X_scaled.shape[1])



plt.figure(figsize=(12,10))

plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=df['diagnosis'], alpha=0.7, s=40);

plt.colorbar()

plt.title('MNIST. PCA projection');
# Invoke the TSNE method

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000,random_state = 17)



df_tsne_scaled = tsne.fit_transform(X_scaled)



plt.figure(figsize=(12,10))

plt.scatter(df_tsne_scaled[:, 0], df_tsne_scaled[:, 1], c=df['diagnosis'], 

            alpha=0.7, s=40)

plt.colorbar()

plt.title('MNIST. t-SNE projection');
pca = decomposition.PCA().fit(X_scaled)



plt.figure(figsize=(10,7))

plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)

plt.xlabel('Number of components')

plt.ylabel('Total explained variance')

plt.xlim(0, 29)

plt.yticks(np.arange(0, 1.1, 0.1))

plt.axvline(6, c='b')

plt.axhline(0.91, c='r')

plt.show();
perimeters = [x for x in df.columns if 'perimeter' in x]

areas = [x for x in df.columns if 'area' in x]

df.drop(perimeters, axis = 1 ,inplace = True)

df.drop(areas, axis = 1 ,inplace = True)

worst = [col for col in df.columns if col.endswith('_worst')]

df.drop(worst, axis = 1 ,inplace = True)
X = df.drop(['diagnosis'], axis=1)

(X+0.001).hist(figsize=(20, 15), color = 'c');
#Log transformation

X = df.drop(['diagnosis'], axis=1)

X_log = np.log(X+0.001)

X_log.hist(figsize=(20, 15), color = 'c');
from sklearn.model_selection import train_test_split



#Scaler should be trained on train set only to prevent information about future from leaking.



y = df['diagnosis']



X_log_train, X_log_holdout, y_train, y_holdout = train_test_split(X_log, y, test_size=0.3, random_state=17)
from sklearn.model_selection import GridSearchCV



X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)



tree = DecisionTreeClassifier(random_state=17)



tree_params = {'max_depth': range(1,5), 'max_features': range(3,6), 'criterion': ['gini','entropy']}



tree_grid = GridSearchCV(tree, tree_params, cv=10, scoring='recall')

tree_grid.fit(X_train, y_train)
tree_grid.best_params_, tree_grid.best_score_
from sklearn.metrics import accuracy_score, recall_score, precision_score



tree_pred = tree_grid.predict(X_holdout)



print ("Accuracy Score : ",accuracy_score(y_holdout, tree_pred) )

print ("Recall Score (how much of malignant tumours were predicted correctly) : ",recall_score(y_holdout, tree_pred))

print ("Precision Score (how much of tumours, which were predicted as 'malignant', were actually 'malignant'): ",precision_score(y_holdout, tree_pred))
from sklearn.metrics import confusion_matrix



confusion_matrix(y_holdout, tree_pred)
from sklearn.tree import export_graphviz

tree_graph = export_graphviz(tree_grid.best_estimator_, class_names = ['benign', 'malignant'], feature_names = df.drop(['diagnosis'], axis=1).columns, filled=True, out_file='tree.dot')

!dot -Tpng tree.dot -o tree.png 
from IPython.display import Image

Image(filename = 'tree.png')
numerical = df.drop('diagnosis',axis=1).columns



df.groupby(['diagnosis'])[numerical].agg([np.mean, np.std, np.min, np.max])
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression



Cs = np.logspace(-1, 8, 5)



lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=17,solver='liblinear'))])



lr_params = {'lr__C': Cs}



lr_pipe_grid = GridSearchCV(lr_pipe, lr_params, cv=10, scoring='recall')

lr_pipe_grid.fit(X_log_train, y_train)
lr_pipe_grid.best_params_, lr_pipe_grid.best_score_
scores=[]

for C in Cs:

    pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(C=C, random_state=17,solver='liblinear'))])

    scores.append(cross_val_score(pipe,X_log_train, y_train,cv=10, scoring='recall').mean())
score_C_1 = lr_pipe_grid.best_score_

sns.set()

plt.figure(figsize=(10,8))

plt.plot(Cs, scores, 'ro-')

plt.xscale('log')

plt.xlabel('C')

plt.ylabel('Recall')

plt.title('Regularization Parameter Tuning')

# horizontal line -- model quality with default C value

plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed') 

plt.show()
print ("Accuracy Score : ",accuracy_score(y_holdout, lr_pipe_grid.predict(X_log_holdout)) )

print ("Recall Score (how much of malignant tumours were predicted correctly) : ",recall_score(y_holdout, lr_pipe_grid.predict(X_log_holdout)))

print ("Precision Score (how much of tumours, which were predicted as 'malignant', were actually 'malignant'): ",precision_score(y_holdout, lr_pipe_grid.predict(X_log_holdout)))
lr_best_pipe = lr_pipe_grid.best_estimator_.named_steps['lr']



#Create Data frame of Regression coefficients

coef= pd.DataFrame(lr_best_pipe.coef_.ravel())

#Merge Regression coefficients with feature names

df_columns = pd.DataFrame(df.drop(['diagnosis'], axis=1).columns)

coef_and_feat = pd.merge(coef,df_columns,left_index= True,right_index= True, how = "left")

coef_and_feat.columns = ["coefficients","features"]

coef_and_feat = coef_and_feat.sort_values(by = "coefficients",ascending = False)



#Set up the matplotlib figure

plt.rcParams['figure.figsize'] = (10,8)

# Let's draw top 10 important features 

sns.barplot(x = 'features', y = 'coefficients', data = coef_and_feat).set_title('Feature importance')

plt.xticks(rotation=45);
C_scores = np.logspace(-1, 8, 5)



lr = LogisticRegression(random_state=17,solver='liblinear')



lr_params = {'C': C_scores}



lr_grid = GridSearchCV(lr, lr_params, cv=10, scoring='recall')

lr_grid.fit(X_train, y_train)
lr_grid.best_params_, lr_grid.best_score_
print ("Accuracy Score : ",accuracy_score(y_holdout, lr_grid.predict(X_log_holdout)) )

print ("Recall Score (how much of malignant tumours were predicted correctly) : ",recall_score(y_holdout, lr_grid.predict(X_log_holdout)))

print ("Precision Score (how much of tumours, which were predicted as 'malignant', were actually 'malignant'): ",precision_score(y_holdout, lr_grid.predict(X_log_holdout)))
lr_best= lr_grid.best_estimator_



#Create Data frame of Regression coefficients

coef= pd.DataFrame(lr_best.coef_.ravel())

#Merge Regression coefficients with feature names

df_columns = pd.DataFrame(df.drop(['diagnosis'], axis=1).columns)

coef_and_feat = pd.merge(coef,df_columns,left_index= True,right_index= True, how = "left")

coef_and_feat.columns = ["coefficients","features"]

coef_and_feat = coef_and_feat.sort_values(by = "coefficients",ascending = False)



#Set up the matplotlib figure

plt.rcParams['figure.figsize'] = (10,8)

# Let's draw top 10 important features 

sns.barplot(x = 'features', y = 'coefficients', data = coef_and_feat).set_title('Feature importance')

plt.xticks(rotation=90);
from sklearn.ensemble import RandomForestClassifier



#Stratified split for the validation process

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)



#initialize the set of parameters for exhaustive search and fit to find out the optimal parameters

rfc_params = {'max_features': range(1,11), 'min_samples_leaf': range(1,3), 'max_depth': range(3,13), 'criterion':['gini','entropy']}



rfc = RandomForestClassifier(n_estimators=100, random_state=17, n_jobs= -1)



gcv = GridSearchCV(rfc, rfc_params, n_jobs=-1, cv=skf, scoring='recall')



gcv.fit(X_train, y_train)
gcv.best_params_, gcv.best_score_
#RandomForest classifier with the default parameters 

rfc = RandomForestClassifier(n_estimators=100, criterion ='gini', max_depth = 8, max_features = 6, min_samples_leaf = 1, random_state = 17, n_jobs=-1)

forest_pred = gcv.predict(X_holdout)



print ("Accuracy Score : ",accuracy_score(y_holdout, forest_pred) )

print ("Recall Score (how much of malignant tumours were predicted correctly) : ",recall_score(y_holdout, forest_pred))

print ("Precision Score (how much of tumours, which were predicted as 'malignant', were actually 'malignant'): ",precision_score(y_holdout, forest_pred))
rfc = gcv.best_estimator_

estimators_tree_98 = rfc.estimators_[98]



estimators_tree_3 = rfc.estimators_[3]



estimators_tree_47 = rfc.estimators_[47]
estimators_tree_3.n_features_
tree_graph_98 = export_graphviz(estimators_tree_98, class_names = ['benign', 'malignant'], feature_names = df.drop(['diagnosis'], axis=1).columns, filled=True, out_file='tree_98.dot')

!dot -Tpng tree_98.dot -o tree_98.png 



tree_graph_3 = export_graphviz(estimators_tree_3, class_names = ['benign', 'malignant'], feature_names = df.drop(['diagnosis'], axis=1).columns, filled=True, out_file='tree_3.dot')

!dot -Tpng tree_3.dot -o tree_3.png 



tree_graph_47 = export_graphviz(estimators_tree_47, class_names = ['benign', 'malignant'], feature_names = df.drop(['diagnosis'], axis=1).columns, filled=True, out_file='tree_47.dot')

!dot -Tpng tree_47.dot -o tree_47.png 
Image(filename = 'tree_98.png')



Image(filename = 'tree_3.png')



Image(filename = 'tree_47.png')
rf_pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, random_state=17, n_jobs= -1))])



rf_params = {'rf__max_features': range(3,10), 'rf__min_samples_leaf': range(1,3), 'rf__max_depth': range(5,12), 'rf__criterion':['gini','entropy']}





rf_pipe_grid = GridSearchCV(rf_pipe, rf_params, cv=10, scoring='recall')

rf_pipe_grid.fit(X_log_train, y_train)
rf_pipe_grid.best_params_, rf_pipe_grid.best_score_
print ("Accuracy Score on scaled data: ",accuracy_score(y_holdout, rf_pipe_grid.predict(X_log_holdout)) )

print ("Recall Score (how much of malignant tumours were predicted correctly) : ",recall_score(y_holdout, rf_pipe_grid.predict(X_log_holdout)))

print ("Precision Score (how much of tumours, which were predicted as 'malignant', were actually 'malignant'): ",precision_score(y_holdout, rf_pipe_grid.predict(X_log_holdout)))