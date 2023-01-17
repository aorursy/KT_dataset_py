import numpy as np

import pandas as pd

import os

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

sns.set_style('darkgrid')

%matplotlib inline
data=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head() 
data['Unnamed: 32']
target=pd.Series(data['diagnosis'], name='diagnosis') 

data.drop(columns=['diagnosis', 'id', 'Unnamed: 32'], inplace=True) #dropping unnecesary columns
print(f'Number of Benigns: {target.value_counts()[0]}')

print(f'Number of Malignants: {target.value_counts()[1]}')





X_train, X_test, y_train, y_test=train_test_split(data, target, test_size=0.3, random_state=42)
X_train.describe()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
columns=data.columns

X_train_scaled_np=scaler.fit_transform(X_train)
X_train_scaled_df=pd.DataFrame(X_train_scaled_np, columns=columns)
X_train_scaled_df.describe()
data_scaled_df=pd.concat((X_train_scaled_df, target), axis=1)
X_train_vis=pd.melt(data_scaled_df, id_vars='diagnosis', var_name='features', value_name='value')
figure=plt.figure(figsize=(18,18))

g1=sns.violinplot(x='features', y='value', hue='diagnosis', inner='quartile', data=X_train_vis, split=True)

g1.set_xticklabels(g1.get_xticklabels(), rotation=90);

figure=plt.figure(figsize=(18,18))

g1=sns.boxplot(x='features', y='value', hue='diagnosis',  data=X_train_vis)

g1.set_xticklabels(g1.get_xticklabels(), rotation=90);
figure=plt.figure(figsize=(18,18))

g1=sns.swarmplot(x='features', y='value', hue='diagnosis',  data=X_train_vis)

g1.set_xticklabels(g1.get_xticklabels(), rotation=90);
figure=plt.figure(figsize=(18,18))

g1=sns.heatmap(X_train.corr(), annot=True, fmt='.1f', cbar=False)
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV

from sklearn.ensemble import RandomForestClassifier
select_feature = SelectKBest(chi2, k='all').fit(X_train, y_train)
select_feature.scores_
figure_importances=pd.Series(data=select_feature.scores_, index=X_train.columns, name='f_imp')

figure=plt.figure(figsize=(14,6))

g1=sns.barplot(y=figure_importances.index, x=figure_importances.values, orient='h')

list_col_drop=['radius_mean', 'perimeter_mean', 'radius_se', 'perimeter_se', 'compactness_mean', 'radius_worst', 'perimeter_worst', 'concavity_worst']
X_train_reduced=X_train.drop(columns=list_col_drop)
rf_clf_fs = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=52)

param_distributions={'n_estimators':[int(a) for a in np.linspace(start=20, stop=200, num=10)],

              'max_features':['sqrt', 'auto'],

              'max_depth':[int(a) for a in np.linspace(start=10, stop=100, num=10)],

              'min_samples_split':[a for a in np.arange(2,10)]}



random_searchcv=RandomizedSearchCV(rf_clf_fs, param_distributions=param_distributions, cv=5, n_iter=200, n_jobs=-1, random_state=52)

random_searchcv.fit(X_train_reduced, y_train)

random_searchcv.best_score_, random_searchcv.best_params_
rf_feature_importances=RandomForestClassifier(**random_searchcv.best_params_)

rf_feature_importances.fit(X_train_reduced, y_train)
feature_importances=pd.Series(rf_feature_importances.feature_importances_, index=X_train_reduced.columns, name='f_imp')
figure=plt.figure(figsize=(14,6))

g1=sns.barplot(x=feature_importances.values, y=feature_importances.index, orient='h')

g1.axvline(np.mean(feature_importances.values), ls='--')
rfecv = RFECV(estimator=rf_feature_importances, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(X_train_reduced, y_train)
rfecv.n_features_
X_train_reduced.columns[rfecv.support_] 
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
X_train_pca_transformed=pca.fit_transform(X_train_scaled_df)
from sklearn.ensemble import ExtraTreesClassifier
X_train_reduced=X_train.loc[:, X_train_reduced.columns[rfecv.support_]]
scores_etc1=[]

scores_etc2=[]

scores_etc3=[]

for _ in range (0, 100):

    

    etc1=ExtraTreesClassifier(random_state=_, bootstrap=True, oob_score=True)

    etc2=ExtraTreesClassifier(random_state=_, bootstrap=True, oob_score=True)

    etc3=ExtraTreesClassifier(random_state=_, bootstrap=True, oob_score=True)

    

    etc1.fit(X_train_scaled_df, y_train)

    etc2.fit(X_train_reduced, y_train)

    etc3.fit(X_train_pca_transformed, y_train)

    

    scores_etc1.append(etc1.oob_score_)

    scores_etc2.append(etc2.oob_score_)

    scores_etc3.append(etc3.oob_score_)



    

    
figure, ax=plt.subplots(1,2,figsize=(14,6))

g1=sns.barplot(x=['etc1','etc2','etc3'], y=[np.mean(scores_etc1),np.mean(scores_etc2),np.mean(scores_etc3)], ax=ax[0])

g2=sns.barplot(x=['etc1','etc2','etc3'], y=[np.std(scores_etc1),np.std(scores_etc2),np.std(scores_etc3)], ax=ax[1])

g1.set_ylabel('Accuracy mean')

g2.set_ylabel('Accuracy std');

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from scipy.stats import expon
dtc=DecisionTreeClassifier()

adaboost=AdaBoostClassifier(dtc)

kneighbors=KNeighborsClassifier()

nb=GaussianNB()

nn=MLPClassifier()

param_distributions_adaboost={'algorithm':['SAMME', 'SAMME.R'],

                              'n_estimators':[int(a) for a in np.linspace(start=10, stop=200, num=20)],

                              'learning_rate': expon(scale=2),

                              'base_estimator__max_features':['sqrt', 'auto'],

                              'base_estimator__max_depth':[int(a) for a in np.linspace(start=10, stop=100, num=10)],

                              'base_estimator__min_samples_split':[a for a in np.arange(2,10)]}



param_distributions_kneighbors={'n_neighbors': [a for a in np.arange(2,10)], 'weights':['uniform', 'distance']}

param_distributions_nb={'var_smoothing':expon(scale=0.5)}

param_distributions_nn={'solver':['lbfgs','sgd','adam'], 'hidden_layer_sizes':([int(a) for a in np.linspace(start=10,stop=50,num=5)],), 'learning_rate_init':expon(scale=1), 'learning_rate':['constant', 'invscaling', 'adaptive'] }
rsch_adaboost=RandomizedSearchCV(adaboost, param_distributions=param_distributions_adaboost, n_jobs=-1, cv=5, n_iter=200, scoring='accuracy',refit=True)

rsch_kneighbors=RandomizedSearchCV(kneighbors, param_distributions=param_distributions_kneighbors, n_jobs=-1, cv=5, n_iter=200,scoring='accuracy',refit=True)

rsch_nb=RandomizedSearchCV(nb, param_distributions=param_distributions_nb, n_jobs=-1, cv=5, n_iter=200,scoring='accuracy',refit=True)

rsch_nn=RandomizedSearchCV(nn, param_distributions=param_distributions_nn, n_jobs=-1, cv=5, n_iter=200,scoring='accuracy', refit=True)
rsch_nb.fit(X_train_reduced, y_train)
rsch_adaboost.fit(X_train_reduced, y_train)
rsch_kneighbors.fit(X_train_reduced, y_train)
rsch_nn.fit(X_train_reduced, y_train)
rsch_adaboost.best_score_,rsch_kneighbors.best_score_,rsch_nb.best_score_,rsch_nn.best_score_
X_train_reduced.shape
from sklearn.ensemble import VotingClassifier
voting_classifier=VotingClassifier(voting='soft', estimators=[

    ('rf', rf_feature_importances),

    ('ada', rsch_adaboost.best_estimator_),

    ('kneighbors', rsch_kneighbors.best_estimator_),

    ('bn', rsch_nb.best_estimator_),

    ('nn', rsch_nn.best_estimator_)

])
from sklearn.model_selection import cross_val_score
voting_score=cross_val_score(voting_classifier, X_train_reduced, y_train, scoring='accuracy', cv=5, n_jobs=-1)
voting_score_mean=np.mean(voting_score)

voting_score_std=np.std(voting_score)
voting_score_mean, voting_score_std
voting_classifier_hard=VotingClassifier(voting='hard', estimators=[

    ('rf', rf_feature_importances),

    ('ada', rsch_adaboost.best_estimator_),

    ('kneighbors', rsch_kneighbors.best_estimator_),

    ('bn', rsch_nb.best_estimator_),

    ('nn', rsch_nn.best_estimator_)

])
voting_score=cross_val_score(voting_classifier_hard, X_train_reduced, y_train, scoring='accuracy', cv=5, n_jobs=-1)
voting_hard_score_mean=np.mean(voting_score)

voting_hard_score_std=np.std(voting_score)
voting_hard_score_mean, voting_hard_score_std
#creating the array to store the estimators predictions

X_train_stacking=np.chararray((len(X_train_reduced), 4),)
for _ in range(0,len(X_train_reduced)):

    X_train_stacking[_,0]=rsch_adaboost.best_estimator_.predict(X_train_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]

    X_train_stacking[_,1]=rsch_kneighbors.best_estimator_.predict(X_train_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]

    X_train_stacking[_,2]= rsch_nb.best_estimator_.predict(X_train_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]

    X_train_stacking[_,3]=rsch_nn.best_estimator_.predict(X_train_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]
#we will use an extra trees classifier

extra_trees_stacking=ExtraTreesClassifier(bootstrap=True, oob_score=True)
X_train_stacking_float=np.empty(X_train_stacking.shape,)
for col in range(0,4):

    for row in range(0, X_train_stacking.shape[0]):

        if X_train_stacking[row, col]==b'B':

            X_train_stacking_float[row, col]=0

        else:

            X_train_stacking_float[row, col]=1

            
param_distributions_stacking={'n_estimators': np.arange(1,10),

                             'max_depth':[2,3,4,5,6,7,8,9,10,20,30,40],

                              'min_samples_split':[2,3,4,5,6],

                              'max_features':['sqrt', 'auto'],

                              }
randomized_searchcv_stacking=RandomizedSearchCV(extra_trees_stacking, param_distributions=param_distributions_stacking, n_jobs=-1, cv=5, n_iter=200, scoring='accuracy', refit=True)
randomized_searchcv_stacking.fit(X_train_stacking_float, y_train)
randomized_searchcv_stacking.best_score_
randomized_searchcv_stacking.best_estimator_
stacking_score_rf=cross_val_score(extra_trees_stacking, X_train_stacking_float, y_train, cv=5, n_jobs=-1, scoring='accuracy')
stacking_score_rf
figure=plt.figure(figsize=(14,6))

g1=sns.barplot(x=['adaboost', 'KNeighbors', 'GaussianNB', 'NN', 'SoftVoting', 'HardVoting', 'Stacking'], y=[rsch_adaboost.best_score_,rsch_kneighbors.best_score_,rsch_nb.best_score_,rsch_nn.best_score_,voting_score_mean,voting_hard_score_mean, randomized_searchcv_stacking.best_score_])

g1.set_title('Classifiers best prediction accuracies');
X_test_reduced=X_test.loc[:, X_train_reduced.columns]
X_test_reduced.shape
X_test_stacking=np.chararray((len(X_test_reduced), 4),)



for _ in range(0,len(X_test_reduced)):

    X_test_stacking[_,0]=rsch_adaboost.best_estimator_.predict(X_test_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]

    X_test_stacking[_,1]=rsch_kneighbors.best_estimator_.predict(X_test_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]

    X_test_stacking[_,2]= rsch_nb.best_estimator_.predict(X_test_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]

    X_test_stacking[_,3]=rsch_nn.best_estimator_.predict(X_test_reduced.iloc[_,:].values.reshape(1,rfecv.n_features_))[0]



X_test_stacking_float=np.empty(X_test_stacking.shape,)



for col in range(0,4):

    for row in range(0, X_test_stacking.shape[0]):

        if X_test_stacking[row, col]==b'B':

            X_test_stacking_float[row, col]=0

        else:

            X_test_stacking_float[row, col]=1

            
stacking_predictions=randomized_searchcv_stacking.best_estimator_.predict(X_test_stacking_float)
from sklearn.metrics import accuracy_score
stacking_accuracy=accuracy_score(y_test, stacking_predictions)
print('The accuracy score we have finally achieved: {:.4f}'.format(stacking_accuracy))