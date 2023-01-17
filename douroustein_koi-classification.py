import pandas as pd

import os

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

kepler_tab = pd.read_csv(os.path.join('../input', 'cumulative.csv'))

kepler_tab.head()

import warnings

warnings.filterwarnings("ignore") #This has been added once the warnings have been verified
# The following variables cannot be considered as features

not_features = ['rowid', #Index of the dataset

                'kepid', #Kepler Identification 

                'kepoi_name', #KOI Name

                'kepler_name', #Kepler Name

                'koi_disposition', #Target (Exoplanet Archive Disposition)

                'koi_pdisposition', #Disposition Using Kepler Data

                'koi_score', # As explained in the proposal, the koi score will be used as a challenger for the models

                'koi_time0bk', # Transit Epoch 

                'koi_time0bk_err1',

                'koi_time0bk_err2',

                'koi_tce_delivname', # TCE delivery name

                'koi_tce_plnt_num'] # TCE Planet Number
# The following variables have missing value for all data points

null_variables = []

for i in kepler_tab.columns:

    if kepler_tab[i].notnull().sum() == 0:

        null_variables.append(i)
# The following variables have the same value for all data points

unique_value = []

for i in kepler_tab.columns:

    if kepler_tab[i].value_counts().count() == 1:

        unique_value.append(i)
# We define here a first list of variables to be dropped based on the previous step of exploratory data analysis

drop_step1 = not_features + null_variables + unique_value

box = kepler_tab.drop(drop_step1,axis=1)
# We display variables values in boxplots to check if there is outliers for each

f = plt.figure(figsize=(20, 60))

i=1

import math

for col in box.columns:

    subplot = f.add_subplot(math.ceil(box.shape[1]/3),3,i)

    sns.boxplot(x=box[col].dropna(),whis=10)

    i=i+1

plt.subplots_adjust(hspace=0.4)

plt.show()
print('koi_period shows value {} that is very high compared to the rest of the values.'.format(box['koi_period'].max()))

print('This values means that the koi takes {:.0f} days = {:.0f} years to turn around its star (comparatively 1 year for the earth).'

      .format(box['koi_period'].max(),box['koi_period'].max()/365))

print('Even this is a very high period, it\'s not impossible')

print()

print('We can consider this koi as an outlier, but it won\'t affect the modeling since it\'s a {}'

      .format(kepler_tab.koi_disposition[np.argmax(box['koi_period'])]))

print()

second = box['koi_period'].sort_values(ascending=False).head(2)

second.index = [0,1]

print('Something interesting to notice also is that the second largest period for all koi is {} ({:.0f} years) which is lower than the mission duration (2009 to 2018).'

      .format(second[1],second[1]/365))

print('This confirm the limitation of the transit method to detect planets that are far from their star, and then have a period loger than the possible observation. If this koi with the highest period is confirmed to be a planet, it has been a happy luck to detect its transit.')
print('koi_depth shows value {} that is out of range, as the max value should be 1000000'.format(box['koi_depth'].max()))

print('The median value for koi_depth is {}, then the wrong value is replaced by {}'

      .format(box['koi_depth'].median(),box['koi_depth'].median()))

print('The error measurement values of koi_depth for the same data point are respectively {} and {}, which correspond to the outliers observed in the previous boxplots'

     .format(box.koi_depth_err1[np.argmax(box['koi_depth'])],box.koi_depth_err2[np.argmax(box['koi_depth'])]))

print('This implies that there has been a measurement issue for this datapoint related to the depth, this also error margin value of the depth are replaced by the median values for this data point')

box.koi_depth_err1[np.argmax(box['koi_depth'])] = box['koi_depth_err1'].median()

box.koi_depth_err2[np.argmax(box['koi_depth'])] = box['koi_depth_err2'].median()

box.koi_depth[np.argmax(box['koi_depth'])] = box['koi_depth'].median()
drop_step2 = drop_step1 # As no additionnal variables have been excluded in the previous step



# Create the dataframe that will be used in the PCA analysis

pca_df = kepler_tab.drop(drop_step2,axis=1)

pca_df_nn = pca_df[pca_df.isnull().sum(axis=1)==0]



# PVE calculation

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 

pca_df_rescaled = scaler.fit_transform(pca_df_nn)

from sklearn.decomposition import PCA

pca = PCA(n_components=None)

pca.fit(pca_df_rescaled)

pve = pca.explained_variance_ratio_



# Scree plot

x_coor = np.arange(1, pca_df.shape[1]+1)

f = plt.figure(figsize=(20, 10))

pve_cumsum = np.cumsum(pve[0:pca_df.shape[1]]) # Calculate cumulative sum

plt.plot(x_coor,pve_cumsum)

plt.xlabel('Number of component')

plt.ylabel('Proportion of variance explained')

plt.title('scree plot (cumulative sum)')

plt.xticks(np.arange(0, pca_df.shape[1]+1, step=10))

plt.show()



# How many components should we keep to retain 90%, 95%, 99% and 99,99% of the variance explained?

print('To retain 90% of the variance explained, we should keep {} components over {}'.

      format(((~(pve_cumsum >= 0.90)).sum())+1,pca_df.shape[1]))

print('To retain 95% of the variance explained, we should keep {} components over {}'.

      format(((~(pve_cumsum >= 0.95)).sum())+1,pca_df.shape[1]))

print('To retain 99% of the variance explained, we should keep {} components over {}'.

      format(((~(pve_cumsum >= 0.99)).sum())+1,pca_df.shape[1]))

print('To retain 99,99% of the variance explained, we should keep {} components over {}'.

      format(((~(pve_cumsum >= 0.9999)).sum())+1,pca_df.shape[1]))
# Variables to drop from the features list

to_drop = drop_step2

X_all = kepler_tab[kepler_tab.koi_disposition != 'CANDIDATE'].drop(to_drop,axis=1)

y_all = kepler_tab[kepler_tab.koi_disposition != 'CANDIDATE'].koi_disposition



# Data points having at least one empty value are dropped

X_nn = X_all[X_all.isnull().sum(axis=1)==0]

y_nn = y_all[X_all.isnull().sum(axis=1)==0]
# comparing input shapes

print('Input shape before filtering datapoints is ',X_all.shape)

print('Input shape after removing all data points with at least one empty value is ',X_nn.shape)
pd.value_counts(y_all, normalize=True)
pd.value_counts(y_nn, normalize=True)
# Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X_nn, y_nn,

    train_size=0.8, test_size=0.2, stratify=y_nn, random_state=1)



# Train/validation split

from sklearn.model_selection import train_test_split

X_tr, X_vl, y_tr, y_vl = train_test_split(

    X_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train, random_state=1)
# The following are commonly used for all, or at least most of the models



#Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# PCA

from sklearn.decomposition import PCA

pca = PCA()



# Grid search with cross-validation

from sklearn.model_selection import GridSearchCV



# Pipeline

from sklearn.pipeline import Pipeline
%%time

# Logistic regression estimator

from sklearn.linear_model import SGDClassifier

logreg = SGDClassifier(loss='log',max_iter=1000, tol=1e-3, random_state=0)



# Pipeline for Logistic regression

lr_pipe = Pipeline([

    ('scaler', scaler),

    ('pca', pca),

    ('logreg', logreg)

])



# Fit and tune the estimator

lr_gridcv = GridSearchCV(lr_pipe, [{

    'pca': [None],

    'logreg__alpha' :[1e-05,0.0001,0.001,0.01,0.1, 1,10]

},{    

    'pca__n_components': [19, 23, 28, 33],

    'logreg__alpha' :[1e-05,0.0001,0.001,0.01,0.1, 1,10]

}], cv=5)

lr_gridcv.fit(X_train, y_train)
# Sort accuracy for each alpha and number of components value

df = pd.DataFrame.from_items([

    ('n components', lr_gridcv.cv_results_['param_pca__n_components']),

    ('alpha', lr_gridcv.cv_results_['param_logreg__alpha']),

    ('mean accuracy', lr_gridcv.cv_results_['mean_test_score']),

    ('standard deviation', lr_gridcv.cv_results_['std_test_score'])

])

df['n components'].fillna(X_nn.shape[1],inplace=True)

df.sort_values(by='mean accuracy', ascending=False).head()
# Logistic regression accuracy on test set

lr_accuracy = lr_gridcv.score(X_test,y_test)

print('Logistic regression accuracy (test set): {:.4f}'.format(lr_accuracy))
%%time

# Creating and tuning Decision tree

from sklearn.tree import DecisionTreeClassifier

depth_values = range(1,21)

train_scores = []

valid_scores = []



for params_dict in depth_values:

    dt = DecisionTreeClassifier(max_depth=params_dict, random_state=0)

    dt.fit(X_tr, y_tr)

    train_scores.append(dt.score(X_tr, y_tr))

    valid_scores.append(dt.score(X_vl, y_vl))

    

# Visualizing tuning

f = plt.figure(figsize=(10, 5))

plt.plot(range(1,21),train_scores,label='Train scores',c='blue')

plt.plot(range(1,21),valid_scores,label='Validation scores',c='red')

plt.xlabel('Depth')

plt.ylabel('Scores')

plt.legend()

plt.show()



depth_opt = np.argmax(valid_scores)+1

print('Best accuracy on validation set is :{:.2f} with depth {}'.format(np.max(valid_scores),depth_opt))



# Decision tree accuracy on test set

dt = DecisionTreeClassifier(criterion='gini', max_depth=depth_opt, random_state=0)

dt.fit(X_train, y_train)

dt.score(X_test, y_test)



dt_accuracy = dt.score(X_test, y_test)

print('Decision tree accuracy (test set): {:.4f}'.format(dt_accuracy))
%%time

# Pipeline

dt_pipe = Pipeline([

    ('scaler', scaler),

    ('pca', pca),

    ('dt1', DecisionTreeClassifier(random_state=0))

])



# Fit and tune the estimator

dt_gridcv = GridSearchCV(dt_pipe, [{

    'pca': [None],

    'dt1__max_depth' :range(1,21),

    'dt1__criterion' :['gini','entropy']

},{

    'pca__n_components': [19, 23, 28, 33],

    'dt1__max_depth' :range(1,21),

    'dt1__criterion' :['gini','entropy']

}], cv=5)

dt_gridcv.fit(X_train, y_train)



# Sort accuracy for each depth and number of components value

df = pd.DataFrame.from_items([

    ('n components', dt_gridcv.cv_results_['param_pca__n_components']),

    ('depth', dt_gridcv.cv_results_['param_dt1__max_depth']),

    ('creterion', dt_gridcv.cv_results_['param_dt1__criterion']),

    ('mean accuracy', dt_gridcv.cv_results_['mean_test_score']),

    ('standard deviation', dt_gridcv.cv_results_['std_test_score'])

])

df['n components'].fillna(X_nn.shape[1],inplace=True)
df.sort_values(by='mean accuracy', ascending=False).head()
# Decision tree (with further tuning)accuracy on test set

dtgridcv_accuracy = dt_gridcv.score(X_test,y_test)

print('Decision tree (grid search with cross validation) accuracy (test set): {:.4f}'.format(dtgridcv_accuracy))
%%time

# Creating and tuning Random forest

from sklearn.ensemble import RandomForestClassifier

nb_estimators = [5,10,15,20,30,40,50,60,100]

valid_scores = []

for nb in nb_estimators:

    rf = RandomForestClassifier(n_estimators=nb, max_depth=None, random_state=0)

    rf.fit(X_tr, y_tr)

    valid_scores.append(rf.score(X_vl, y_vl))

import pandas as pd

rf_tune = pd.DataFrame({'n estimators':nb_estimators,'validation accuracy':valid_scores})
rf_tune.sort_values(by='validation accuracy', ascending=False)
# Random forest accuracy on test set

rf = RandomForestClassifier(n_estimators=15, max_depth=None, random_state=0)

rf.fit(X_train, y_train)



rf_accuracy = rf.score(X_test, y_test)

print('Random forest accuracy (test set): {:.4f}'.format(rf_accuracy))
%%time

# Pipeline

rf_pipe = Pipeline([

    ('scaler', scaler),

    ('pca', pca),

    ('rf1', RandomForestClassifier(max_depth=None, random_state=0))

])



# Fit and tune the estimator

rf_gridcv = GridSearchCV(rf_pipe, [{

    'pca': [None],

    'rf1__n_estimators' :[5,10,15,20,30,40,50,60,100],

    'rf1__criterion' :['gini','entropy']

},{

    'pca__n_components': [19, 23, 28, 33],

    'rf1__n_estimators' :[5,10,15,20,30,40,50,60,100],

    'rf1__criterion' :['gini','entropy']

}], cv=5)

rf_gridcv.fit(X_train, y_train)



# Sort accuracy for each depth and number of components value

df = pd.DataFrame.from_items([

    ('n components', rf_gridcv.cv_results_['param_pca__n_components']),

    ('n estimators', rf_gridcv.cv_results_['param_rf1__n_estimators']),

    ('creterion', rf_gridcv.cv_results_['param_rf1__criterion']),

    ('mean accuracy', rf_gridcv.cv_results_['mean_test_score']),

    ('standard deviation', rf_gridcv.cv_results_['std_test_score'])

])

df['n components'].fillna(X_nn.shape[1],inplace=True)
df.sort_values(by='mean accuracy', ascending=False).head()
# Random forest (with further tuning) accuracy on test set

rfgridcv_accuracy = rf_gridcv.score(X_test,y_test)

print('Random forest (grid search with cross validation) accuracy (test set): {:.4f}'.format(rfgridcv_accuracy))
%%time

# SVM linear

from sklearn.svm import LinearSVC



# Pipeline

svmlinear_pipe = Pipeline([

    ('scaler', scaler),

    ('pca', pca),

    ('svmlinear', LinearSVC(random_state=0))

])



# Fit and tune the estimator

svmlinear_gridcv = GridSearchCV(svmlinear_pipe, [{

    'pca': [None],

    'svmlinear__C' :[0.0001,0.001,0.01,0.1,1,10,100]

},{

    'pca__n_components': [19, 23, 28, 33],

    'svmlinear__C' :[0.0001,0.001,0.01,0.1,1,10,100]

}], cv=5)



svmlinear_gridcv.fit(X_train, y_train)



# Sort accuracy for each C value

import pandas as pd

df = pd.DataFrame.from_items([

    ('n components', svmlinear_gridcv.cv_results_['param_pca__n_components']),

    ('C', svmlinear_gridcv.cv_results_['param_svmlinear__C']),

    ('mean accuracy', svmlinear_gridcv.cv_results_['mean_test_score']),

    ('standard deviation', svmlinear_gridcv.cv_results_['std_test_score'])

])

df['n components'].fillna(X_nn.shape[1],inplace=True)
df.sort_values(by='mean accuracy', ascending=False).head()
# SVM Linear accuracy

svmlinear_accuracy = svmlinear_gridcv.score(X_test,y_test)

print('Linear SVM accuracy (test set): {:.4f}'.format(svmlinear_accuracy))
%%time

# SVC

from sklearn.svm import SVC # SVC doesn't scale well to a large number of data points

svc = SVC(kernel='rbf')



# Pipeline

from sklearn.pipeline import Pipeline

svmrbf_pipe = Pipeline([

    ('scaler', scaler),

    ('svc', svc)

])



# Fit and tune the estimator

from sklearn.model_selection import GridSearchCV

svmrbf_gridcv = GridSearchCV(svmrbf_pipe, {'svc__C': [0.01,0.1, 1, 10,100],

                                     'svc__gamma':[1e-05,1e-04,0.001,0.01,0.1]}, cv=5)

svmrbf_gridcv.fit(X_train, y_train)



# Sort accuracy for each C and gamma value

df = pd.DataFrame.from_items([

    ('C', svmrbf_gridcv.cv_results_['param_svc__C']),

    ('gamma', svmrbf_gridcv.cv_results_['param_svc__gamma']),

    ('mean accuracy', svmrbf_gridcv.cv_results_['mean_test_score']),

    ('standard deviation', svmrbf_gridcv.cv_results_['std_test_score'])

])
df.sort_values(by='mean accuracy', ascending=False).head()
# SVM RBF accuracy

svmrbf_accuracy = svmrbf_gridcv.score(X_test,y_test)

print('RBF SVM accuracy (test set): {:.4f}'.format(svmrbf_accuracy))
%%time

# KNN classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_jobs=-1)



# Pipeline

from sklearn.pipeline import Pipeline

knn_pipe = Pipeline([

    ('scaler', scaler),

    ('pca', pca),

    ('knn', knn)])



# Fit and tune the estimator

knn_gridcv = GridSearchCV(knn_pipe, [{

    'pca': [None],

    'knn__n_neighbors': np.arange(1, 50),

    'knn__p': [1, 2],

    'knn__weights': ['distance','uniform']    

},{

    'pca__n_components': [19, 23, 28, 33],

    'knn__n_neighbors': np.arange(1, 50),

    'knn__p': [1, 2],

    'knn__weights': ['distance','uniform']

}], cv=5)

knn_gridcv.fit(X_train, y_train)



# Sort accuracy for each depth and number of components value

df = pd.DataFrame.from_items([

    ('n components', knn_gridcv.cv_results_['param_pca__n_components']),

    ('n neighbors', knn_gridcv.cv_results_['param_knn__n_neighbors']),

    ('distance', knn_gridcv.cv_results_['param_knn__p']),

    ('weight', knn_gridcv.cv_results_['param_knn__weights']),

    ('mean accuracy', knn_gridcv.cv_results_['mean_test_score']),

    ('standard deviation', knn_gridcv.cv_results_['std_test_score'])

])

df['n components'].fillna(X_nn.shape[1],inplace=True)
df.sort_values(by='mean accuracy', ascending=False).head()
# Knn accuracy on the test set

knngridcv_accuracy = knn_gridcv.score(X_test,y_test)

print('knn accuracy (test set): {:.4f}'.format(knngridcv_accuracy))
# Checking score distribution depending on the class

score = kepler_tab[kepler_tab.koi_score.notnull()].loc[:,['koi_disposition','koi_score']]

plt.figure(figsize=(20, 10))

plt.hist(score.koi_score[score.koi_disposition=='CONFIRMED'],histtype='stepfilled', alpha=0.3, bins=20,

         label='CONFIRMED')

plt.hist(score.koi_score[score.koi_disposition=='FALSE POSITIVE'],histtype='stepfilled', alpha=0.3, bins=20,

         label='FALSE POSITIVE')

plt.legend(loc='center')

plt.show()
# Here we calculate the koi_score accuracy on the test set

y_pred_koiscore = []

for i in X_test.index:

    if kepler_tab.koi_score[i]>0.5:

        y_pred_koiscore.append('CONFIRMED')

    if kepler_tab.koi_score[i]<=0.5:

        y_pred_koiscore.append('FALSE POSITIVE')

nb_correct=0

#Note that some data points have no koi_score calculated, and will then be removed

y_test_reset = y_test[kepler_tab.koi_score.notnull()].reset_index(drop=True) 

for i in y_test_reset.index:

    if y_test_reset[i]==y_pred_koiscore[i]:

        nb_correct=nb_correct+1

koiscore_accuracy =nb_correct/y_test_reset.shape[0]

print('koi_score Accuracy: {:.4f}'.format(koiscore_accuracy))
# Accuracy

accuracy_list = pd.DataFrame({'Model':['Knn','Decision tree','Random forest','SVM Linear','SVM RBF',

                                       'Logistic regression','koi_score'],

                              'Test accuracy':[knngridcv_accuracy,dtgridcv_accuracy,rf_accuracy,

                                               svmlinear_accuracy,svmrbf_accuracy,lr_accuracy,koiscore_accuracy]})

accuracy_list.sort_values(by='Test accuracy', ascending=False)
# Source of this function : 

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-

# selection-plot-confusion-matrix-py

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,

                          title=None,

                          cmap=plt.cm.Blues):

    

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax





np.set_printoptions(precision=2)

pass
# Confusion Matrices

models = [['Logistic regression',lr_gridcv],

['Decision tree',dt_gridcv],

['Random forest',rf],

['SVM linear',svmlinear_gridcv],

['SVM RBF',svmrbf_gridcv],

['Knn baseline',knn_gridcv]]

from sklearn.metrics import confusion_matrix

y_true = y_test

plot_confusion_matrix(y_test_reset, y_pred_koiscore, classes=['CONFIRMED','FALSE POSITIVE'], title='koi_score')

for i in range(6):

    y_pred = models[i][1].predict(X_test)

    plot_confusion_matrix(y_true, y_pred, classes=['CONFIRMED','FALSE POSITIVE'], title=models[i][0])
prob = lr_gridcv.predict_proba(X_test[kepler_tab.koi_score.notnull()].values)[:,0]

score = kepler_tab.koi_score[X_test[kepler_tab.koi_score.notnull()].index].values

y_lr_pred = lr_gridcv.predict(X_test[kepler_tab.koi_score.notnull()].values)

f = plt.figure(figsize=(20, 5))



# Plot a scatter of probabilities and koi_score

subplot = f.add_subplot(1,2,1)

plt.scatter(prob,score)

# Plot separately, in a different color, points where logistic regression failed to classify

plt.scatter(prob[~(y_lr_pred == y_test[kepler_tab.koi_score.notnull()])],

            score[~(y_lr_pred == y_test[kepler_tab.koi_score.notnull()])],

            color='red',label='wrong lr classification')

# Plot separately, in a different color, points where koi_score failed to classify

plt.scatter(prob[~(y_pred_koiscore == y_test[kepler_tab.koi_score.notnull()])],

            score[~(y_pred_koiscore == y_test[kepler_tab.koi_score.notnull()])],

            color='orange',label='wrong koi_score classification')

plt.hlines(0.5, -0.1, 1.1,linestyles='solid', color='red')

plt.vlines(0.5, -0.1, 1.1,linestyles='solid', color='red')

plt.xlabel('Logistic regression probability')

plt.xlim(-0.1,1.1)

plt.ylim(-0.1,1.1)

plt.ylabel('koi_score')

plt.legend(loc='center')
from sklearn import metrics

y_true = y_test[kepler_tab.koi_score.notnull()]

fpr_lr, tpr_lr, thr_lr = metrics.roc_curve(y_true, prob, pos_label='CONFIRMED')

fpr_koiscore, tpr_koiscore, thr_koiscore = metrics.roc_curve(y_true, score, pos_label='CONFIRMED')

plt.figure(figsize=(20,10))

plt.plot(fpr_lr, tpr_lr,color='blue',label='Logistic regression')

plt.plot(fpr_koiscore, tpr_koiscore,color='red',label='koi_score')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend()

plt.show()
%%time

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('Logistic regression',lr_gridcv), ('Random forest',rf), ('Decision tree',dt_gridcv), 

                                     ('SVM linear',svmlinear_gridcv), ('SVM RBF',svmrbf_gridcv), ('knn',knn_gridcv)], voting='hard')

voting.fit(X_train,y_train)

voting_accuracy = voting.score(X_test,y_test)

print('Voting classifier accuracy (test set): {:.4f}'.format(voting_accuracy))
# Select the input

X_pred = kepler_tab[(kepler_tab.koi_disposition == 'CANDIDATE')].drop(to_drop,axis=1)

X_pred_nn = X_pred[X_pred.isnull().sum(axis=1)==0]

# The input for the koi_score classifier is different, as using different features

X_koiscore = kepler_tab[(kepler_tab.koi_disposition == 'CANDIDATE')&(kepler_tab.koi_score.notnull())].drop(to_drop,axis=1)



# Prediction for each classifier

y_pred_lr = lr_gridcv.predict(X_pred_nn)

y_pred_dt = dt.predict(X_pred_nn)

y_pred_rf = rf.predict(X_pred_nn)

y_pred_svmlinear = svmlinear_gridcv.predict(X_pred_nn)

y_pred_svmrbf = svmrbf_gridcv.predict(X_pred_nn)

y_pred_knn = knn_gridcv.predict(X_pred_nn)

y_pred_koiscore = []

for i in X_koiscore.index:

    if kepler_tab.koi_score[i]>0.5:

        y_pred_koiscore.append('CONFIRMED')

    if kepler_tab.koi_score[i]<=0.5:

        y_pred_koiscore.append('FALSE POSITIVE')

y_pred_voting = voting.predict(X_pred_nn)
# Show results

index=np.array([0,1])

plt.figure(figsize=(20,10))

unique, counts = np.unique(y_pred_voting, return_counts=True)

plt.bar(index-0.4, counts, width=0.1, color='g', edgecolor='black',label='Voting')

print('Voting',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_lr, return_counts=True)

plt.bar(index-0.3, counts, width=0.1, color='b', edgecolor='black',label='Logistic regression')

print('Logistic regression',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_dt, return_counts=True)

plt.bar(index-0.2, counts, width=0.1, color='r', edgecolor='black',label='Decision tree')

print('Decision tree',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_rf, return_counts=True)

plt.bar(index-0.1, counts, width=0.1, color='y', edgecolor='black',label='Random forest')

print('Random forest',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_svmlinear, return_counts=True)

plt.bar(index+0.1, counts, width=0.1, color='g', edgecolor='black',label='SVM Linear',alpha=0.3)

print('SVM Linear',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_svmrbf, return_counts=True)

plt.bar(index+0.2, counts, width=0.1, color='b', edgecolor='black',label='SVM RBF',alpha=0.3)

print('SVM RBF',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_knn, return_counts=True)

plt.bar(index+0.3, counts, width=0.1, color='r', edgecolor='black',label='Knn',alpha=0.3)

print('Knn',dict(zip(unique, counts)))

unique, counts = np.unique(y_pred_koiscore, return_counts=True)

plt.bar(index+0.4, counts, width=0.1, color='y', edgecolor='black',label='koi_score',alpha=0.3)

print('koi_score',dict(zip(unique, counts)))

plt.bar(index, [0,0],tick_label=unique,align='center')

plt.ylim(top=2000)

plt.legend()

plt.show()
# Let's create the prediction output dataframe

output = pd.DataFrame({'Name':kepler_tab.kepoi_name[X_pred_nn.index].values,'Result':y_pred_voting})

output.head()