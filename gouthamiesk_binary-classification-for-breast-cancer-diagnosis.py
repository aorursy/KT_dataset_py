import numpy as np 

import pandas as pd 

import seaborn as sns 

from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks

%matplotlib inline

import matplotlib.pyplot as pl

import matplotlib.patches as mpatches

import importlib

importlib.import_module('mpl_toolkits.mplot3d').Axes3D

from time import time

from sklearn.metrics import f1_score, accuracy_score

from collections import Counter

import re

from sklearn.model_selection import cross_validate

import math

import random 

from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_validate

import re



random.seed(50)
def als_split_data(data):

    output = data['diagnosis']

    features = pd.DataFrame(data=data)

    cols = ['id', 'diagnosis']

    for col in cols:

        if col in features.columns:

            features = features.drop(col, axis=1)

    return output, features



data = pd.read_csv('../input/breast-cancer.csv')

data.columns

# Removing the last unnamed columns

data = data.drop(['Unnamed: 32'], axis =1)

output, features = als_split_data(data)

data.head()
display(features.describe())
def vs_distribution(data):

    """

    Visualization code for displaying skewed distributions of features

    """

    

    # Create figure

    fig = pl.figure(figsize = (18,15));



    # Skewed feature plotting

    for i, feature in enumerate(data.columns[:10]):

        ax = fig.add_subplot(5, 5, i+1)

        ax.hist(data[feature], bins = 25, color = '#00A0A0')

        ax.set_title("'%s'"%(feature), fontsize = 14)

        ax.set_xlabel("Value")

        ax.set_ylabel("Number of Records")

        #ax.set_ylim((0, 2000))

        #ax.set_yticks([0, 500, 1000, 1500, 2000])

        #ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])



        fig.suptitle("Distributions Features", \

            fontsize = 16, y = 1.03)



    fig.tight_layout()

    fig.show()
vs_distribution(features)
# For each feature print the rows which have outliers in all features 

def als_print_outliers(data, how_far=2, worst_th=6, to_display=False):

    # Select the last 10 features as they are the worst collected during measurements

    data = data.iloc[:,11:30]

    really_bad_data = defaultdict(int)

    for col in data.columns:

        Q1 = np.percentile(data[col], 25)

        Q3 = np.percentile(data[col], 75)

        step = (Q3-Q1)*how_far

        bad_data = list(data[~((data[col]>=Q1-step)&(data[col]<=Q3+step))].index)

        for i in bad_data:

            really_bad_data[i]+= 1

        # Display the outliers

    max_ind = max(really_bad_data.values())

    worst_points = [k for k, v in really_bad_data.items() if v > max_ind-worst_th]

    if to_display:

        print("Data points considered outliers are:") 

        display(data.ix[worst_points,:])

    return worst_points

    
outlier_indices = als_print_outliers(features, worst_th=3)
# Cleaning dataset by dropping outliers (cl)

data_cl = data.drop(data.index[outlier_indices]).reset_index(drop=True) # cleaned data

output_cl, features_cl = als_split_data(data_cl)



vs_distribution(features_cl)
print('Size of new dataset is {0:.2f} % of the original'.format(100.0*len(data_cl)/len(data)))
def als_transform_log_minmax(data):

    cols = data.columns

    data_transformed = pd.DataFrame(data=data)

    scaler = MinMaxScaler()

    for col in cols:

        data_transformed[col] = data[col].apply(lambda x: np.log(x+1))

        data_transformed[col] = scaler.fit_transform(data[col].values.reshape(-1,1))

    return data_transformed
# Applying log transfromation and minmax scaling (tr)

features_cl_tr = als_transform_log_minmax(features_cl) # cleaned, transformed data

data_cl_tr = pd.concat([output_cl, features_cl_tr], axis=1)
vs_distribution(features_cl_tr)
outlier_indices = als_print_outliers(features_cl_tr,worst_th=3)
# Cleaning dataset again - dropping outliers

data_cl_tr_cl = data_cl_tr.drop(data_cl.index[outlier_indices]) # cleaned transformed cleaned data

output_cl_tr_cl, features_cl_tr_cl = als_split_data(data_cl_tr_cl)
print('Size of new dataset is {0:.2f} % of the original'.format(100.0*len(data_cl_tr_cl)/len(data)))
def als_encode_diagnosis(d):

    if d== 'B':

        ed = 0

    else:

        ed = 1

    return ed
def vs_show_output_classes(data, data_clean):

    '''

    Visualization code for histogram of classes

    '''

    

    # Create figure

    fig = pl.figure(figsize=(10,6))

    

    encoded_data = data.apply(lambda x: als_encode_diagnosis(x))

    encoded_data_clean = data_clean.apply(lambda x: als_encode_diagnosis(x))



    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(encoded_data, bins=np.arange(3), alpha=0.5, color='b', label='Lost data', width=0.5)

    n_c, bins_c, patches_c = ax.hist(encoded_data_clean, bins=np.arange(3), color='k', label = 'Data filtered for outliers',width=0.5)

    '''

    colors = ['r', 'g']

    for i in range(2):

        patches[i].set_fc(colors[i])

        patches_c[i].set_fc(colors[i])

    '''    

    ax.set_title('Barplot of output classes', fontsize=16)

    ax.set_xticks([b+0.25 for b in bins[:-1]])

    ax.set_xticklabels(['Benign', 'Malign'], fontsize=16)

    

    ax.legend(fontsize=16)

    ax.set_ylabel('Number of records', fontsize=16)

    fig.tight_layout()

    fig.show()
vs_show_output_classes(output, output_cl_tr_cl)
def als_return_select_cols(data, **kwargs):

    checks = ['radius', 'area', 'perimeter']

    cols = [c for c in data.columns for ch in checks if re.search('{}(.)'.format(ch), c)]

    if kwargs['which']=='mean_non_dims':

        cols = [c for c in data.columns if c not in cols]

        cols = [c for c in cols if re.search('(.)_mean', c)]

    elif kwargs['which']=='se_non_dims':

        cols = [c for c in data.columns if c not in cols]

        cols = [c for c in cols if re.search('(.)_se', c)]

    elif kwargs['which']=='worst_non_dims':

        cols = [c for c in data.columns if c not in cols]

        cols = [c for c in cols if re.search('(.)_worst', c)]

    elif kwargs['which']=='all':

        cols = data.columns

    return cols
def vs_violin_swarm_plots(data, **kwargs):

    cols = ['diagnosis'] + als_return_select_cols(data, which=kwargs['which'])

    d = pd.melt(data[cols], id_vars = 'diagnosis', var_name = 'features', value_name = 'value')

    

    sns.set(font_scale=1.5)

    fig, ax = pl.subplots(figsize=(15,10))

    ax = sns.violinplot(x='features', y = 'value', hue='diagnosis', data=d, split=True, inner='quart')

    for tick in ax.get_xticklabels():

        tick.set_rotation(45)    

    fig.tight_layout()

    pl.subplots_adjust(bottom=0.2)

    pl.show()
vs_violin_swarm_plots(data_cl_tr_cl,which='only_dims') # first 10 features
vs_violin_swarm_plots(data_cl_tr_cl,which='mean_non_dims') # Next 10 features
vs_violin_swarm_plots(data_cl_tr_cl,which='se_non_dims') # Last 10 features
vs_violin_swarm_plots(data_cl_tr_cl,which='worst_non_dims') # Last 10 features
def vs_observe_correlations(data, **kwargs):

    cols = als_return_select_cols(data, which=kwargs['which'])

    fig,ax = pl.subplots(figsize=(10,7))

    sns.heatmap(data[cols].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

    fig.tight_layout()

    pl.show()
vs_observe_correlations(features_cl_tr_cl, which='only_dims')
vs_observe_correlations(features_cl_tr_cl, which='mean_non_dims')
vs_observe_correlations(features_cl_tr_cl, which='se_non_dims')
vs_observe_correlations(features_cl_tr_cl, which='worst_non_dims')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()

output_float = output_cl_tr_cl.apply(lambda x: als_encode_diagnosis(x))

coeffs = clf.fit(features_cl_tr_cl[:450], output_float[:450]).coef_.T

LDA_F = features_cl_tr_cl[:450].dot(coeffs)



preds = clf.predict(features_cl_tr_cl[451:])

fig, ax = pl.subplots()

ax.scatter(np.arange(len(preds)), preds-output_float[451:], c = preds, cmap='winter')

ax.legend()
# Applying PCA

from sklearn.decomposition import PCA 
def vs_plot_pca_variance(pca):

    x = np.arange(1,len(pca.components_)+1)

    fig, ax = pl.subplots(figsize=(10,6))

    

    # plot the cumulative variance

    ax.plot(x, np.cumsum(pca.explained_variance_ratio_), '-o', color='black')



    # plot the components' variance

    ax.bar(x, pca.explained_variance_ratio_, align='center', alpha=0.5)



    # plot styling

    ax.set_ylim(0, 1.05)

    

    for i,j in zip(x, np.cumsum(pca.explained_variance_ratio_)):

        ax.annotate(str(j.round(2)),xy=(i+.2,j-.02))

    ax.set_xticks(range(1,len(pca.components_)+1))

    ax.set_xlabel('PCA components')

    ax.set_ylabel('Explained Variance')

    

    fig.tight_layout()

    pl.show()

    
pca = PCA(n_components = 6).fit(features_cl_tr_cl)

vs_plot_pca_variance(pca)
def vs_pca_results(good_data, pca, **kwargs):

    cols = als_return_select_cols(good_data, which=kwargs['which'])

    cols_indices = [i for i, j in enumerate(good_data.keys()) if j in cols]

    # Dimension indexing

    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



    # PCA components

    components = pd.DataFrame(np.round(pca.components_[:,cols_indices], 4), columns = good_data.keys()[cols_indices])

    components.index = dimensions



    # PCA explained variance

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

    variance_ratios.index = dimensions



    # Create a bar plot visualization

    fig, ax = pl.subplots(figsize = (14,8))



    # Plot the feature weights as a function of the components

    components.plot(ax = ax, kind = 'bar');

    ax.set_ylabel("Feature Weights")

    ax.set_xticklabels(dimensions, rotation=0)
# Generate PCA results plot

vs_pca_results(features_cl_tr_cl, pca, which='only_dims')
vs_pca_results(features_cl_tr_cl, pca, which='mean_non_dims')
vs_pca_results(features_cl_tr_cl, pca, which='se_non_dims')
vs_pca_results(features_cl_tr_cl, pca, which='worst_non_dims')
def als_return_reduced_data(good_data, pca):

    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    reduced_data = pd.DataFrame(data=pca.transform(good_data), columns=dimensions)

    return reduced_data
reduced_features = als_return_reduced_data(features_cl_tr_cl, pca)

output_float = output_cl_tr_cl.apply(lambda x: als_encode_diagnosis(x))
def vs_scatter_two_dimensions(reduced_features, output_float):

    fig, ax = pl.subplots(figsize=(8,5))

    ax.scatter(reduced_features.loc[:,'Dimension 1'], reduced_features.loc[:, 'Dimension 2'], c=output_float, cmap='winter')

    ax.set_xlabel('Dimension 1')

    ax.set_ylabel('Dimension 2')

    ax.set_title('Projections of features on first two principal components')

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    

    fig.tight_layout()

    pl.show()
vs_scatter_two_dimensions(reduced_features, output_float)
def als_print_evaluation_metrics(clf, x, y, scoring, cv=5, only_times=True, print_times=True):

    scores = cross_validate(clf, x, y, cv=cv, scoring=scoring, return_train_score=True)

    if print_times:

        print('Average fit time is:   {:.3f}s'.format(np.mean(scores['fit_time'])))

        print('Average score time is: {:.3f}s\n'.format(np.mean(scores['score_time'])))

    if not only_times:

        print(' {: >7} {: >10} |  {: >3}    |  {: >3}    |  {: >3}    |'.format(' ', ' ', 'Avg', 'Min', 'Max'))

        for f in ['train', 'test']:

            for s in scoring:

                key = [sc for sc in scores.keys() if re.search('{}(.){}'.format(f,s),sc)]

                print(' {: >7} {: >10} |  {: >.3f}  |  {: >.3f}  |  {: >.3f}  |'.format(f, s, np.mean(scores[key[0]]), np.min(scores[key[0]]), np.max(scores[key[0]])))
def vs_plot_evaluation_metrics(clfs, clf_labels, x, y, cv=5):

    scoring = ['accuracy', 'precision', 'recall']

    scores = {}

    for label, clf in zip(clf_labels, clfs):

        scores[label] = cross_validate(clf, x, y, cv=cv, scoring=scoring, return_train_score=True)

    colors = ['b', 'g', 'r', 'k', 'c']

    lab2 = ['Avg', 'Min', 'Max']

    fig, ax = pl.subplots(2,3, figsize = (20,15))

    for i, f in enumerate(['train', 'test']):

        for j, s in enumerate(scoring):

            minval=1

            for k, lab1 in enumerate(clf_labels):

                scs = scores[lab1]

                key = [sc for sc in scs.keys() if re.search('{}(.){}'.format(f,s),sc)][0]

                alphac = [1,0.2,0.5]

                for l, lval in enumerate([np.mean(scs[key]), np.min(scs[key]), np.max(scs[key])]):

                    lab = lab1 + ' ' + lab2[l]

                    if lval < minval:

                        minval = lval

                    ax[i,j].bar(k+1+(0.23*l), lval, 0.23, color=colors[k], label=lab, alpha=alphac[l]) 

            ax[i,j].legend()

            ax[i,j].set_ylim(minval-0.01,1)

            ax[i,j].set_xlim(0,8)

            ax[i,j].set_title('{} {} scores'.format(f,s))

            ax[i,j].set_ylabel('Score')

            ax[i,j].set_xticklabels([])

    fig.tight_layout()

    pl.show()
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



clf_GNB = GaussianNB()

clf_RF = RandomForestClassifier()

clf_KNN = KNeighborsClassifier()

clf_SVM = SVC()

clfs = [clf_GNB, clf_RF, clf_KNN, clf_SVM]

clf_labels = ['GNB', 'RF', 'KNN', 'SVC']

vs_plot_evaluation_metrics(clfs, clf_labels, reduced_features, output_float, cv=5)

scoring=['accuracy', 'precision', 'recall', 'f1']

for label, clf in zip(clf_labels, clfs):

    print('{}:'.format(label))

    als_print_evaluation_metrics(clf, reduced_features, output_float, scoring, cv=5)
scoring=['f1']

for label, clf in zip(['KNN', 'SVC'], [clf_KNN, clf_SVM]):

    print('{}:'.format(label))

    als_print_evaluation_metrics(clf, reduced_features, output_float, scoring, cv=5, only_times=False, print_times=False)

    print('\n')
cols = ['radius_mean', 'radius_se', 'radius_worst', 'perimeter_mean', 'perimeter_se', 'perimeter_worst', 'area_se', 'area_worst', 'smoothness_se', 'compactness_se', 'concave points_se', 'concavity_se', 'symmetry_se']

selected_features = pd.DataFrame(features_cl_tr_cl)

for col in cols:

    selected_features = selected_features.drop([col],axis=1)

pca = PCA(n_components = 6).fit(selected_features)

vs_plot_pca_variance(pca)
selected_reduced_features = als_return_reduced_data(selected_features, pca)

clf_GNB = GaussianNB()

clf_RF = RandomForestClassifier()

clf_KNN = KNeighborsClassifier()

clf_SVM = SVC()

clfs = [clf_GNB, clf_RF, clf_KNN, clf_SVM]

clf_labels = ['GNB', 'RF', 'KNN', 'SVC']

output_float = output_cl_tr_cl.apply(lambda x: als_encode_diagnosis(x))

vs_plot_evaluation_metrics(clfs, clf_labels, selected_reduced_features, output_float, cv=5)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, fbeta_score, precision_score, make_scorer

from sklearn.model_selection import ShuffleSplit



knn = KNeighborsClassifier()

parameters = {'n_neighbors':list(range(2,7)), 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}

scoring=make_scorer(fbeta_score, beta=0.5)

clf = GridSearchCV(knn, parameters, scoring=scoring, cv=5)

clf.fit(reduced_features, output_float)

results_pd = pd.DataFrame(clf.cv_results_)
select_cols = ['mean_train_score', 'mean_test_score', 'param_algorithm', 'param_n_neighbors', 'rank_test_score', 'std_test_score']

results_pd[select_cols].sort_values(['rank_test_score'], ascending=True).reset_index(drop=True).head(8)
best_clf = KNeighborsClassifier(n_neighbors=4, algorithm='auto')

next_best_clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
scoring=['accuracy', 'precision', 'recall', 'f1']

print('Best obtained from grid search:\n')

als_print_evaluation_metrics(best_clf, reduced_features, output_float, scoring, only_times=False)

print('\nSecond best model from grid search, it has lower variance of test scores across folds:\n')

als_print_evaluation_metrics(next_best_clf, reduced_features, output_float, scoring, only_times=False)
def als_corrupt_output(output_float, f):

    positives = output_float[output_float==1]

    turnovers = int(len(positives)*f)

    turnover_index = np.random.choice(positives.index, turnovers)

    output_float[turnover_index] = 0

    return output_float

               
from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix 

# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(reduced_features, 

                                                    output_float, 

                                                    test_size = 0.2, 

                                                    random_state = 0)

best_clf.fit(X_train, y_train)

y_preds = best_clf.predict(X_test)

print('Confusion matrix of best model tested on original testing data:')

print(pd.DataFrame(confusion_matrix(y_test, y_preds), columns=['TP', 'FN'], index=['FN', 'TN']))

print('\n')

print('Confusion matrix of best model tested on testing data corrupted with false benign diagnoses, by 30%:')

y_test_corrupted = als_corrupt_output(y_test, 0.3)

print(pd.DataFrame(confusion_matrix(y_test_corrupted, y_preds), columns=['TP', 'FN'], index=['FN', 'TN']))

def vs_biplot(good_data, reduced_data, output_float, pca):

    '''

    Produce a biplot that shows a scatterplot of the reduced

    data and the projections of the original features.

    

    good_data: original data, before transformation.

               Needs to be a pandas dataframe with valid column names

    reduced_data: the reduced data (the first two dimensions are plotted)

    pca: pca object that contains the components_ attribute



    return: a matplotlib AxesSubplot object (for any additional customization)

    

    This procedure is inspired by the script:

    https://github.com/teddyroland/python-biplot

    '''



    fig = pl.figure(figsize = (22,10))

    ax1 = fig.add_subplot(1,3,3, projection='3d')

    # scatterplot of the reduced data    

    xs = reduced_data.loc[:, 'Dimension 1']

    ys = reduced_data.loc[:, 'Dimension 2']

    zs = reduced_data.loc[:, 'Dimension 3']

    

    ax1.scatter(xs, ys, zs, c=output_float, cmap='winter')

    feature_vectors = pca.components_.T



    # we use scaling factors to make the arrows easier to see

    arrow_size, text_pos = 5.6, 6



    # projections of the original features

    for i, v in enumerate(feature_vectors):

        ax1.plot([0, arrow_size*v[0]], [0, arrow_size*v[1]], [0, arrow_size*v[2]], lw=1.5, color='red')

        ax1.text(v[0]*text_pos, v[1]*text_pos, v[2]*text_pos, good_data.columns[i], color='black', 

                 ha='center', va='center', fontsize=14)



    ax1.set_xlabel("Dimension 1", fontsize=14)

    ax1.set_ylabel("Dimension 2", fontsize=14)

    ax1.set_zlabel("Dimension 3", fontsize=14)



    ax1.set_title("Scatter on first 3 PCs", fontsize=18);

    

    ax2 = fig.add_subplot(1,3,1, projection='3d')

    cols = ['smoothness_mean', 'concavity_mean', 'compactness_se']

    # scatterplot of the reduced data    

    xs = good_data.loc[:, cols[0]]

    ys = good_data.loc[:, cols[1]]

    zs = good_data.loc[:, cols[2]]

    

    ax2.scatter(xs, ys, zs, c=output_float, cmap='winter')

    ax2.set_xlabel(cols[0], fontsize=14)

    ax2.set_ylabel(cols[1], fontsize=14)

    ax2.set_zlabel(cols[2], fontsize=14)



    ax2.set_title("Scatter on any 3 'non-significant' features", fontsize=18);

    

    ax3 = fig.add_subplot(1,3,2, projection='3d')

    cols = ['area_mean', 'texture_mean', 'fractal_dimension_mean']

    # scatterplot of the reduced data    

    xs = good_data.loc[:, cols[0]]

    ys = good_data.loc[:, cols[1]]

    zs = good_data.loc[:, cols[2]]

    

    ax3.scatter(xs, ys, zs, c=output_float, cmap='winter')

    ax3.set_xlabel(cols[0], fontsize=14)

    ax3.set_ylabel(cols[1], fontsize=14)

    ax3.set_zlabel(cols[2], fontsize=14)



    ax3.set_title("Scatter on any 3 'significant' features", fontsize=18);

    

    fig.tight_layout()

    pl.show()

    
vs_biplot(features_cl_tr_cl, reduced_features, output_float, pca)