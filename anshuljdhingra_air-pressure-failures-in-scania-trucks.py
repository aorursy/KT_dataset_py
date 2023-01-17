import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from IPython.display import display, Markdown



from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import svm



import scipy.cluster.hierarchy as hac



%matplotlib inline

%config IPCompleter.greedy=True

warnings.filterwarnings('ignore')
# Set normalization

enable_normalization = True

normalization_type = 'minmax' # 'minmax' or 'standard'



# Exploratory analysis



# Set correlation

enable_correlation = False

enable_dendrogram = False

enable_heatmap = False



# Features Selection



# Set features selection with correlation criteria

enable_correlation_selec = False

factor = 0.95 # number close to 1



# Set features selection with univariate statitics test criteria

enable_univariate_selec = True

method_selec = 'selectkbest' # 'selectkbest', 'pca', ...

pca_variance = 0.95 

criteria_k_best = mutual_info_classif # chi2, mutual_info_classif

k_best = 84 # number of best features to select on select k best.



# Balancing



# Train balancing

enable_balancing = True

number_samples = 2500



# Machine learning method



ml_method = 'randomforestreg' # 'gradientboosting', 'svm', ...

gbc_loss = 'deviance' # gradient boosting loss

rfc_criterion = 'gini' # random forest criterion

enable_cv = True # enable cross-validation

# Print the bar graph from data

def bar(acumm_data):

    # Do plot

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(111)

    ax = sns.barplot(x=acumm_data.index, y=acumm_data.values, palette='tab20b', ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)    

    return ax



def dendrogram(df):    

    # Do correlation matrix

    corr_matrix = df.corr()



    # Do the clustering

    Z = hac.linkage(corr_matrix, 'single')



    # Plot dendogram

    fig, ax = plt.subplots(figsize=(25, 10))

    plt.title('Hierarchical Clustering Dendrogram')

    plt.xlabel('sample index')

    plt.ylabel('distance')

    groups = hac.dendrogram(

        Z,

        leaf_rotation=90.,  # rotates the x axis labels

        leaf_font_size=8., # font size for the x axis labels

        color_threshold = 0#,

        #truncate_mode='lastp',

        #p=30

    )



    labels_dict = pd.DataFrame(df.columns).to_dict()[0]

    actual_labels = [item.get_text() for item in ax.get_xticklabels()]

    new_labels = [labels_dict[int(i)] for i in actual_labels]

    ax.set_xticklabels(new_labels)

    plt.tight_layout()



def corr_drop(corr_m, factor=.9):

    

    global cm

    cm = corr_m

    # Get correlation score, as high as this score, more chances to be dropped.

    cum_corr = cm.applymap(abs).sum()

    def remove_corr():

        global cm

        for col in cm.columns:

            for ind in cm.index:

                if (ind in cm.columns) and (col in cm.index):

                    # Compare if are high correlated.

                    if (cm.loc[ind,col] > factor) and (ind!=col):

                        cum = cum_corr[[ind,col]].sort_values(ascending=False)

                        cm.drop(cum.index[0], axis=0, inplace=True)

                        cm.drop(cum.index[0], axis=1, inplace=True)

                        # Do recursion until the last high correlated.

                        remove_corr()

        return cm

    return remove_corr()
train_features = pd.read_csv('../input/aps_failure_training_set_processed_8bit.csv', na_values='na')

test_features =  pd.read_csv('../input/aps_failure_test_set_processed_8bit.csv', na_values='na')



train_labels = train_features['class']

test_labels = test_features['class']

train_features = train_features.drop('class', axis=1)

test_features = test_features.drop('class', axis=1)
train_features.describe()
flat_data = train_features.values.flatten()

count=0

for value in flat_data:

    if value is not None:

        continue

    count+= 1

pct_nan = round(100*count/len(flat_data))

print(f'{pct_nan}% of data are non-valid.')
from sklearn.preprocessing import MinMaxScaler

if enable_normalization and normalization_type=='minmax':

    scaler = MinMaxScaler()

    scaler.fit(train_features)

    train_features = pd.DataFrame(scaler.transform(train_features), columns=train_features.columns)
train_features.describe()
train_labels = train_labels.apply(round)

train_labels = train_labels.replace({-1:0})
bar(train_labels.value_counts())

plt.show()
if enable_correlation and enable_dendrogram:

    corr_matrix = train_features.corr()

    dendrogram(corr_matrix)

    plt.tight_layout()
if enable_correlation and enable_heatmap:

    fig, ax = plt.subplots(figsize=(10,10))

    ax = sns.heatmap(corr_matrix, square=True, cmap='Purples', ax=ax)

    plt.tight_layout()

    plt.show()
# to enable run correlation selection without univariate selection.

best_train_features = train_features 

new_corr_matrix = best_train_features.corr()
if enable_univariate_selec:

    if method_selec=='selectkbest':

        selectKBest = SelectKBest(chi2, k_best)

        selectKBest.fit(train_features, train_labels)

        best_train_features = selectKBest.transform(train_features)



        idxs_selected = selectKBest.get_support(indices=True)

        best_train_features = train_features.iloc[:,idxs_selected]

if enable_univariate_selec:

    if method_selec=='selectkbest':

        print(best_train_features.columns) # selected columns
if enable_univariate_selec:

    if method_selec=='selectkbest':

        best_train_features.describe()
if enable_univariate_selec:

    if method_selec=='selectkbest':

        new_corr_matrix = best_train_features.corr()

        dendrogram(new_corr_matrix)

        plt.tight_layout()
if enable_correlation_selec:

    new_new_corr_matrix = corr_drop(new_corr_matrix, factor)

    print(f'Number of features selected is {len(new_new_corr_matrix.columns)}.')
if enable_correlation_selec:

    dendrogram(new_new_corr_matrix)

    plt.tight_layout()
if enable_correlation_selec:

    fig, ax = plt.subplots(figsize=(10,10))

    ax = sns.heatmap(new_new_corr_matrix, square=True, cmap='Purples', ax=ax)

    plt.tight_layout()

    plt.show()
if enable_correlation_selec:

    best_train_features = best_train_features.loc[:,new_new_corr_matrix.columns]
if method_selec=='pca':

    pca = PCA(pca_variance)

    pca.fit(train_features)

    best_train_features = pca.transform(train_features)

    best_train_features = pd.DataFrame(best_train_features)
if method_selec=='pca':

    print('Number of components {pca.n_components_}')
# to enable run without balancing

best_train_features_balanced = best_train_features

train_labels_balanced = train_labels
if enable_balancing:

    idxs_pos = train_labels[train_labels==1].index

    idxs_neg = train_labels[train_labels==0].sample(n=number_samples, replace=False, random_state=0).index

    idxs_balanced = np.concatenate((idxs_pos,idxs_neg))

    best_train_features_balanced = best_train_features.loc[idxs_balanced]

    train_labels_balanced = train_labels.loc[idxs_balanced]

    print(f'Proportion balanced: {int(number_samples/1000)}/1')
if ml_method=='gradientboosting':

    gbc = GradientBoostingClassifier(loss=gbc_loss, random_state=0)

    if not enable_cv:

        gbc.fit(best_train_features_balanced, train_labels_balanced)
if ml_method=='gradientboosting' and enable_cv:

    #Seleciona os parâmetros do GB que deseja testar

    params = [{'loss': ['deviance', 'exponential']}]

    

    #Executa grid search com cross validation

    gbcc = GridSearchCV(gbc, params, cv=5, scoring='recall', verbose=10, n_jobs=3)

    gbcc.fit(best_train_features_balanced, train_labels_balanced)

    gbc = gbcc

    
if ml_method=='gradientboosting':

    display(gbc)
if ml_method=='randomforestreg':

    rfc = RandomForestRegressor(n_estimators=100, oob_score = True, random_state=0, n_jobs=3)

    rfc.fit(best_train_features_balanced, train_labels_balanced)

    print(rfc)
if ml_method=='randomforest':

    rfc = RandomForestClassifier(criterion=rfc_criterion, random_state=0)

    if not enable_cv:

        rfc.fit(best_train_features_balanced, train_labels_balanced)

    
if ml_method=='randomforest' and enable_cv:

    #Seleciona os parâmetros do GB que deseja testar

    params = [{'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2'], 'n_estimators': [10, 100]}]

    rfc = RandomForestClassifier(random_state=0)

    #Executa grid search com cross validation

    rfcc = GridSearchCV(rfc, params, cv=5, scoring='recall', verbose=10, n_jobs=3)

    rfcc.fit(best_train_features_balanced, train_labels_balanced)

    rfc = rfcc
if ml_method=='randomforest':

    display(rfcc)
if ml_method=='svm':

    #Seleciona os parâmetros da SVM que deseja testar

    params = [{'kernel': ['rbf'], 'gamma': [0.01], 'C': [0.001, 0.01, 0.1, 1, 10]}, 

              {'kernel': ['linear'], 'gamma': [0.01],  'C':  [0.001, 0.01, 0.1, 1, 10]}

             ]

    #Executa grid search com cross validation

    svmc = GridSearchCV(svm.SVC(C=1), params, cv=5, scoring='recall', verbose=10, n_jobs=3)

    svmc.fit(best_train_features_balanced, train_labels_balanced)
# to enable change feature selection method

best_test_features = test_features  
if enable_normalization:

    scaler.transform(best_test_features)

    best_test_features = pd.DataFrame(scaler.transform(best_test_features), columns=best_test_features.columns)
if enable_univariate_selec:

    if method_selec=='selectkbest':        

        X = selectKBest.transform(best_test_features)

        idxs_selected = selectKBest.get_support(indices=True)

        best_test_features = best_test_features.iloc[:,idxs_selected]

    if method_selec=='pca':        

        best_test_features = pca.transform(best_test_features)

if enable_correlation_selec:

    best_test_features = best_test_features.loc[:,new_new_corr_matrix.columns]
test_labels = test_labels.apply(round)

test_labels = test_labels.replace({-1:0})
if ml_method=='gradientboosting':

    y_pred = gbc.predict(best_test_features)

    report = classification_report(test_labels, y_pred)

    print(report)
if ml_method=='randomforestreg':

    y_pred = rfc.predict(best_test_features)

    y_pred = np.round(y_pred)

    report = classification_report(test_labels, y_pred)

    print(report)
if ml_method=='randomforest':

    y_pred = rfc.predict(best_test_features)

    report = classification_report(test_labels, y_pred)

    print(report)
if ml_method=='svm':

    y_pred = svmc.predict(best_test_features)

    report = classification_report(test_labels, y_pred)

    print(report)
cm = confusion_matrix(test_labels, y_pred).ravel()

cm = pd.DataFrame(cm.reshape((1,4)), columns=['tn', 'fp', 'fn', 'tp'])

display(cm)
total_cost = 10*cm.fp + 500*cm.fn

def printmd(string):

    display(Markdown(string))

printmd(f'Total cost is: \n# <p><span style="color:purple">${float(total_cost.values[0])}</span></p>')
